import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch.nn.functional as F

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    from flash_attn import flash_attn_varlen_func, flash_attn_func
except ImportError:
    print("WARNING: flash_attention not found")

@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    vocab_size: int = 128
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-6
    max_seq_len: int = 2048
    dropout: float = 0.0
    flash_attention: bool = False

class Llama2P(nn.Module):
    """
    Wrapper to be able to use Llama2 model implemented in PyTorch.

    Parameters
    ----------
    transformers_config : list
        Model configuration.
    """

    def __init__(self, transformers_config):
        super(Llama2P, self).__init__()

        config = ModelArgs()
        config.vocab_size = transformers_config.vocab_size
        config.max_seq_len = transformers_config.n_positions
        config.n_heads = transformers_config.n_head
        config.n_kv_heads = transformers_config.n_head # For MHA   
        config.n_layers = transformers_config.n_layer
        config.dim = transformers_config.n_embd
        config.hidden_dim = transformers_config.n_embd * 4
        config.dropout = transformers_config.attn_pdrop
        config.flash_attention = transformers_config.flash_attention

        self.config = config
        self.feature_extractor = Transformer(config)

    def forward(self, batch):
        """
        Forward pass Neural Network.

        Parameters
        ----------
        batch : torch.Tensor
            Input data.

        Returns
        -------
        out : torch.Tensor
            Output feature map.
        """

        # Forward pass
        out = self.feature_extractor(
            tokens=batch.long(),
            attention_mask=(batch != 0.0).long(),  # Shape (batch_size, sequence_length)
        )

        return out

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def precumpute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) # type: ignore
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcast
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply rotary using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleace(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_kv_heads must divide n_heads"
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or manual implementation
        self.flash, self.sdpa = False, False
        if args.flash_attention:
            self.flash = True
        elif hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            self.sdpa = True
        else:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)
    
    def forward(
            self,
            x: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep) # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep) # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # mask for attention
        causal_mask = self.mask[:, :, :seqlen, :seqlen]
        if attention_mask is not None and not self.flash:
            attention_mask = (attention_mask.float() - 1)[:, None, None, :]
            attention_mask.masked_fill_(attention_mask != 0, float("-inf"))
            causal_mask = causal_mask + attention_mask

        # flash implementation
        if self.flash:
            query_states = xq.transpose(1,2)
            key_states = xk.transpose(1,2)
            value_states = xv.transpose(1,2)
            if attention_mask is not None:
                query_states, key_states, value_states, indices_q, cu_seqlens, max_seqlen_in_batch = self._unpad_input(
                    query_states, key_states, value_states, attention_mask, seqlen
                )

                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seqlen_in_batch
                output = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q = cu_seqlens_q,
                    cu_seqlens_k = cu_seqlens_k,
                    max_seqlen_q = max_seqlen_in_batch_q,
                    max_seqlen_k = max_seqlen_in_batch_k,
                    dropout_p = self.dropout if self.training else 0.0,
                    causal=False)
                output = pad_input(output, indices_q, bsz, seqlen).transpose(1,2)
            else:
                output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout_p = self.dropout if self.training else 0.0,
                    causal=True)
                output = output.transpose(1,2)

        elif self.sdpa:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=causal_mask, dropout_p = self.dropout if self.training else 0.0,
                is_causal=False)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask"), "mask not found"
            scores = scores + causal_mask # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv) # (bs, n_local_heads, seqlen, head_dim)
        
        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
    
    def _unpad_input(self,query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = self._get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_local_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
    
    def _get_unpad_data(self, attention_mask):
        seqlens_in_batch = attention_mask.sum(-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1,0))
        return (
            indices,
            cu_seqlens,
            max_seqlen_in_batch
        )
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # order to match HF
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, attention_mask=None):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, attention_mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    

class Transformer(nn.Module):
    last_loss = Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        
        # TODO:share the unembedding parameters with the embedding parameters
        
        # some useful precompute for the RoPE relative positional embeddings
        freq_cos, freq_sin = precumpute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freq_cos", freq_cos, persistent=False)
        self.register_buffer("freq_sin", freq_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freq_cos[:seqlen]
        freqs_sin = self.freq_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, attention_mask)
        h = self.norm(h)
        return h