# A compact self-attention class
import torch.nn as nn

class SelfAttention_v1(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.W_query = nn.Parameter(torch.rand(d_in, d_out))
    self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
    self.W_value = nn.Parameter(torch.rand(d_in, d_out))

  def forward(self, x):
    keys        = x @ self.W_key
    queries     = x @ self.W_query
    values      = x @ self.W_value
    attn_scores = queries @ keys.T #omega
    attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim = -1
    )
    context_vec = attn_weights @ values
    return context_vec


class SelfAttention_v2(nn.Module):
  def __init__(self, d_in, d_out, qkv_bias=False):
    super().__init__()
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

  def forward(self, x):
    keys        = self.W_key(x)
    queries     = self.W_query(x)
    values      = self.W_value(x)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim = -1
    )
    context_vec = attn_weights @ values
    return context_vec


class CausalAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length,
               droupout, qkv_bias=False):
    super().__init__()
    self.d_out   = d_out
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout  = nn.Dropout(droupout)
    self.register_buffer(
        'mask',
        torch.triu(
            torch.ones(context_length, context_length),
            diagonal = 1
        )
    )

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys        = self.W_key(x)
    queries     = self.W_query(x)
    values      = self.W_value(x)

    attn_scores = queries @ keys.transpose(1, 2)
    attn_scores.masked_fill_(
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

    attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim = -1
    )
    attn_weights = self.dropout(attn_weights)
    context_vec = attn_weights @ values

    return context_vec


# A wrapper class to implement multi-head attention
class MultiHeadAttentionWrapper(nn.Module):
  def __init__(self, d_in, d_out, context_length,
               droupout, num_heads, qkv_bias=False):
    super().__init__()
    self.heads = nn.ModuleList(
        [CausalAttention(
            d_in, d_out, context_length, droupout, qkv_bias=qkv_bias
        )
        for _ in range(num_heads)]
    )

  def forward(self, x):
    return torch.cat([head(x) for head in self.heads], dim=-1)


# A more efficient multi-head attention class
class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out,
               context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
        "mask",
        torch.triu(torch.ones(context_length, context_length),
                   diagonal=1)
    )

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(
        b, num_tokens, self.num_heads, self.head_dim
    )

    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)

    attn_scores = queries @ keys.transpose(2, 3)
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

    attn_scores.masked_fill_(mask_bool, -torch.inf)

    attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim = -1
    )
    attn_weights = self.dropout(attn_weights)

    context_vec = (attn_weights @ values).transpose(1, 2)

    context_vec = context_vec.contiguous().view(
        b, num_tokens, self.d_out
    )
    context_vec = self.out_proj(context_vec)
    return context_vec













