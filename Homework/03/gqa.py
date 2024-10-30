import torch
import torch.nn.functional as F


def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """
    num_heads = query.size(2)
    num_kv_heads = key.size(2)
    seq_len = query.size(1)
    if num_heads < num_kv_heads:
        raise ValueError
    mul = num_heads//num_kv_heads
    batch_size, kv_seq_len, num_kv_heads, hidden_dim = key.shape
    A = torch.zeros((batch_size, num_heads, seq_len, kv_seq_len))
    H = torch.zeros((batch_size, seq_len, num_heads, hidden_dim))
    for k in range(mul):
        sub_q = query[:,:,k::mul]
        if is_causal:
            tmp = torch.matmul(sub_q.transpose(1,2), key.transpose(1,2).transpose(2,3))/ hidden_dim ** 0.5
            causal_mask = torch.triu(torch.ones(batch_size, num_kv_heads, seq_len, kv_seq_len), diagonal=1).bool()
            tmp = tmp.masked_fill(causal_mask, float('-inf'))
            attention_scores = F.softmax(tmp, dim=-1).tril()
        else:
            attention_scores = F.softmax(torch.matmul(sub_q.transpose(1,2), key.transpose(1,2).transpose(2,3)) / hidden_dim ** 0.5, dim=-1)
        attention_output = torch.matmul(attention_scores, value.transpose(1,2))
        H[:,:,k::mul,:] = attention_output.transpose(1,2).clone()
        A[:,k::mul,:,:] = attention_scores.clone()
    if need_weights:
        return H, A
    return H
