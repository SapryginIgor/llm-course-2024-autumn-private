import torch


def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Compute ALiBi for a sequence.

    ALiBi can be used not only with causal models.
    In this case, the biases will be symmetrical about the diagonal up to the sign.

    Args:
        num_heads (int): Number of attention heads.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: A tensor containing ALiBi to be added to attention scores.
    """
    fir = torch.arange(1, seq_len+1).unsqueeze(1)
    sec = torch.arange(1, seq_len+1).unsqueeze(0)
    dist =  (-abs(fir - sec).tril()).unsqueeze(0).float()
    dist = dist - torch.transpose(dist, 1, 2)
    answer = torch.clone(dist.expand(num_heads,seq_len,seq_len))
    powers = -torch.linspace(8/num_heads, 8, num_heads)
    m = torch.pow(2, powers)
    answer *= m.view(-1,1,1)
    return answer



if __name__ == "__main__":
    bias = compute_alibi(4, 5)
    print(bias)
