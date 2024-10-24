import math
import torch
import torch.nn.functional as F


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    hidden = queries.shape[-1]
    attention_scores = F.softmax(torch.matmul(queries, torch.transpose(keys, 1,2))/hidden**0.5, dim=-1)
    attention_output = torch.matmul(attention_scores, values)
    return attention_output


def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    attentions = []
    # attentions = [compute_attention(q,k,v) for q,k,v in
    #               zip(torch.unbind(queries, 1), torch.unbind(keys, 1), torch.unbind(values, 1))]
    n_heads = queries.shape[1]
    for i in range(n_heads):
        attention = compute_attention(queries[:,i], keys[:,i], values[:,i])
        attentions.append(attention)

    result = torch.cat(attentions,dim=-1)
    ans = torch.matmul(result, torch.transpose(projection_matrix,0,1))
    return torch.matmul(result, torch.transpose(projection_matrix,0,1))


def compute_rotary_embeddings(x)-> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    d = x.shape[-1]
    seq = x.shape[1]
    Rs = torch.zeros((seq,d,d))
    for m in range(seq):
        Theta = [10000**(-2*(i-1)/d) for i in range(1, d//2+1)]
        blocks = [torch.tensor([[math.cos(m*Theta[i]), -math.sin(m*Theta[i])],[math.sin(m*Theta[i]), math.cos(m*Theta[i])]]) for i in range(d//2)]
        for j in range(d//2):
            Rs[m][j*2:(j+1)*2,j*2:(j+1)*2] = blocks[j]
        trans = torch.transpose(Rs[m], 0,1)
        Rs[m] = trans.clone()
    res = torch.matmul(x, Rs)
    return res
