import torch
import torch.nn.functional as F


def create_causal_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask


seq_len = 5
mask = create_causal_mask(seq_len)
print("Causal mask:")
print(mask)

d_k = 64
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)

scores = (Q @ K.T) / (d_k ** 0.5)
scores_masked = scores + mask
attention_weights = F.softmax(scores_masked, dim=-1)

print("\nAttention weights after masking:")
print(attention_weights)
