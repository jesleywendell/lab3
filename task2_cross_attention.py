import torch
import torch.nn.functional as F


def cross_attention(encoder_out, decoder_state):
    d_model = encoder_out.shape[-1]

    W_q = torch.randn(d_model, d_model)
    W_k = torch.randn(d_model, d_model)
    W_v = torch.randn(d_model, d_model)

    Q = decoder_state @ W_q
    K = encoder_out @ W_k
    V = encoder_out @ W_v

    scores = (Q @ K.transpose(-2, -1)) / (d_model ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = weights @ V

    return output, weights


batch_size = 1
seq_len_french = 10
seq_len_english = 4
d_model = 512

encoder_output = torch.randn(batch_size, seq_len_french, d_model)
decoder_state = torch.randn(batch_size, seq_len_english, d_model)

output, weights = cross_attention(encoder_output, decoder_state)

print("encoder_output shape:", encoder_output.shape)
print("decoder_state shape:", decoder_state.shape)
print("cross-attention output shape:", output.shape)
print("attention weights shape:", weights.shape)
print("\nAttention weights (first head, first batch):")
print(weights[0])
