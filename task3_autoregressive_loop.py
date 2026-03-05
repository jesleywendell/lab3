import torch
import torch.nn.functional as F


VOCAB_SIZE = 10000
EOS_TOKEN = "<EOS>"
VOCAB = [f"word_{i}" for i in range(VOCAB_SIZE - 1)] + [EOS_TOKEN]
EOS_IDX = VOCAB_SIZE - 1

d_model = 512
batch_size = 1
seq_len_french = 10


def generate_next_token(current_sequence, encoder_out):
    seq_len = len(current_sequence)
    decoder_state = torch.randn(batch_size, seq_len, d_model)

    W_proj = torch.randn(d_model, VOCAB_SIZE)
    logits = decoder_state[:, -1, :] @ W_proj
    probs = F.softmax(logits, dim=-1)
    return probs.squeeze(0)


encoder_out = torch.randn(batch_size, seq_len_french, d_model)

current_sequence = ["<START>"]
max_steps = 20

print(f"Starting generation with: {current_sequence}")

while len(current_sequence) < max_steps:
    probs = generate_next_token(current_sequence, encoder_out)
    next_token_idx = torch.argmax(probs).item()
    next_token = VOCAB[next_token_idx]

    current_sequence.append(next_token)
    print(f"Step {len(current_sequence) - 1}: generated '{next_token}'")

    if next_token == EOS_TOKEN:
        break

print("\nFinal sequence:", " ".join(current_sequence))
