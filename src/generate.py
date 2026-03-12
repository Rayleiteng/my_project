import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from data_loader import get_math_dataloader

from train import MathGPT


def generate_math_formula(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)

    print(f"\n (Prompt is): {prompt}")
    print(" thinking...\n")
    print(prompt, end="")

    generated_ids = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            cond_input = input_ids[:, -128:]

            logits = model(cond_input)

            next_token_logits = logits[0, -1, :]

            next_token_id = torch.argmax(next_token_logits).item()

            generated_ids.append(next_token_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=device)], dim=1
            )

            new_word = tokenizer.decode([next_token_id])
            print(new_word, end="", flush=True)

    print("\n\n finish!")

    return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file("../data/tokenizer_math.json")
    vocab_size = tokenizer.get_vocab_size()

    model = MathGPT(vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=4).to(
        device
    )

    weights_path = "../data/math_gpt_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    test_prompt_set = [
        r"||T|| = \sup_{||x||=1}",
        r"X:=S^{n-1} \geq",
        r"\sum_{i=0}^{n-1}t_i",
        r"b_i(X;\FF)\le",
        r"$d=n-1$ and $k\to \mathbb{E_1} \ldots \mathbb{R}$",
    ]
    for test_prompt in test_prompt_set:
        generate_math_formula(
            model, tokenizer, test_prompt, max_new_tokens=40, device=device
        )
        print("")
