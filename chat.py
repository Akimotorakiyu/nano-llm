import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.model import NanoLLM, NanoConfig
from src.tokenizer.tokenizer import NanoTokenizer


def load_model(checkpoint_path, vocab_size):
    config = NanoConfig(vocab_size=vocab_size)
    model = NanoLLM(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def chat(model, tokenizer, max_length=100, temperature=1.0, top_k=50):
    print("=" * 50)
    print("Chat Program - Type 'exit' to quit")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        input_ids = tokenizer.tokenizer(user_input).squeeze(0)
        # 添加 BOS token，与训练数据格式一致
        input_ids = torch.cat([torch.tensor([tokenizer.bos_token_id], dtype=torch.long), input_ids], dim=0)

        generated_ids = []

        for _ in range(max_length):
            with torch.no_grad():
                outputs = model(input_ids.unsqueeze(0))
                logits = outputs[:, -1, :] / temperature

                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    probs = torch.softmax(values, dim=-1)
                    chosen_idx = torch.multinomial(probs.squeeze(0), 1)
                    next_token = indices[0, chosen_idx.item()].item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs.squeeze(0), 1).item()

                # 如果还没生成任何内容就遇到 EOS，跳过继续生成
                if next_token == tokenizer.eos_token_id and len(generated_ids) == 0:
                    continue

                generated_ids.append(next_token)

                input_ids = torch.cat([input_ids, torch.tensor([next_token], dtype=torch.long)], dim=0)

                if next_token == tokenizer.eos_token_id:
                    break

        response_ids = torch.tensor(generated_ids).unsqueeze(0)
        response = tokenizer.decode(response_ids)

        print(f"Bot: {response}")


if __name__ == "__main__":
    checkpoint_path = "checkpoints/nano_llm_last.pth"

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first!")
        sys.exit(1)

    tokenizer = NanoTokenizer()

    print("Loading model...")
    model = load_model(checkpoint_path, vocab_size=tokenizer.vocab_size)
    print("Model loaded successfully!")

    chat(model, tokenizer)
