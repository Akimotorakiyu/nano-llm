import torch
from .model import NanoLLM, NanoLLMConfig
from .dataset import startToken, endToken


class Chat:
    def __init__(self, model: NanoLLM):
        self.model = model
        self.model.eval()

    def chat(self, prompt: str, max_length: int = 100) -> str:
        input_ids = [startToken.item()] + [ord(c) for c in prompt]
        input_ids = torch.tensor([input_ids])

        generated = input_ids.clone()

        for _ in range(max_length):
            output = self.model(generated)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == endToken.item():
                break

        input_len = input_ids.shape[1]
        output_ids = generated[0, input_len:].tolist()
        if startToken.item() in output_ids:
            output_ids = output_ids[output_ids.index(startToken.item()) + 1 :]
        if endToken.item() in output_ids:
            output_ids = output_ids[: output_ids.index(endToken.item())]

        return "".join(chr(c) for c in output_ids if c < 128)


def main():
    config = NanoLLMConfig()
    model = NanoLLM(config)

    checkpoint_path = "checkpoints/last.pt"

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded model from epoch {checkpoint.get('epoch', '?')}, loss: {checkpoint.get('loss', '?')}"
        )
    except (FileNotFoundError, RuntimeError, KeyError):
        print("No checkpoint found, using untrained model")

    chat = Chat(model)

    print("Chat started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat.chat(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
