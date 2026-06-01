import torch
from .model import NanoLLM, NanoLLMConfig
from .dataset import startTokenId, endTokenId


class Chat:
    def __init__(self, model: NanoLLM):
        self.model = model
        self.model.eval()

    def chat(self, prompt: str, max_length: int = 100) -> str:
        """
        基于新架构的chat：逐个token推理，维护状态s
        """
        with torch.no_grad():
            # 初始化状态
            s = None

            # 编码输入prompt，逐个token处理
            input_ids = [startTokenId] + [ord(c) for c in prompt]

            # 逐个处理输入token，更新状态
            for token_id in input_ids:
                x = torch.tensor([token_id])  # (1,)
                _, s = self.model(x, s)

            # 开始生成
            generated_tokens = []

            for _ in range(max_length):
                # 使用最后一个输入的token继续生成
                last_token = torch.tensor([input_ids[-1]]) if not generated_tokens else torch.tensor([generated_tokens[-1]])

                # 前向传播获取预测
                logits, s = self.model(last_token, s)

                # 取概率最高的token
                next_token = logits.argmax(dim=-1).item()

                # 检查结束符
                if next_token == endTokenId:
                    break

                generated_tokens.append(next_token)
                input_ids.append(next_token)

            # 解码输出（过滤特殊token）
            output_chars = []
            for token_id in generated_tokens:
                if token_id < 128 and token_id not in [startTokenId, endTokenId]:
                    output_chars.append(chr(token_id))

            return "".join(output_chars)

    def chat_efficient(self, prompt: str, max_length: int = 100) -> str:
        """
        更高效的版本：先用prompt建立状态，然后生成
        """
        with torch.no_grad():
            # 初始化状态
            s = None

            # 处理prompt建立状态
            sequence = [startTokenId] + [ord(c) for c in prompt]

            for token_id in sequence[:-1]:  # 处理到倒数第二个token
                x = torch.tensor([token_id])
                _, s = self.model(x, s)

            # 从最后一个token开始生成
            current_token = sequence[-1]
            generated = []

            for _ in range(max_length):
                x = torch.tensor([current_token])
                logits, s = self.model(x, s)
                next_token = logits.argmax(dim=-1).item()

                if next_token == endTokenId:
                    break

                generated.append(next_token)
                current_token = next_token

            return "".join(chr(c) for c in generated if c < 128)


def main():
    config = NanoLLMConfig()
    model = NanoLLM(config)

    checkpoint_path = "checkpoints/last.pt"

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded model from epoch {checkpoint.get('epoch', '?')}, loss: {checkpoint.get('loss', '?')}"
        )
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        print(f"No checkpoint found or error loading: {e}, using untrained model")

    chat = Chat(model)

    print("Chat started. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = chat.chat(user_input)
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
