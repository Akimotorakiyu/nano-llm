from src.model import NanoConfig, NanoLLM
from src.trainer import Trainer
from src.tokenizer.tokenizer import NanoTokenizer
from src.dataset import NanoDataset


def main():
    print("Hello from nano-llm!")
    # 初始化分词器
    tokenizer = NanoTokenizer()
    # 加载数据集
    dataset = NanoDataset("./data/pretrain_t2t_mini.jsonl", tokenizer)
    print("Dataset length:", len(dataset))
    print("First item tokenized:", dataset[0])

    config = NanoConfig(
        vocab_size=tokenizer.vocab_size,
    )

    model = NanoLLM(config)
    from src.dataloader import NanoDataLoader

    dataloader = NanoDataLoader(dataset, batch_size=8)
    trainer = Trainer(model, dataloader)
    trainer.train(epochs=10)


if __name__ == "__main__":
    main()
