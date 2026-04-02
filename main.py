from src.model import NanoConfig, NanoLLM
from src.trainer import NanoTrainer
from src.tokenizer.tokenizer import NanoTokenizer
from src.dataset import NanoDataset
from src.dataloader import NanoDataLoader

def main():
    print("Hello from nano-llm!")
    # 初始化分词器
    tokenizer = NanoTokenizer()
    # 加载数据集
    dataset = NanoDataset("./data/pretrain_t2t_mini.jsonl", tokenizer)

    config = NanoConfig(
        vocab_size=tokenizer.vocab_size,
    )

    model = NanoLLM(config)

    dataloader = NanoDataLoader(dataset, batch_size=8)
    trainer = NanoTrainer(model, dataloader)
    trainer.train(epochs=10)


if __name__ == "__main__":
    main()
