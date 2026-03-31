from model import NanoConfig, NanoLLM
from src.trainer import NanoTrainer
from src.tokenizer.tokenizer import NanoTokenizer
from src.dataset import NanoDataset

def main():
    print("Hello from nano-llm!")
    # 初始化分词器
    tokenizer = NanoTokenizer()
    # 加载数据集
    dataset = NanoDataset("./data/sample.json", tokenizer)
    print("Dataset length:", len(dataset))
    print("First item tokenized:", dataset[0])

    config = NanoConfig(
        vocab_size=tokenizer.vocab_size(),
        max_seq_len=512,
        num_layers=4,
        num_heads=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1
    )

    model = NanoLLM(config)
    trainer = NanoTrainer(model, dataset)
    trainer.train(epochs=10)

if __name__ == "__main__":
    main()
