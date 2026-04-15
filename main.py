from src.train import Train
from src.model import NanoLLMConfig, NanoLLM
from src.dataset import NanoDataSet
from src.dataloader import NanoDataLoader


def main():
    dataset = NanoDataSet()
    dataLoader = NanoDataLoader(dataset)
    config = NanoLLMConfig()
    llm = NanoLLM(config)
    train = Train(llm)

    train.train(dataLoader,32)


if __name__ == "__main__":
    main()
