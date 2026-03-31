from transformers import AutoTokenizer

path = "./src/tokenizer/mini_mind"  # 预训练分词器的路径

class NanoTokenizer:
    def __init__(self, vocab_size=3000):
        self.vocab_size = vocab_size
        print(f"Initializing NanoTokenizer...")
        self.mini_mind_tokenizer = AutoTokenizer.from_pretrained(path)  # 加载预训练的分词器
        print(f"Tokenizer loaded. Vocabulary size: {self.mini_mind_tokenizer.vocab_size}")

    def tokenizer(self, text):
        # 使用预训练的分词器进行分词
        inputs = self.mini_mind_tokenizer(text, return_tensors="pt")
        token_ids = inputs["input_ids"] # 获取分词后的 token IDs
        return token_ids
    
    def decode(self, token_ids):
        # 使用预训练的分词器进行反向分词
        text = self.mini_mind_tokenizer.decode(token_ids[0], skip_special_tokens=True)
        return text
    
    def vocab_size(self):
        return self.mini_mind_tokenizer.vocab_size
    

if __name__ == "__main__":
    tokenizer = NanoTokenizer()
    text = "Hello, how are you?"
    token_ids = tokenizer.tokenizer(text)
    print("Token IDs sssss:", token_ids)
    
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded Text:", decoded_text)

    print("Vocabulary Size:", tokenizer.vocab_size())