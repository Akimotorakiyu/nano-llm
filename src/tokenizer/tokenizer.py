from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_TOKENIZER_PATH = Path(__file__).parent / "mini_mind"


class NanoTokenizer:
    def __init__(self, tokenizer_path=None):
        path = tokenizer_path or DEFAULT_TOKENIZER_PATH
        print(f"Initializing NanoTokenizer from {path}...")
        self.mini_mind_tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Tokenizer loaded. Vocabulary size: {self.mini_mind_tokenizer.vocab_size}")

    def __call__(
        self, text, add_special_tokens=False, max_length=None, truncation=False
    ):
        return self.mini_mind_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            return_tensors="pt",
        )

    def tokenizer(self, text):
        inputs = self.mini_mind_tokenizer(text, return_tensors="pt")
        token_ids = inputs["input_ids"]
        return token_ids

    def decode(self, token_ids):
        if token_ids.dim() > 1:
            token_ids = token_ids[0]
        return self.mini_mind_tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def bos_token_id(self):
        return self.mini_mind_tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.mini_mind_tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.mini_mind_tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return self.mini_mind_tokenizer.vocab_size


if __name__ == "__main__":
    tokenizer = NanoTokenizer()
    text = "Hello, how are you?"
    token_ids = tokenizer.tokenizer(text)
    print("Token IDs:", token_ids)

    decoded_text = tokenizer.decode(token_ids)
    print("Decoded Text:", decoded_text)

    print("Vocabulary Size:", tokenizer.vocab_size)
