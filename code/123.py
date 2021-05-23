from transformers import BertTokenizer, RobertaTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(r"D:\public_models\bert_base")
    print(tokenizer.encode("hello, nice to meet you", add_special_tokens=False))
