from transformers import BertTokenizer, RobertaTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("../data/public_models/bert_base")
    tokenizer.add_tokens(["[asda]","[ASDF]"])
    tokenizer.save_pretrained(".")
