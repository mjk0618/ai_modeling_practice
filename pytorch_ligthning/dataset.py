from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data["ko"].iloc[idx]
        target_text = self.data["en"].iloc[idx]

        source_tokenized = self.tokenize_text(source_text)
        target_tokenized = self.tokenize_text(target_text)

        return {
            "input_ids": source_tokenized["input_ids"],
            "attention_mask": source_tokenized["attention_mask"],
            "labels": target_tokenized["input_ids"]
        }

    def tokenize_text(self, text):
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True
        )
        
        return {
            key: value.squeeze(dim=0) for key, value in tokenized_text.items()
        }