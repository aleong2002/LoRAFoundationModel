from torch.utils.data import DataLoader

# code references used:
# code reference: https://anirbansen2709.medium.com/finetuning-llms-using-lora-77fb02cbbc48
# huggingface docs: https://huggingface.co/docs

class Preprocessor:

    def __init__(self, tokenizer, max_length: int = 64, batch_size: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def _build_labels(self, input_ids, target_str):
        
        try:
            mask_index = input_ids.index(self.tokenizer.mask_token_id)
        except ValueError:
            return [-100] * len(input_ids)

        target_str = target_str.strip()
        if not target_str:
            target_word = self.tokenizer.unk_token
        else:
            target_word = target_str.split()[0]  # word-level

        target_ids = self.tokenizer(
            target_word,
            add_special_tokens=False
        )["input_ids"]

        if not target_ids:
            label_id = self.tokenizer.unk_token_id
        else:
            label_id = target_ids[0]

        labels = [-100] * len(input_ids)
        labels[mask_index] = label_id
        return labels

    def preprocess(self, example):
       
        tokenizer = self.tokenizer

        raw_input = example["input"]
        target = example["target"]

        parts = raw_input.split("[MASK]")
        if len(parts) > 2:
            raw_input = "[MASK]".join(parts[:2])

        masked_text = raw_input.replace("[MASK]", tokenizer.mask_token)

        # tokenize input
        enc = tokenizer(
            masked_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        input_ids = enc["input_ids"]
        labels = self._build_labels(input_ids, target)

        enc["labels"] = labels
        return enc

    def get_dataloader(self, masked_dataset):
        
        tokenized_train = masked_dataset["train"].map(
            self.preprocess,
            batched=False
        )

        # use hugging face format
        tokenized_train.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        dataloader = DataLoader(
            tokenized_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        return dataloader
