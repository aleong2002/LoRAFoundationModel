from torch.utils.data import DataLoader


class Preprocessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess(self, example):
        tokenizer = self.tokenizer
        masked_input = example["input"].replace("[MASK]", tokenizer.mask_token)  # Just in case
        inputs = tokenizer(masked_input, padding="max_length", truncation=True, max_length=64)

        try:
            mask_index = inputs["input_ids"].index(tokenizer.mask_token_id)
        except ValueError:
            inputs["labels"] = [-100] * len(inputs["input_ids"])
            return inputs

        target_id = tokenizer.convert_tokens_to_ids(example["target"])
        labels = [-100] * len(inputs["input_ids"])
        labels[mask_index] = target_id
        inputs["labels"] = labels
        return inputs

    def get_dataloader(self, masked_dataset):
        tokenized_train = masked_dataset["train"].map(self.preprocess, batched=False)
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dataloader = DataLoader(tokenized_train, batch_size=2, shuffle=True)
        return dataloader
