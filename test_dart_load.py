from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "./dataset/dart_masked_train.json",
    "validation": "./dataset/dart-v1.1.1-full-dev.json",
    "test": "./dataset/dart_masked_test.json"
})

train_data = dataset["train"]
print(train_data[1])