from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "./dataset/dart-v1.1.1-full-train.json",
    "validation": "./dataset/dart-v1.1.1-full-dev.json",
    "test": "./dataset/dart-v1.1.1-full-test.json"
})

train_data = dataset["train"]
print(train_data[0])