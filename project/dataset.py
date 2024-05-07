# Read and load dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset


class IntentConan(TorchDataset):
    def __init__(
        self, data, tokenizer, max_length, text_col, label_col, labels_map=None
    ) -> None:
        super(IntentConan, self).__init__()
        self.data = data
        # self.data.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
        self.data_size = self.data.shape[0]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = list(self.data["label"].unique())
        self.labels_map = labels_map if labels_map is not None else {}
        self.transform_label_col()

        print("\n\n******************* Dataset Stats *******************")
        print(f"Total dataset: {self.data_size}")
        print(f"Total unique labels: {len(self.labels)}")
        print(f"Labels: {self.labels_map}")

        self.encodings = self.calculate_encodings()

    def transform_label_col(self):
        if not self.labels_map:
            for label in self.labels:
                if label not in self.labels_map:
                    self.labels_map[label] = len(self.labels_map)

        self.data["label_encoded"] = self.data["label"].apply(
            lambda x: self.labels_map[x]
        )

    def calculate_encodings(self):
        encodings = self.tokenizer(
            self.texts(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encodings

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        label = torch.tensor(self.data["label_encoded"][idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }

    def __len__(self):
        return self.data_size

    def __num_labels__(self):
        return len(self.labels)

    def __data__(self):
        return self.data

    def labels_encoded(self):
        return self.data["label_encoded"].values.tolist()

    def texts(self):
        return self.data["text"].values.tolist()
