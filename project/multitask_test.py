import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer


# Define the toy dataset
class IntentConanDataset(Dataset):
    def __init__(self, tokenizer, csv_file):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file)
        self.input_col = "hatespeech"
        self.target_col = [
            "hatespeechOffensiveness",
            "targetGroup",
            "speakerIntent",
            "relevantPowerDynamics",
            "hatespeechImplication",
            "targetGroupEmotionalReaction",
            "targetGroupCognitiveReaction",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # source_text = self.data.iloc[idx, 0]
        # target_texts = self.data.iloc[idx, 1:].tolist()
        source_text = self.data.iloc[idx][self.input_col]
        target_texts = self.data.iloc[idx][self.target_col].tolist()

        source_inputs = self.tokenizer.encode_plus(
            source_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        target_inputs_list = [
            self.tokenizer.encode_plus(
                target_text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            for target_text in target_texts
        ]

        return {
            "source_ids": source_inputs["input_ids"].squeeze(),
            "source_mask": source_inputs["attention_mask"].squeeze(),
            "target_ids_list": [
                target_inputs["input_ids"].squeeze()
                for target_inputs in target_inputs_list
            ],
            "target_mask_list": [
                target_inputs["attention_mask"].squeeze()
                for target_inputs in target_inputs_list
            ],
        }


# Initialize the T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

# Load the trained model
model.load_state_dict(torch.load("t5_model.pt"))

# Create the toy test dataset and dataloader
test_dataset = IntentConanDataset(
    tokenizer, "/home/ameyh/depository/Counter-Speech-Generation/data/m2/test.csv"
)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Set up the ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Test loop
model.eval()
with torch.no_grad():
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougel = 0
    predictions = []
    for batch in test_dataloader:
        preds_list = []
        for i in range(len(batch["target_ids_list"])):
            outputs = model.generate(
                input_ids=batch["source_ids"], attention_mask=batch["source_mask"]
            )
            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for g in outputs
            ]
            preds_list.append(preds)

        targets_list = [
            [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for g in batch["target_ids_list"][i]
            ]
            for i in range(len(batch["target_ids_list"]))
        ]

        for preds, targets in zip(preds_list, targets_list):
            for pred, target in zip(preds, targets):
                scores = scorer.score(target, pred)
                total_rouge1 += scores["rouge1"].fmeasure
                total_rouge2 += scores["rouge2"].fmeasure
                total_rougel += scores["rougeL"].fmeasure

                predictions.append((target, pred))

    avg_rouge1 = total_rouge1 / len(test_dataloader)
    avg_rouge2 = total_rouge2 / len(test_dataloader)
    avg_rougel = total_rougel / len(test_dataloader)
    print(f"Test ROUGE-1: {avg_rouge1}, ROUGE-2: {avg_rouge2}, ROUGE-L: {avg_rougel}")

# Save the predictions to a CSV file using pandas
df = pd.DataFrame(predictions, columns=["Target", "Prediction"])
df.to_csv("predictions.csv", index=False)
