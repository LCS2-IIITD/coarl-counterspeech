import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from rouge_score import rouge_scorer

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is supported by this system.")
    print(f"CUDA version: {torch.version.cuda}")

    # Set the CUDA device (you can choose a specific device ID)
    cuda_id = torch.cuda.current_device()
    torch.cuda.set_device(cuda_id)

    # Print GPU information
    print(f"ID of current CUDA device: {cuda_id}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
else:
    print("CUDA is not supported by this system. Training will be done on CPU.")


# Define the toy dataset
class IntentConanDataset(Dataset):
    def __init__(self, tokenizer, csv_file):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file)
        if "level_0" not in self.data.columns:
            self.data.reset_index(inplace=True)
        self.hatespeech_col = "hatespeech"
        self.csType_col = "csType"
        self.target_col = [
            "hatespeechOffensiveness",
            "targetGroup",
            "speakerIntent",
            "relevantPowerDynamics",
            "hatespeechImplication",
            "targetGroupEmotionalReaction",
            "targetGroupCognitiveReaction",
        ]
        self.input_col = [
            "prompt_offensiveness",
            "prompt_target_group",
            "prompt_speaker_intent",
            "prompt_power_dynamics",
            "prompt_implication",
            "prompt_emotional_reaction",
            "prompt_cognitive_reaction",
            "prompt_cs_generation",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_texts = self.data.iloc[idx][self.input_col].tolist()
        target_texts = self.data.iloc[idx][self.target_col].tolist()

        source_inputs_list = [
            self.tokenizer.encode_plus(
                source_text,
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt",
            )
            for source_text in source_texts
        ]

        target_inputs_list = [
            self.tokenizer.encode_plus(
                target_text,
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt",
            )
            for target_text in target_texts
        ]

        return {
            "source_texts": source_texts,
            "target_texts": target_texts,
            "source_ids_list": [
                source_inputs["input_ids"].squeeze()
                for source_inputs in source_inputs_list
            ],
            "source_mask_list": [
                source_inputs["attention_mask"].squeeze()
                for source_inputs in source_inputs_list
            ],
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

# Create the toy dataset and dataloader
train_file = "/home/amey/coarl-counterspeech/data/train.csv"
validation_file = "/home/amey/coarl-counterspeech/data/validation.csv"
df_train = IntentConanDataset(tokenizer, train_file)
df_validation = IntentConanDataset(tokenizer, validation_file)

# Split the dataset into training and validation sets
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(df_train, batch_size=16)
val_dataloader = DataLoader(df_validation, batch_size=16)
print(f"Dataloading Done")

# Set up the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Set up the ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Define the weights for each task
weights = [
    0.1,
    0.2,
    0.3,
    0.1,
    0.1,
    0.1,
    0.1,
]  # Modify this list according to your tasks

# Training loop
model.train()
print(f"Training Started")
for epoch in range(3):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        losses = []
        for i in range(len(batch["target_ids_list"])):
            outputs = model(
                input_ids=batch["source_ids_list"][i],
                attention_mask=batch["source_mask_list"][i],
                labels=batch["target_ids_list"][i],
                decoder_attention_mask=batch["target_mask_list"][i],
            )

            loss = loss_fn(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                batch["target_ids_list"][i].view(-1),
            )
            losses.append(loss)

        loss = sum(l * w for l, w in zip(losses, weights)) / sum(weights)
        loss.backward()
        print(f"Epoch {epoch + 1}/{3}, Training Loss: {loss:.4f}")

        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{3}, Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    print(f"Running Validation loop")
    with torch.no_grad():
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougel = 0
        for batch in val_dataloader:
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

        avg_rouge1 = total_rouge1 / len(val_dataloader)
        avg_rouge2 = total_rouge2 / len(val_dataloader)
        avg_rougel = total_rougel / len(val_dataloader)
        print(
            f"Validation ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougel:.4f}"
        )

    model.train()


# Save the model
torch.save(model.state_dict(), "multitask_t5_model.pt")