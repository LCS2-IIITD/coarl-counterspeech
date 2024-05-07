# %%

import pandas as pd
import time
import torch
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GenerationConfig, TrainingArguments, Trainer
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
import os
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)
from datasets import load_dataset, load_from_disk
from peft import PeftModel, PeftConfig, LoraConfig, TaskType

# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler

import torch
import evaluate

import numpy as np
import pandas as pd

# tqdm library makes the loops show a smart progress meter.
from tqdm import tqdm

tqdm.pandas()


os.environ["http_proxy"] = "http://xen03.iitd.ac.in:3128"
os.environ["https_proxy"] = "http://xen03.iitd.ac.in:3128"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


# %%
train_file = "/home/amey/coarl-counterspeech/data/train.csv"
validation_file = "/home/amey/coarl-counterspeech/data/validation.csv"
df_train = pd.read_csv(train_file)
df_val = pd.read_csv(validation_file)
src_col = "prompt_cs_generation"
trg_col = "counterspeech"

df_train = df_train[[src_col, trg_col]].dropna()
df_val = df_val[[src_col, trg_col]].dropna()

print(f"Train size: {df_train.shape}", f"Test size: {df_val.shape}")

sample_data = df_train.sample()
dash_line = "_".join(" " for x in range(100))

print(f"Input Text:\n{sample_data[src_col].iloc[0]}")
print(dash_line, "\n")
print(f"Groundtruth Counterspeech:\n{sample_data[trg_col].iloc[0]}")


data_dict_train = {
    "input_prompt": df_train[src_col].values.tolist(),
    "output_prompt": df_train[trg_col].values.tolist(),
}

data_dict_eval = {
    "input_prompt": df_val[src_col].values.tolist(),
    "output_prompt": df_val[trg_col].values.tolist(),
}

# Create a Dataset object
dataset_train = Dataset.from_dict(data_dict_train)
dataset_eval = Dataset.from_dict(data_dict_eval)

dataset = DatasetDict(
    {
        "train": dataset_train,
        "validation": dataset_eval,
    }
)

dataset

# %%
model_name = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

max_source_length = 1024
min_source_length = 64
max_target_length = 512
min_target_length = 32


def tokenize_function(sample):
    sample["input_ids"] = tokenizer.encode(sample["input_prompt"])
    sample["query"] = tokenizer.decode(sample["input_ids"])

    return sample


dataset_train_tokenized = dataset_train.map(tokenize_function, batched=False)
dataset_train_tokenized.set_format(type="torch")

dataset_eval_tokenized = dataset_eval.map(tokenize_function, batched=False)
dataset_eval_tokenized.set_format(type="torch")


# %%
# print(f"Input Prompt:")
# print(dataset_train_tokenized['input_prompt'][0])
# print(dash_line, "\n")
# print(f"Output Prompt:")
# print(dataset_train_tokenized['output_prompt'][0])

# print(f"Input Ids:")
# print(dataset_train_tokenized['input_ids'][0])
# print(dash_line, "\n")
# print(f"Groundtruth Ids:")
# print(dataset_train_tokenized['labels'][0])

print(f"Input:")
print(dataset_train_tokenized["input_ids"][0])
print(dash_line, "\n")
print(f"Query:")
print(dataset_train_tokenized["query"][0])

# %% [markdown]
# ### Load the PEFT-tuned model

# %%
peft_model_path = "/home/amey/coarl-counterspeech/checkpoints/best_model_small"

# %%
lora_config = LoraConfig(
    r=256,
    lora_alpha=512,
    target_modules=["q", "v"],
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

peft_model = PeftModel.from_pretrained(
    base_model,
    peft_model_path,
    lora_config=lora_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    is_trainable=True,
)

print(
    f"PEFT model parameters to be updated:\n{print_number_of_trainable_model_parameters(peft_model)}\n"
)

# %% [markdown]
# ### Initialize Reward Model

# %%
from reward_modelling import RewardModel

# Initialize the Reward Model
reward_model = RewardModel()

# Define the input data: hatespeech and predicted counterspeech
hatespeech = ["Muslims are terrorists", "Muslims are terrorists"]
predicted_counterspeech = [
    "Muslims are extremely peace loving people.",
    "I agree, Muslims are terrorists.",
]

# Print input data
print("Hatespeech inputs:")
for hs in hatespeech:
    print(f"- {hs}")
print("Counterspeech inputs:")
for cs in predicted_counterspeech:
    print(f"- {cs}")

# Compute the rewards and related scores
reward_scores, pc_scores, aq_scores, toxicity_scores = reward_model.compute_rewards(
    hatespeech, predicted_counterspeech
)

# Print output data
print("Computed Rewards:")
print(f"Reward Scores: {reward_scores}")
print(f"Pro-Con Scores: {pc_scores}")
print(f"Argument Quality Scores: {aq_scores}")
print(f"Toxicity Scores: {toxicity_scores}")

mean = np.mean(reward_scores)
std = np.std(reward_scores)
print(f"Mean reward score: {mean}")
print(f"Std reward score: {std}")

# %% [markdown]
# ### Initialize the PPO model

# %%
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    peft_model, is_trainable=True
)

print(
    f"PPO model parameters to be updated (ValueHead + 769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}\n"
)
print(ppo_model.v_head)

# %% [markdown]
# ### Create Reference Model for KL-Divergence

# %%
ref_model = create_reference_model(ppo_model)

print(
    f"Reference model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n"
)

# %% [markdown]
# ### PPO Trainer


# %%
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


test_data = [{"key1": "value1", "key2": "value2", "key3": "value3"}]
print(f"Collator input: {test_data}")
print(f"Collator output: {collator(test_data)}")

# %%
learning_rate = 1.41e-5
max_ppo_epochs = 1
mini_batch_size = 4
batch_size = 16

config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=ppo_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset_train_tokenized,
    data_collator=collator,
)

# %%
output_min_length = 30
output_max_length = 1024
output_length_sampler = LengthSampler(output_min_length, output_max_length)
max_ppo_steps = 10000

generation_kwargs = {"min_length": 30, "top_k": 0.0, "top_p": 1.0, "do_sample": True}

reward_kwargs = {
    "top_k": None,  # Return all scores.
    "function_to_apply": "none",  # You want the raw logits without softmax.
    "batch_size": 16,
}

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):

    # Break when you reach max_steps.
    if step >= max_ppo_steps:
        break

    prompt_tensors = batch["input_ids"]

    # Get response from FLAN-T5/PEFT LLM.
    counterspeech_tensors = []

    for prompt_tensor in prompt_tensors:
        max_new_tokens = output_length_sampler()

        generation_kwargs["max_new_tokens"] = max_new_tokens
        predicted_counterspeech = ppo_trainer.generate(
            prompt_tensor, **generation_kwargs
        )
        counterspeech_tensors.append(
            predicted_counterspeech.squeeze()[-max_new_tokens:]
        )

    # This needs to be called "response".
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in counterspeech_tensors]
    print(len(batch["response"]))
    # Compute reward outputs.
    query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
    queries = [q for q in batch["query"]]
    responses = [r for r in batch["response"]]
    # rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)
    rewards, pc_scores, aq_scores, toxicity_scores = reward_model.compute_rewards(
        hatespeech=queries, predicted_counterspeech=responses
    )

    # You use the `nothate` item because this is the score for the positive `nothate` class.
    reward_tensors = [torch.tensor(reward) for reward in rewards]

    # Run PPO step.
    stats = ppo_trainer.step(prompt_tensors, counterspeech_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
    print("-".join("" for x in range(100)))

# %%
ppo_model.save_pretrained("/checkpoints/ppo_flant5-xxl")
tokenizer.save_pretrained("/checkpoints/ppo_flant5-xxl")
