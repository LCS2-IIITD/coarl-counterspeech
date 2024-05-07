import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_INPUT_LENGTH = 512


class CI_Pipeline:
    def __init__(self, model_path, use_cuda=False):
        self.use_cuda = use_cuda
        self.model_path = model_path

        # Load model configuration
        self.config = self.load_config()
        self.num_labels = len(self.config["id2label"])

        # Print model configuration
        print("-" * 100)
        print(f"Model ID to Label Mapping: {self.config['id2label']}")
        print("-" * 100)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, device_map="auto"
        )
        self.model = self.load_model()

        # Transfer model to GPU if CUDA is available
        if self.use_cuda:
            self.model.to("cuda")

    def load_config(self):
        """Load configuration from the model directory."""
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, "r") as file:
            return json.load(file)

    def load_model(self):
        """Load the pre-trained model with the appropriate configuration."""
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=self.num_labels,
            id2label=self.config["id2label"],
        )

    def __call__(self, inputs, inference_type="cat_acc"):
        """Process inputs and return model predictions based on inference type."""
        processed_inputs = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            return_tensors="pt",
        )

        # Transfer tensors to GPU if CUDA is available
        if self.use_cuda:
            processed_inputs.to("cuda")

        outputs = self.model(**processed_inputs)

        if inference_type == "cat_acc":
            return self.extract_category_accuracy(outputs)
        else:
            return self.extract_logits_mapping(outputs)

    def extract_category_accuracy(self, outputs):
        """Extract category accuracy from model outputs."""
        predictions = outputs.logits.argmax(dim=-1)
        return [self.config["id2label"][str(id)] for id in predictions.tolist()]

    def extract_logits_mapping(self, outputs):
        """Extract logits and their mapping from model outputs."""
        logits = outputs.logits
        return logits, [
            {
                "label": self.config["id2label"][str(index)],
                "score": logits.T[int(index)],
            }
            for index in self.config["id2label"]
        ]
