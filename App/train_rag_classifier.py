from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
import numpy as np

# 1. Load and prepare dataset
dataset = load_dataset("databricks/databricks-dolly-15k")


# Filter and prepare classification labels
def map_labels(example):
    # Label 1 if context is present (needs retrieval), 0 otherwise
    example["label"] = 1 if example["context"] else 0
    return example


dataset = dataset.map(map_labels)

# Balance classes using undersampling
label_counts = dataset["train"].to_pandas()["label"].value_counts()
minority_class = label_counts.idxmin()
minority_count = label_counts.min()
majority_count = label_counts.max()

# Get indices for both classes
minority_indices = [
    i for i, x in enumerate(dataset["train"]) if x["label"] == minority_class
]
majority_indices = [
    i for i, x in enumerate(dataset["train"]) if x["label"] != minority_class
]

# Randomly select majority indices to match minority count
np.random.seed(42)
undersampled_majority = np.random.choice(
    majority_indices, size=minority_count, replace=False
)

# Combine and shuffle indices
balanced_indices = np.concatenate([minority_indices, undersampled_majority])
np.random.shuffle(balanced_indices)

# Create balanced dataset
balanced_dataset = dataset["train"].select(balanced_indices)

# Split dataset with stratification
dataset = balanced_dataset.train_test_split(test_size=0.2, stratify_by_column="label")

# 2. Initialize ModernBERT model and tokenizer
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "No_Retrieval", 1: "Needs_Retrieval"},
    trust_remote_code=True,
)


# 3. Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["instruction"], truncation=True, max_length=512, padding="max_length"
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Training setup
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )


training_args = TrainingArguments(
    output_dir="rag_classifier",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    logging_dir="./logs",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 5. Execute training
trainer.train()

# Save final model
trainer.save_model("best_rag_classifier")
tokenizer.save_pretrained("best_rag_classifier")
