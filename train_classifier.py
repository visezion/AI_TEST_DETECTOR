import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from data_prep import load_ai_human_dataset
import config

train_df, val_df = load_ai_human_dataset()

tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL_NAME)

train_tokenized = tokenizer(
    train_df["text"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=config.MAX_SEQ_LENGTH,
)

val_tokenized = tokenizer(
    val_df["text"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=config.MAX_SEQ_LENGTH,
)


class DetectorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


model = AutoModelForSequenceClassification.from_pretrained(
    config.CLASSIFIER_MODEL_NAME,
    num_labels=2
)

train_dataset = DetectorDataset(train_tokenized, train_df["label"].tolist())
val_dataset = DetectorDataset(val_tokenized, val_df["label"].tolist())


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


training_args = TrainingArguments(
    output_dir="classifier_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("classifier_model")
tokenizer.save_pretrained("classifier_model")
