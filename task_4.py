import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AdamW
from task_2 import MultiTaskTextModel

# Adding placeholder datasets for classification and NER tasks
classification_data = []
ner_data = []

# Creating DataLoaders for batching and shuffling
classification_loader = DataLoader(classification_data, batch_size=8, shuffle=True)
ner_loader = DataLoader(ner_data, batch_size=8, shuffle=True)

# Initializing multi-task model with transformer backbone and label definitions
multi_task_model = MultiTaskTextModel(
    model_name="distilbert-base-uncased",
    classification_labels=["negative", "neutral", "positive"],
    ner_labels=["O", "PER", "ORG", "LOC", "MISC"]
)

# Moving model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_task_model.to(device)

# Defining loss functions for each task
classification_criterion = nn.CrossEntropyLoss()
ner_criterion = nn.CrossEntropyLoss()

# Using AdamW optimizer for parameter updates
optimizer = AdamW(multi_task_model.parameters(), lr=2e-5)

# Defining number of training epochs
total_epochs = 3

# Training loop
for epoch in range(total_epochs):
    multi_task_model.train()

    for cls_batch, ner_batch in zip(classification_loader, ner_loader):

        # Classification Task
        cls_inputs = cls_batch["input"]  # List of raw texts
        cls_labels = cls_batch["label"].to(device)

        cls_logits = multi_task_model(cls_inputs, task="classification")
        cls_loss = classification_criterion(cls_logits, cls_labels)

        # NER Task
        ner_inputs = ner_batch["input"]  # List of tokenized or raw text
        ner_labels = ner_batch["labels"].to(device)

        ner_logits, ner_mask = multi_task_model(ner_inputs, task="ner")

        # Masking padded tokens
        valid_tokens = ner_mask.view(-1) == 1
        logits_filtered = ner_logits.view(-1, ner_logits.shape[-1])[valid_tokens]
        labels_filtered = ner_labels.view(-1)[valid_tokens]
        ner_loss = ner_criterion(logits_filtered, labels_filtered)

        # Combining Loss and Backpropagation
        total_loss = cls_loss + ner_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(
        f"Epoch [{epoch + 1}/{total_epochs}] → "
        f"Classification Loss: {cls_loss.item():.4f} | "
        f"NER Loss: {ner_loss.item():.4f}"
    )
