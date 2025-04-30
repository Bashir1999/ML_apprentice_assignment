from transformers import AutoModel, AutoTokenizer  # Loading pretrained models and tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-task learning model handling both sentence classification and NER
class MultiTaskTextModel(nn.Module):
    def __init__(self, transformer_model_name, classification_labels, ner_labels):
        super().__init__()

        # Loading transformer encoder and tokenizer
        self.backbone = AutoModel.from_pretrained(transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

        # Mapping classification labels to indices
        self.cls_label_to_id = {label: idx for idx, label in enumerate(classification_labels)}
        self.cls_id_to_label = {idx: label for label, idx in self.cls_label_to_id.items()}

        # Mapping NER labels to indices
        self.ner_label_to_id = {label: idx for idx, label in enumerate(ner_labels)}
        self.ner_id_to_label = {idx: label for label, idx in self.ner_label_to_id.items()}

        # Dimensions
        num_cls_labels = len(classification_labels)
        num_ner_labels = len(ner_labels)

        # Sentence classification head
        self.cls_proj = nn.Linear(self.backbone.config.hidden_size, 256)
        self.cls_activation = nn.Tanh()
        self.cls_output = nn.Linear(256, num_cls_labels)

        # NER token classification head
        self.ner_output = nn.Linear(self.backbone.config.hidden_size, num_ner_labels)

    def forward(self, input_texts, task):
        # Tokenizing input with attention mask
        tokens = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True)

        # Forwarding through transformer
        transformer_out = self.backbone(**tokens)
        attention_mask = tokens['attention_mask']

        if task == "classification":
            # Mean pooling across token embeddings
            hidden_states = transformer_out.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            masked_sum = torch.sum(hidden_states * mask_expanded, dim=1)
            token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = masked_sum / token_counts

            # Forwarding through classification head
            x = self.cls_proj(pooled_output)
            x = self.cls_activation(x)
            logits = self.cls_output(x)
            return logits

        elif task == "ner":
            # Applying token classification head directly
            token_embeddings = transformer_out.last_hidden_state
            logits = self.ner_output(token_embeddings)
            return logits, attention_mask

        else:
            raise ValueError("Task must be 'classification' or 'NER'")


# Function for classification predictions
def decode_classification_logits(logits, id_to_label):
    probabilities = F.softmax(logits, dim=-1)
    predicted_indices = torch.argmax(probabilities, dim=1)
    confidences = torch.max(probabilities, dim=1).values
    return [(id_to_label[int(pred)], float(conf)) for pred, conf in zip(predicted_indices, confidences)]


# Function for NER predictions
def decode_ner_logits(logits, attention_mask, id_to_label):
    probabilities = F.softmax(logits, dim=-1)
    predicted_indices = torch.argmax(probabilities, dim=-1)

    all_labels = []
    for i in range(predicted_indices.size(0)):  # Loop over batch
        labels = [
            id_to_label[int(idx)] for idx, mask_val in zip(predicted_indices[i], attention_mask[i]) if mask_val == 1
        ]
        all_labels.append(labels)
    return all_labels


# Labels
classification_labels = ["negative", "neutral", "positive"]
ner_entity_labels = ["PER", "ORG", "LOC"]

# Instantiating Model
multi_task_model = MultiTaskTextModel('distilbert-base-uncased', classification_labels, ner_entity_labels)

# Sentence Classification Example
sample_texts = ["The weather is rainy today", "Those cologne are not cheap.", "This movie was awful."]
classification_logits = multi_task_model(sample_texts, task="classification")
classification_results = decode_classification_logits(classification_logits, multi_task_model.cls_id_to_label)

for text, (label, confidence) in zip(sample_texts, classification_results):
    print(f"[CLASSIFICATION] {text} → {label} ({confidence:.2f})")

# Named Entity Recognition Example
ner_inputs = ["Google HQ is located in California."]
ner_logits, ner_mask = multi_task_model(ner_inputs, task="ner")
ner_results = decode_ner_logits(ner_logits, ner_mask, multi_task_model.ner_id_to_label)

for text, ner_tags in zip(ner_inputs, ner_results):
    print(f"\n[NER] {text}")
    print("Labels:", ner_tags)
