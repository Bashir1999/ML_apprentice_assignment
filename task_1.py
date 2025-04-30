# Loading pretrained model and tokenizer from Hugging Face
from transformers import AutoModel, AutoTokenizer

# Importing PyTorch for neural network operations
import torch


# Custom sentence embedding model
class CustomSentenceEmbedder(torch.nn.Module):
    def __init__(self, transformer_name):
        super(CustomSentenceEmbedder, self).__init__()

        # Loading the transformer encoder
        self.transformer = AutoModel.from_pretrained(transformer_name)

        # Loading associated tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)

        # Adding Linear layer to project to desired embedding dimension
        self.projection = torch.nn.Linear(self.transformer.config.hidden_size, 256)

        # Adding Non-linear activation for output embeddings
        self.activation_fn = torch.nn.Tanh()

    def forward(self, input_texts):
        # Tokenizing inputs with padding/truncation and convert to tensor format
        tokenized = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')

        # Passing through transformer to get contextual embeddings
        transformer_output = self.transformer(**tokenized)

        # Retrieving attention mask and last hidden states
        attention_mask = tokenized['attention_mask']
        hidden_states = transformer_output.last_hidden_state

        # Expanding mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Applying attention mask and compute the sum of valid token embeddings
        masked_sum = torch.sum(hidden_states * mask_expanded, dim=1)

        # Avoiding division by zero in case of all-masked input
        token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Computing mean pooled sentence embedding
        pooled_output = masked_sum / token_counts

        # Applying activation function
        final_embedding = self.activation_fn(self.projection(pooled_output))

        return final_embedding


# Instantiating the model
sentence_embedder = CustomSentenceEmbedder('distilbert-base-uncased')

sample_sentences = ["This is sentence number 1.", "This is sentence number 2",  "This is sentence number 3"]

# Generating embeddings
sentence_embeddings = sentence_embedder(sample_sentences)
print(sentence_embeddings)
