import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformer import TransformerClassifier  # Assuming your model is saved in transformer.py

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load IMDB dataset and sample points for quick training
dataset = load_dataset('imdb')
sample_size = 10000  # Small sample size for comparison
train_dataset = dataset["train"].shuffle(seed=42).select(range(sample_size))
test_dataset = dataset["test"].shuffle(seed=42).select(range(int(sample_size * 0.1)))

# Model parameters
d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 5
num_classes = 1  # Single output for binary classification with BCEWithLogitsLoss

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size
model = TransformerClassifier(d_model, ffn_hidden, num_heads, drop_prob, num_layers, num_classes, vocab_size=vocab_size).to(device)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch tensors
def format_dataset(dataset):
    input_ids = torch.tensor(dataset['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(dataset['attention_mask'], dtype=torch.float)  # Not used in this vanilla transformer
    labels = torch.tensor(dataset['label']).float().unsqueeze(1)  # Float for BCEWithLogitsLoss
    return input_ids, attention_mask, labels

train_inputs, _, train_labels = format_dataset(train_dataset)
test_inputs, _, test_labels = format_dataset(test_dataset)

# Define DataLoader
batch_size = 8
train_data = TensorDataset(train_inputs, train_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = TensorDataset(test_inputs, test_labels)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

# Training function
def train(model, dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Training loss: {avg_loss}")

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            outputs = model(inputs)
            preds = (outputs > 0).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Evaluation accuracy: {accuracy}")
    return accuracy

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader)
    evaluate(model, test_dataloader)

# Save the trained model for comparison
torch.save(model.state_dict(), "transformer_classifier.pth")
