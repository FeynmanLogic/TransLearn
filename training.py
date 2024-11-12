import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset
from transformer import TransformerClassifier
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load IMDB dataset
dataset = load_dataset('imdb')
sample_size = 5000
train_dataset = dataset["train"].shuffle(seed=42).select(range(sample_size))
test_dataset = dataset["test"].shuffle(seed=42).select(range(int(sample_size * 0.2)))

# Model parameters
d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 5
num_classes = 1  # Single output for binary classification with BCEWithLogitsLoss

# Initialize tokenizer and custom model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size  # Retrieve vocabulary size for the tokenizer

# Instantiate the custom model and move to device
model = TransformerClassifier(d_model, ffn_hidden, num_heads, drop_prob, num_layers, num_classes, vocab_size=vocab_size).to(device)

# Define directories to save the model
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch tensors with appropriate data types
def format_dataset(dataset):
    input_ids = torch.tensor(dataset['input_ids'], dtype=torch.long)  # Convert to LongTensor for embedding layer
    attention_mask = torch.tensor(dataset['attention_mask'], dtype=torch.float)  # Keep as float for masking
    labels = torch.tensor(dataset['label']).float().unsqueeze(1)  # Float for BCEWithLogitsLoss
    return input_ids, attention_mask, labels

train_inputs, train_masks, train_labels = format_dataset(train_dataset)
test_inputs, test_masks, test_labels = format_dataset(test_dataset)

# Define training parameters
batch_size = 8
learning_rate = 2e-5
num_epochs = 3

# Create DataLoader for test set
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()

# Training function
def train(model, dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, masks, labels = batch

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Training loss: {avg_loss}")

# Evaluation function with percentage of correct predictions
import torch
import torch.nn.functional as F

# Updated Evaluation Function
def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs, masks, labels = batch

            # Get model outputs (logits) and apply sigmoid for probability
            logits = model(inputs)
            probs = torch.sigmoid(logits)  # Convert logits to probabilities
            
            # Convert probabilities to binary predictions
            preds = (probs >= 0.5).float()  # Threshold at 0.5 for binary classification
            
            # Calculate number of correct predictions
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    percentage_correct = accuracy * 100
    print(f"Evaluation accuracy: {accuracy}")
    print(f"Percentage of correct predictions: {percentage_correct}%")
    
    return accuracy


# Active Learning Loop
num_iterations = 3
labeled_data = train_dataset.select(range(25))  # Initial small labeled data
unlabeled_data = train_dataset.select(range(25, len(train_dataset)))  # Remaining data

for iteration in range(num_iterations):
    print(f"Active Learning Iteration {iteration + 1}")
    
    # Step 1: Train the model on the current labeled dataset
    train_inputs, train_masks, train_labels = format_dataset(labeled_data)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train(model, train_dataloader)
    
    # Evaluation on test set (optional)
    evaluate(model, test_dataloader)

# Save the model after all active learning iterations
final_model_path = os.path.join(save_dir, "transformer_imdb_active_learning.pt")
torch.save(model.state_dict(), final_model_path)
print(f"Model saved at {final_model_path}")
