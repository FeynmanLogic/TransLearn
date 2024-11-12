import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset
from transformer import TransformerClassifier
import os
import time

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load IMDB dataset
dataset = load_dataset('imdb')
sample_size = 10000  # Reduced sample size for faster active learning
train_dataset = dataset["train"].shuffle(seed=42).select(range(sample_size))
test_dataset = dataset["test"].shuffle(seed=42).select(range(int(sample_size * 0.1)))

# Model parameters
d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 5
num_classes = 1

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size
model = TransformerClassifier(d_model, ffn_hidden, num_heads, drop_prob, num_layers, num_classes, vocab_size=vocab_size).to(device)

# Define directories to save the model
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch tensors
def format_dataset(dataset):
    input_ids = torch.tensor(dataset['input_ids'], dtype=torch.long)
    labels = torch.tensor(dataset['label']).float().unsqueeze(1)
    return input_ids, labels

train_inputs, train_labels = format_dataset(train_dataset)
test_inputs, test_labels = format_dataset(test_dataset)

# Define training parameters
batch_size = 8
learning_rate = 2e-5
num_epochs = 3

# Create DataLoader for test set
test_data = TensorDataset(test_inputs, test_labels)
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
        inputs, labels = batch

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Training loss: {avg_loss}")

# Evaluation function with percentage of correct predictions
def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch

            logits = model(inputs)
            probs = torch.sigmoid(logits)
            
            preds = (probs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f"Evaluation accuracy: {accuracy}")
    
    return accuracy

# Uncertainty sampling based on entropy
def uncertainty_sampling(model, dataset, n_samples=10):
    model.eval()
    uncertainties = []
    
    for idx, sample in enumerate(dataset):
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_ids)
            probs = torch.sigmoid(logits).squeeze()
            
            uncertainty = - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
            uncertainties.append((idx, uncertainty.item()))
    
    # Sort samples by highest uncertainty
    uncertainties.sort(key=lambda x: x[1], reverse=True)
    uncertain_indices = [idx for idx, _ in uncertainties[:n_samples]]
    
    # Retrieve uncertain samples
    uncertain_samples = dataset.select(uncertain_indices)
    
    return uncertain_samples, uncertain_indices

# Active Learning Loop with Time Tracking
num_iterations = 3
labeled_data = train_dataset.select(range(25))  # Initial labeled data
unlabeled_data = train_dataset.select(range(25, len(train_dataset)))  # Remaining data

for iteration in range(num_iterations):
    print(f"Active Learning Iteration {iteration + 1}")
    start_time = time.time()
    
    # Step 1: Train the model on the current labeled dataset
    train_inputs, train_labels = format_dataset(labeled_data)
    train_data = TensorDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train(model, train_dataloader)
    
    # Step 2: Evaluate the model on the test set
    evaluate(model, test_dataloader)
    
    # Step 3: Use uncertainty sampling to find uncertain samples
    uncertain_samples, uncertain_indices = uncertainty_sampling(model, unlabeled_data, n_samples=10)
    
    # Step 4: Add uncertain samples to the labeled data
    labeled_data = Dataset.from_dict({
        'text': labeled_data['text'] + uncertain_samples['text'],
        'label': labeled_data['label'] + uncertain_samples['label'],
        'input_ids': labeled_data['input_ids'] + uncertain_samples['input_ids'],
        'attention_mask': labeled_data['attention_mask'] + uncertain_samples['attention_mask']
    })
    
    # Step 5: Remove selected uncertain samples from the unlabeled data
    unlabeled_data = unlabeled_data.filter(lambda _, idx: idx not in uncertain_indices, with_indices=True)
    
    end_time = time.time()
    print(f"Time for Active Learning Iteration {iteration + 1}: {end_time - start_time:.2f} seconds")
    
    # Step 6: Save model after each active learning iteration
    torch.save(model.state_dict(), os.path.join(save_dir, f"transformer_iteration_{iteration + 1}.pt"))

# Final model save
final_model_path = os.path.join(save_dir, "transformer_imdb_active_learning_final.pt")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")
