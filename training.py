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
sample_size = 5000
train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(5000,6000))

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Model parameters
d_model_teacher = 512
ffn_hidden_teacher = 2048
num_heads_teacher = 8
drop_prob_teacher = 0.1
num_layers_teacher = 5
num_classes = 1  # Binary classification with BCEWithLogitsLoss

# Initialize the teacher model
teacher_model = TransformerClassifier(d_model_teacher, ffn_hidden_teacher, num_heads_teacher, drop_prob_teacher,
                                      num_layers_teacher, num_classes, vocab_size=tokenizer.vocab_size).to(device)

# Define directories to save the model
save_dir = "./saved_models"
os.makedirs(save_dir, exist_ok=True)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert datasets to PyTorch tensors
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

# Define optimizer and loss function for teacher model
optimizer_teacher = optim.AdamW(teacher_model.parameters(), lr=learning_rate)
loss_fn_teacher = nn.BCEWithLogitsLoss()

# Active Learning with Teacher Model
num_iterations = 3
labeled_data = train_dataset.select(range(25))  # Initial small labeled data
unlabeled_data = train_dataset.select(range(25, len(train_dataset)))  # Remaining data

# Training function for teacher model
def train_teacher(model, dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch

        optimizer_teacher.zero_grad()
        outputs = model(inputs)
        loss = loss_fn_teacher(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer_teacher.step()

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

            logits = model(inputs)
            probs = torch.sigmoid(logits)
            
            preds = (probs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f"Evaluation accuracy: {accuracy * 100:.2f}%")
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

# Active Learning Loop
for iteration in range(num_iterations):
    print(f"Active Learning Iteration {iteration + 1}")
    
    # Train the teacher model on the current labeled dataset
    train_inputs, train_labels = format_dataset(labeled_data)
    train_data = TensorDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_teacher(teacher_model, train_dataloader)
    
    # Evaluate the teacher model on the test set
    evaluate(teacher_model, test_dataloader)
    
    uncertain_samples, uncertain_indices = uncertainty_sampling(teacher_model, unlabeled_data, n_samples=10)

    labeled_data = Dataset.from_dict({
        'text': labeled_data['text'] + uncertain_samples['text'],
        'label': labeled_data['label'] + uncertain_samples['label'],
        'input_ids': labeled_data['input_ids'] + uncertain_samples['input_ids'],
        'attention_mask': labeled_data['attention_mask'] + uncertain_samples['attention_mask']
    })
    
    unlabeled_data = unlabeled_data.filter(lambda _, idx: idx not in uncertain_indices, with_indices=True)
    if iteration == num_iterations - 1:
        torch.save(teacher_model.state_dict(), os.path.join(save_dir, "transformer_imdb_active_learning.pt"))
student_model = TransformerClassifier(d_model=256, ffn_hidden=512, num_heads=4, drop_prob=0.1, num_layers=3, num_classes=1, vocab_size=tokenizer.vocab_size).to(device)
optimizer_student = optim.AdamW(student_model.parameters(), lr=learning_rate)
criterion_hard = nn.BCEWithLogitsLoss()  
criterion_soft = nn.KLDivLoss(reduction="batchmean")  
temperature = 3.0
alpha = 0.5

def train_student(student_model, teacher_model, dataloader, temperature, alpha=0.5):
    student_model.train()
    total_loss = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        optimizer_student.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        student_logits = student_model(inputs)
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        student_probs = torch.sigmoid(student_logits / temperature)
        distillation_loss = criterion_soft(torch.log(student_probs), teacher_probs)
        hard_label_loss = criterion_hard(student_logits, labels)
        loss = alpha * distillation_loss + (1 - alpha) * hard_label_loss
        loss.backward()
        optimizer_student.step()
        total_loss += loss.item()
    print(f"Training Loss: {total_loss / len(dataloader):.4f}")
train_data = TensorDataset(train_inputs, train_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_student(student_model, teacher_model, train_dataloader, temperature, alpha)
    evaluate(student_model, test_dataloader)

# Save student model
torch.save(student_model.state_dict(), os.path.join(save_dir, "student_transformer_imdb.pt"))
