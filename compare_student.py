import torch
import time
from transformers import RobertaTokenizer
from datasets import load_dataset
from transformer import TransformerClassifier  # Assuming this is your model definition

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and dataset, selecting entries from 11000 onwards
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
dataset = load_dataset('imdb', split='test').select(range(11000, 11000 + 1000))

# Define student model parameters
d_model_student = 256
ffn_hidden_student = 512
num_heads_student = 4
num_layers_student = 3
drop_prob_student = 0.1
num_classes = 1

# Initialize the student model and load weights
student_model = TransformerClassifier(
    d_model_student, 
    ffn_hidden_student, 
    num_heads_student, 
    drop_prob_student,
    num_layers_student, 
    num_classes, 
    vocab_size=tokenizer.vocab_size
).to(device)
student_model.load_state_dict(torch.load("./saved_models/student_transformer_imdb.pt", map_location=device))
student_model.eval()

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)

test_dataset = dataset.map(tokenize_function, batched=True)
test_inputs = torch.tensor(test_dataset['input_ids'], dtype=torch.long).to(device)

# Measure inference time for the student model
total_time = 0
num_runs = 10  # Number of times to run the inference to get an average

with torch.no_grad():
    for _ in range(num_runs):
        start_time = time.time()
        _ = student_model(test_inputs)
        end_time = time.time()
        total_time += (end_time - start_time)

average_inference_time = total_time / num_runs
print(f"Average inference time for Student model: {average_inference_time:.4f} seconds")

# Write results to a file
with open("results.txt", "a") as file:  # Append to results.txt
    file.write("\nStudent Model Inference Time Results\n")
    file.write(f"Average inference time for Student model: {average_inference_time:.4f} seconds\n")

print("Student model inference time results saved to results.txt")
