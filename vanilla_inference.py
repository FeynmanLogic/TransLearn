import torch
import time
from transformers import RobertaTokenizer
from datasets import load_dataset
from transformer import TransformerClassifier  


device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
dataset = load_dataset('imdb', split='test').select(range(11000, 11000 + 1000))  # Start from entry 11000


d_model = 512
ffn_hidden = 2048
num_heads = 8
num_layers = 5
drop_prob = 0.1
num_classes = 1  


vanilla_model = TransformerClassifier(d_model, ffn_hidden, num_heads, drop_prob, num_layers, num_classes, vocab_size=tokenizer.vocab_size).to(device)
vanilla_model.load_state_dict(torch.load("transformer_classifier_vanilla.pth"))
vanilla_model.eval()


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=200)

test_dataset = dataset.map(tokenize_function, batched=True)
test_inputs = torch.tensor(test_dataset['input_ids'], dtype=torch.long).to(device)


total_time = 0
for _ in range(10):  # Run inference 10 times for averaging
    start_time = time.time()
    with torch.no_grad():
        _ = vanilla_model(test_inputs)
    end_time = time.time()
    total_time += (end_time - start_time)

average_inference_time = total_time / 10
print(f"Average inference time for Vanilla model: {average_inference_time:.4f} seconds")

with open("results.txt", "w") as file:
    file.write(f"Vanilla Model Inference Time Results\n")
    file.write(f"Average inference time for Vanilla model: {average_inference_time:.4f} seconds\n")

print("Vanilla model inference time results saved to results.txt")
