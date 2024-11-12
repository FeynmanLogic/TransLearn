# Transformer-Based Model with Active Learning and Knowledge Distillation

Transformers have become essential for sequential data tasks, offering high accuracy but at a computational cost. This project addresses these limitations through:

1. **Active Learning**: Improves model efficiency by selecting the most informative samples during training.
2. **Knowledge Distillation**: Transfers knowledge from a large, complex model (teacher) to a smaller, efficient model (student) to enhance performance in resource-constrained environments.

## Why Use the Student-Teacher Model for Knowledge Distillation?

Knowledge distillation aims to deploy a smaller, faster student model that mimics a more complex teacher model's accuracy. The teacher-student approach provides the student model with nuanced knowledge through "soft labels," leading to improved generalization and robustness on unseen data. This approach is ideal for scenarios with limited computational resources, such as mobile and edge devices.

## Key Papers Referenced

This project builds on foundational works in active learning and knowledge distillation:
1. **Burr Settles** - *Active Learning Literature Survey* (2009).
2. **Geoffrey Hinton et al.** - *Distilling the Knowledge in a Neural Network*.
3. **Vaswani et al.** - *Attention is All You Need*.

## Project Status

Active learning and knowledge distillation have been implemented. WE TRADE OFF SOME LOSS FOR HIGH THROUGHPUT IN PERFORMANCE

## Requirements

To run the project, install Python 3.0+, PyTorch, and the required libraries:

```bash
pip install torch transformers datasets
```

## Files and Execution Order

1. **`transformer.py`**  
   This file contains the Transformer model architecture used for both teacher and student models. Ensure this file is in the project directory before proceeding.

2. **`training.py`**  
   Trains the teacher model using active learning. The teacher model's parameters are specified, and the model is trained on the IMDB dataset. Active learning selects the most informative samples for efficient training.  
   ```bash
   python training.py
   ```

3. **`origi.py`**  
   Distills knowledge from the trained teacher model to a smaller student model. The student model learns from the teacher's "soft labels," improving generalization on unseen data. This script saves the trained student model's weights.
   ```bash
   python origi.py
   ```

4. **`compare_student.py`**  
   Measures the inference time for the student model. This script is useful for understanding the computational efficiency gained through knowledge distillation.
   ```bash
   python compare_student.py
   ```

5. **`vanilla_inference.py`**  
   Evaluates the original (vanilla) model's inference time on the same data used for the student model. Comparing the results provides insights into the student model's performance improvements.
   ```bash
   python vanilla_inference.py
   ```

## Running Order and Results

Run the scripts in the following order for optimal results:

1. **Training the student model post active learning**
   ```bash
   python training.py
   ```
2. **Training the Vanilla Model**
   ```bash
   python origi.py
   ```
3. **Comparing Inference Times**
   - Measure the student model's inference time:
     ```bash
     python compare_student.py
     ```
   - Measure the vanilla model's inference time:
     ```bash
     python vanilla_inference.py
     ```

### Results Output

Each script outputs key metrics to the console and saves them to `results.txt`, allowing easy comparison of inference times and model accuracy across configurations.
