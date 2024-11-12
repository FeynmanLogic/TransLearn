# Transformer-Based Model with Active Learning and Knowledge Distillation

Transformers have become a core model for various sequential data tasks. While they show high accuracy, they come with significant drawbacks, including long training times and inference delays. In this project, we aim to alleviate these limitations by incorporating two key strategies:

1. **Active Learning**: Improves accuracy and robustness by selecting the most informative data points during training.
2. **Knowledge Distillation**: Reduces the computational cost by transferring knowledge from a large, complex model to a smaller, more efficient one.

## Why Should We Use Student-Teacher Model for Knowledge Distillation

Knowledge distillation often aims to deploy the student model in scenarios where computational resources are limited (e.g., on mobile or edge devices). The student model, trained to mimic the teacher, captures most of the teacher's accuracy but is significantly smaller and faster, making it efficient for deployment in resource-constrained environments.

The teacher-student approach captures and conveys the nuanced knowledge of the teacher through the "soft labels" or probability distributions over classes. Instead of just using hard labels (one-hot encoding), the soft targets provide the student model with a richer signal that includes information about the relationships between classes (e.g., how the teacher model perceives the similarity between different categories).
This richer training signal helps the student model learn the underlying structure of the data more effectively, leading to better generalization.

The soft labels produced by the teacher model offer additional information that the student model can use to avoid overfitting on the training data. This is because soft labels smooth out the targets, providing a form of regularization, especially in cases where the student model is simpler and more prone to overfitting.
This regularization effect typically results in improved generalization on unseen data, allowing the student model to perform well even though it's less complex than the teacher.

## Key Papers Referenced
This project builds upon these key papers:
1. **Burr Settles**. *Active Learning Literature Survey*. Computer Sciences Technical Report 1648, University of Wisconsin–Madison. 2009.
2. **Geoffrey Hinton et al.** *Distilling the Knowledge in a Neural Network*.
3. **Vaswani et al.** *Attention is All You Need*.

## Project Status
We have implemented active learning, the binary cross entropy loss is steadily decreasing, but for some reason predicted accuracy is high.
- **Next Steps**:
  - Investigate alternative knowledge distillation techniques beyond pruning.

## Requirements

To run the project, install the following dependencies:
Python 3.0, Pytorch

```bash
pip install torch 
python transformer.py
