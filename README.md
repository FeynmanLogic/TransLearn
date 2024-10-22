# Transformer-Based Model with Active Learning and Knowledge Distillation

Transformers have become a core model for various sequential data tasks. While they show high accuracy, they come with significant drawbacks, including long training times and inference delays. In this project, we aim to alleviate these limitations by incorporating two key strategies:

1. **Active Learning**: Improves accuracy and robustness by selecting the most informative data points during training.
2. **Knowledge Distillation**: Reduces the computational cost by transferring knowledge from a large, complex model to a smaller, more efficient one.

## Key Papers Referenced
This project builds upon these key papers:
1. **Burr Settles**. *Active Learning Literature Survey*. Computer Sciences Technical Report 1648, University of Wisconsin–Madison. 2009.
2. **Geoffrey Hinton et al.** *Distilling the Knowledge in a Neural Network*.
3. **Vaswani et al.** *Attention is All You Need*.

## Project Status
- **Current Progress**: The transformer model has been implemented.
- **Next Steps**:
  - Explore SVM-based active learning.
  - Investigate alternative knowledge distillation techniques beyond pruning.

## Requirements

To run the project, install the following dependencies:

```bash
pip install -r requirements.txt
python transformer.py
