# Transformers: Improving NLP with Attention Mechanisms
### CAP6606: Machine Learning for ISR
#### Dr. Brian Jalaian

<div style="text-align: right"><font size="4">1</font></div>

---
### Lecture Overview
- Introduction to Transformers
- Attention Mechanisms in RNNs
- Self-Attention and Scaled Dot-Product Attention
- Transformer Architecture: Encoder-Decoder Structure
- Pre-training and Fine-Tuning with Large Models
- Fine-Tuning BERT in PyTorch

<div style="text-align: right"><font size="4">2</font></div>

---
### Challenges & Solutions

| Challenges with Traditional Models | Solutions |
|----------------------------------|-----------|
| RNNs/LSTMs struggle with long dependencies | Attention mechanisms |
| Performance degrades with sequence length | Parallel processing |
| Sequential processing limits speed | Direct connections |

<div style="text-align: right"><font size="4">3</font></div>

---
### Key Innovations in Transformers

- **Architecture:** Replace RNN with self-attention mechanism
- **Modeling:** Direct modeling of all relationships in sequence
- **Benefits:**
  - Parallel training & computation
  - Better handling of long sequences
  - Improved model scalability

<div style="text-align: right"><font size="4">4</font></div>
---
### The Rise of Attention Mechanisms
- Attention mechanisms originally introduced to improve RNNs.
- Key idea: Focus on important parts of input rather than processing entire sequences.
- Allows dynamic weighting of inputs based on relevance.
- Major breakthrough: Transformer architecture (Vaswani et al., 2017).
- "Attention is All You Need" paper introduced self-attention to replace RNNs.

<div style="text-align: right"><font size="4">5</font></div>
---
### Adding Attention to RNNs
- Challenge: RNNs suffer from long-range dependencies.
- Solution: Introduce an attention mechanism to access relevant parts of the input.
- The model learns to generate context vectors using attention weights.
- Example: Machine translation and text summarization.

<div style="text-align: right"><font size="4">6</font></div>
---
# How Attention Works in RNNs
- Compute context vector for each output token.
- Context vector is a weighted sum of encoder hidden states.
- Weights are computed using:
  - Dot product between current hidden state and encoder states.
  - Softmax function to normalize weights.

<div style="text-align: right"><font size="4">7</font></div>
---
### How Attention Works in RNNs
- Compute context vector for each output token.
- Context vector is a weighted sum of encoder hidden states.
- Weights are computed using:
  - Dot product between current hidden state and encoder states.
  - Softmax function to normalize weights.

<div style="text-align: right"><font size="4">8</font></div>
---
# Attention Weight Computation
Given encoder hidden states (h₁, h₂, ..., hₙ) and decoder hidden state (d),
attention weights are computed as:

$\alpha_i = \frac{\exp(d \cdot h_i)}{\sum_{j=1}^{n} \exp(d \cdot h_j)}$

Context vector:

$c = \sum_{i=1}^{n} \alpha_i \cdot h_i$

<div style="text-align: right"><font size="4">9</font></div>
---

