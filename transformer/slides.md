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
|:--------------------------------:|:----------:|
| **RNNs/LSTMs** struggle with long dependencies | **Attention** mechanisms |
| Performance degrades with sequence length | **Parallel** processing |
| Sequential processing limits speed | **Direct** connections |

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
### How Attention Works in RNNs??
- Compute context vector for each output token.
- Context vector is a weighted sum of encoder hidden states.
- Weights are computed using:
  - Dot product between current hidden state and encoder states.
  - Softmax function to normalize weights.

<div style="text-align: right"><font size="4">8</font></div>
---
### Attention Weight Computation
Given encoder hidden states (h₁, h₂, ..., hₙ) and decoder hidden state (d),
attention weights are computed as:

$\alpha_i = \frac{\exp(d \cdot h_i)}{\sum_{j=1}^{n} \exp(d \cdot h_j)}$

Context vector:

$c = \sum_{i=1}^{n} \alpha_i \cdot h_i$

<div style="text-align: right"><font size="4">9</font></div>
---
### Why Use Attention Mechanisms?
- Allows RNNs to handle longer sequences without losing context.
- Improves performance in tasks like translation and summarization.
- Reduces model complexity by focusing on important inputs.
- Paves the way for the development of the Transformer model.

<div style="text-align: right"><font size="4">10</font></div>
---
### Self-Attention Mechanism

| From\To | I | like | pizza |
|:----:|:----:|:----:|:----:|
| **I** | 0.7 | 0.2 | 0.1 |
| **like** | 0.3 | 0.3 | 0.4 |
| **pizza** | 0.1 | 0.5 | 0.4 |

**Key Points:**
- Each row: how one word attends to all others
- Row weights sum to 1.0 (probability distribution)
- Higher values = stronger relationships

<div style="text-align: right"><font size="4">11</font></div>
---
### Key Formula for Self-Attention
Given query (Q), key (K), and value (V) matrices:

$\text{Attention}(Q, K, V) = \text{softmax} \left(\frac{QK^T}{\sqrt{d_k}}\right) V$

Where:
- $d_k$: Dimensionality of keys
- Softmax ensures weights sum to 1.

**Why Scaling?**
- Scaling factor $\sqrt{d_k}$ prevents large dot products.

<div style="text-align: right"><font size="4">12</font></div>
---
### Self-Attention in Practice

```python
import torch
import torch.nn.functional as F

def self_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    context = torch.matmul(attention_weights, V)
    return context, attention_weights

# Example usage with random tensors
Q = torch.randn(1, 3, 4)  # Batch x Seq x Dim
K = torch.randn(1, 3, 4)
V = torch.randn(1, 3, 4)
context, weights = self_attention(Q, K, V)
print("Context:", context)
print("Attention Weights:", weights)
```

<div style="text-align: right"><font size="4">13</font></div>
---
### Comparison of RNNs and Transformers

| Aspect             | RNNs                     | Transformers                |
|-------------------|----------------------------|-----------------------------|
| Handling Sequences | Sequential, one at a time  | Parallel processing          |
| Long-Range Context | Struggles with long inputs | Efficient with attention     |
| Training Speed     | Slow due to recurrence     | Fast due to parallelization  |
| Memory Usage       | High for long sequences    | Efficient with fixed length  |

**Takeaway:** Transformers outperform RNNs in both efficiency and accuracy.
---
### Multi-Head Self-Attention
- Breaks the input into multiple heads for parallel processing.
- Each head learns different aspects of word interactions.
- Improves model's capacity to focus on different semantic relations.

### Visualization
(Insert Figure: Multi-Head Attention Architecture)
---
### How Multi-Head Attention Works
1. Split the input into multiple attention heads.
2. Perform self-attention independently on each head.
3. Concatenate the outputs and project to the final dimensions.

**Mathematical Formulation:**

$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$

- Each head:
$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
---
### Self-Attention Numerical Example

Suppose Q, K, V matrices are as follows:
Q = [[1, 0], [0, 1]]
K = [[1, 0], [0, 1]]
V = [[1, 2], [3, 4]]
### Calculation
1. Dot product: Q * K^T
2. Apply scaling and softmax.
3. Multiply softmax output by V.

| Q · K^T | Scaling | Softmax  | Context Vector |
|--------|--------|---------|----------------|
| 1.0    | 0.707  | 0.62    | 1.86           |
| 0.0    | 0.707  | 0.38    | 2.14           |
---