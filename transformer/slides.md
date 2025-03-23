# Introduction to Transformers

- Transformers revolutionized NLP by using **attention mechanisms**.
- Unlike RNNs, Transformers handle **long-range dependencies** efficiently.
- Key advantage: Parallelization through **self-attention**.
- Transformer models like **BERT**, **GPT**, and **T5** have set new benchmarks.

---

# Why Transformers?

- Limitations of RNNs and LSTMs:
  - Sequential processing limits parallelism.
  - Struggles with long-range dependencies.
- Self-attention addresses these challenges by:
  - Allowing **direct connections** between any pair of words.
  - Enabling parallel processing.
- **Attention is All You Need** (Vaswani et al., 2017) introduced the Transformer.

---

# Attention Mechanism

- Attention allows the model to focus on **relevant parts** of the input.
- Computes a weighted sum of **values**, using a similarity score.
- Three key components:
  - Query (Q)
  - Key (K)
  - Value (V)

**Equation:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

---

# Self-Attention Mechanism

- Each word attends to every other word in the sentence.
- Captures **relationships** regardless of their distance.

**Self-Attention Calculation:**

1. Compute **Query**, **Key**, and **Value** matrices.
2. Calculate attention scores as a dot product.
3. Apply **softmax** to obtain weights.
4. Aggregate weighted sum.

**Code Example:**

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)
```

---

# Multi-Head Attention

- Multiple attention heads allow learning of different representations.
- Instead of a single **self-attention**, **multi-head attention** has:
  - Multiple sets of Q, K, V.
  - Outputs concatenated and linearly transformed.

**Equation:**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

---

# Positional Encoding

- Transformers lack an **intrinsic notion of order**.
- Positional encodings inject order information.
- Use sine and cosine functions of different frequencies.

**Equation:**

$$
PE_{(pos, 2i)} = \sin \left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos \left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

---

# Transformer Architecture

| Component          | Description                                      |
| ----------------- | ------------------------------------------------ |
| Encoder            | Encodes the input sequence                      |
| Decoder            | Generates the output sequence                   |
| Multi-Head Attention | Captures multiple aspects of relationships     |
| Feed-Forward Network | Applies non-linearity                          |
| Positional Encoding | Adds positional information to embeddings       |

---

# Pre-training and Fine-Tuning

- Pre-training on massive text corpora to learn representations.
- Fine-tuning on specific tasks for adaptation.
- Examples:
  - **BERT**: Bidirectional representation
  - **GPT**: Unidirectional generation

---

# Code Implementation of Transformer Block

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
```

---

# Hands-On Coding Session

- Try implementing a **Transformer Encoder Block** using PyTorch.
- Experiment with different numbers of heads and embedding dimensions.
- Discuss your observations with your peers.

---

# Q&A and Wrap-Up

- Open floor for questions.
- Key takeaways:
  - Transformers are powerful for NLP tasks.
  - Attention mechanisms enhance performance.
  - Pre-training and fine-tuning are essential for SOTA models.
