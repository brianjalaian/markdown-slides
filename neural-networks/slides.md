# Neural Networks: From Neurons to Backpropagation
### CAP6606: Machine Learning for ISR
#### Professor Brian Jalaian
##### Raschka & Mirjalili, Chapters 11–12

<div class="source">Raschka & Mirjalili (2022), Chs. 11–12</div>
<div class="slide-no">1</div>

---
## Learning Objectives

By the end, you should be able to:

1. Trace tensor shapes through a multilayer perceptron
2. Explain why nonlinear activations create expressive decision boundaries
3. Derive backpropagation for an affine–activation–affine network
4. Connect initialization, optimization, and regularization to training behavior
5. Diagnose underfitting, overfitting, and unstable gradients

<div class="source">Raschka & Mirjalili (2022), Chs. 11–12</div>
<div class="slide-no">2</div>

---
## Notation — Lock It Once

| Symbol | Shape | Meaning |
|:--|:--:|:--|
| $\mathbf{X}$ | $(B, D)$ | mini-batch: $B$ examples, $D$ input features |
| $\mathbf{W}^{(1)}, \mathbf{b}^{(1)}$ | $(D,H), (H,)$ | input-to-hidden parameters |
| $\mathbf{Z}^{(1)}, \mathbf{A}^{(1)}$ | $(B,H)$ | hidden pre-activations and activations |
| $\mathbf{W}^{(2)}, \mathbf{b}^{(2)}$ | $(H,K), (K,)$ | hidden-to-output parameters |
| $\mathbf{P}, \mathbf{Y}$ | $(B,K)$ | predicted probabilities and one-hot targets |

$B$: batch size &nbsp; $D$: input width &nbsp; $H$: hidden width &nbsp; $K$: classes

<div class="source">Raschka & Mirjalili (2022), §11</div>
<div class="slide-no">3</div>

---
## A Neuron Is an Affine Map + Activation

<div class="flow-row">
  <div class="flow-box">$\mathbf{x}\in\mathbb{R}^{D}$<small>features</small></div>
  <div class="arrow">→</div>
  <div class="flow-box accent">$z=\mathbf{x}^{\top}\mathbf{w}+b$<small>affine score</small></div>
  <div class="arrow">→</div>
  <div class="flow-box green">$a=\phi(z)$<small>activation</small></div>
</div>

- The affine map rotates, rescales, and shifts
- The activation bends the function
- Learning chooses $\mathbf{w}\in\mathbb{R}^{D}$ and $b\in\mathbb{R}$

<div class="key">Without $\phi$, stacking layers still collapses to one affine map.</div>

<div class="source">Rosenblatt (1958); Raschka & Mirjalili (2022), §11</div>
<div class="slide-no">4</div>

---
## Why Nonlinearity Matters — XOR

No single line separates the positive and negative XOR points.

<div class="xor-grid">
  <div class="zero">0<br><small>(0,0)</small></div><div class="one">1<br><small>(1,0)</small></div>
  <div class="one">1<br><small>(0,1)</small></div><div class="zero">0<br><small>(1,1)</small></div>
</div>

Two ReLU features solve it:

$$h\_1=\operatorname{ReLU}(x\_1+x\_2),\qquad h\_2=\operatorname{ReLU}(x\_1+x\_2-1)$$
$$\hat y=h\_1-2h\_2$$

For binary inputs, $\hat y=[0,1,1,0]$.

<div class="source">Minsky & Papert (1969); Goodfellow, Bengio & Courville (2016), §6</div>
<div class="slide-no">5</div>

---
## Multilayer Perceptron — the Picture

<div class="netfig">
<svg viewBox="0 0 1030 404" width="100%" style="max-height:430px"><defs><marker id="ah" markerWidth="7" markerHeight="7" refX="6" refY="2.4" orient="auto"><path d="M0,0 L0,4.8 L6,2.4 z" fill="#ffb86c"/></marker></defs><line x1="135" y1="118" x2="381" y2="80" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="118" x2="381" y2="158" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="118" x2="381" y2="236" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="118" x2="381" y2="314" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="196" x2="381" y2="80" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="196" x2="381" y2="158" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="196" x2="381" y2="236" stroke="#ffb86c" stroke-width="2.6" opacity="1"/><line x1="135" y1="196" x2="381" y2="314" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="274" x2="381" y2="80" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="274" x2="381" y2="158" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="274" x2="381" y2="236" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="135" y1="274" x2="381" y2="314" stroke="#8be9fd" stroke-width="0.7" opacity="0.28"/><line x1="427" y1="80" x2="649" y2="140" stroke="#bd93f9" stroke-width=".7" opacity=".28"/><line x1="427" y1="80" x2="649" y2="252" stroke="#bd93f9" stroke-width=".7" opacity=".28"/><line x1="427" y1="158" x2="649" y2="140" stroke="#bd93f9" stroke-width=".7" opacity=".28"/><line x1="427" y1="158" x2="649" y2="252" stroke="#bd93f9" stroke-width=".7" opacity=".28"/><line x1="427" y1="236" x2="649" y2="140" stroke="#bd93f9" stroke-width=".7" opacity=".28"/><line x1="427" y1="236" x2="649" y2="252" stroke="#bd93f9" stroke-width=".7" opacity=".28"/><line x1="427" y1="314" x2="649" y2="140" stroke="#bd93f9" stroke-width=".7" opacity=".28"/><line x1="427" y1="314" x2="649" y2="252" stroke="#bd93f9" stroke-width=".7" opacity=".28"/><circle cx="112" cy="118" r="23" fill="rgba(139,233,253,.10)" stroke="#8be9fd" stroke-width="2"/><text x="112" y="124" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">x<tspan font-size="11" dy="4">1</tspan></text><circle cx="112" cy="196" r="23" fill="rgba(139,233,253,.10)" stroke="#8be9fd" stroke-width="2"/><text x="112" y="202" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">x<tspan font-size="11" dy="4">2</tspan></text><circle cx="112" cy="274" r="23" fill="rgba(139,233,253,.10)" stroke="#8be9fd" stroke-width="2"/><text x="112" y="280" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">x<tspan font-size="11" dy="4">3</tspan></text><circle cx="404" cy="80" r="23" fill="rgba(189,147,249,.12)" stroke="#bd93f9" stroke-width="2"/><text x="404" y="86" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">a<tspan font-size="11" dy="4">1</tspan></text><circle cx="404" cy="158" r="23" fill="rgba(189,147,249,.12)" stroke="#bd93f9" stroke-width="2"/><text x="404" y="164" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">a<tspan font-size="11" dy="4">2</tspan></text><circle cx="404" cy="236" r="23" fill="rgba(189,147,249,.12)" stroke="#bd93f9" stroke-width="2"/><text x="404" y="242" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">a<tspan font-size="11" dy="4">3</tspan></text><circle cx="404" cy="314" r="23" fill="rgba(189,147,249,.12)" stroke="#bd93f9" stroke-width="2"/><text x="404" y="320" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">a<tspan font-size="11" dy="4">4</tspan></text><circle cx="672" cy="140" r="23" fill="rgba(80,250,123,.10)" stroke="#50fa7b" stroke-width="2"/><text x="672" y="146" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">p<tspan font-size="11" dy="4">1</tspan></text><circle cx="672" cy="252" r="23" fill="rgba(80,250,123,.10)" stroke="#50fa7b" stroke-width="2"/><text x="672" y="258" text-anchor="middle" font-size="17" fill="#f8f8f2" font-style="italic">p<tspan font-size="11" dy="4">2</tspan></text><text x="112" y="372" text-anchor="middle" font-size="15" fill="#8be9fd">input</text><text x="112" y="391" text-anchor="middle" font-size="13" fill="#a9b1c6">(B, D)</text><text x="404" y="372" text-anchor="middle" font-size="15" fill="#bd93f9">hidden</text><text x="404" y="391" text-anchor="middle" font-size="13" fill="#a9b1c6">(B, H)</text><text x="672" y="372" text-anchor="middle" font-size="15" fill="#50fa7b">output</text><text x="672" y="391" text-anchor="middle" font-size="13" fill="#a9b1c6">(B, K)</text><text x="258" y="26" text-anchor="middle" font-size="15" fill="#8be9fd">W<tspan font-size="10" dy="-5">(1)</tspan><tspan dy="5" font-size="13" fill="#a9b1c6"> (D,H)</tspan></text><text x="538" y="26" text-anchor="middle" font-size="15" fill="#bd93f9">W<tspan font-size="10" dy="-5">(2)</tspan><tspan dy="5" font-size="13" fill="#a9b1c6"> (H,K)</tspan></text><line x1="264" y1="220" x2="300" y2="330" stroke="#ffb86c" stroke-width="1" stroke-dasharray="3,3" opacity=".8" marker-end="url(#ah)"/><text x="296" y="348" text-anchor="end" font-size="14" fill="#ffb86c">one edge = one weight &#160;w<tspan font-size="10" dy="-5">(1)</tspan><tspan font-size="10" dy="9">23</tspan></text><line x1="429" y1="236" x2="756" y2="144" stroke="#bd93f9" stroke-width="1" stroke-dasharray="3,3" opacity=".7"/><rect x="762" y="92" width="256" height="150" rx="11" fill="rgba(189,147,249,.09)" stroke="#bd93f9" stroke-width="1.6"/><text x="778" y="120" font-size="14" fill="#ffb86c">inside hidden unit a<tspan font-size="10" dy="4">3</tspan></text><text x="778" y="154" font-size="15" fill="#f8f8f2">z<tspan font-size="10" dy="-5">(1)</tspan><tspan font-size="10" dy="9">3</tspan><tspan dy="-4"> = &#8721; x</tspan><tspan font-size="10" dy="4">d</tspan><tspan dy="-4"> w</tspan><tspan font-size="10" dy="4">d3</tspan><tspan dy="-4"> + b</tspan><tspan font-size="10" dy="4">3</tspan></text><text x="778" y="184" font-size="13" fill="#a9b1c6">weighted sum, then add the bias</text><text x="778" y="216" font-size="15" fill="#50fa7b">a<tspan font-size="10" dy="-5">(1)</tspan><tspan font-size="10" dy="9">3</tspan><tspan dy="-4"> = &#981;( z</tspan><tspan font-size="10" dy="-5">(1)</tspan><tspan font-size="10" dy="9">3</tspan><tspan dy="-4"> )</tspan></text><text x="778" y="235" font-size="13" fill="#a9b1c6">activation &#981; makes it nonlinear</text></svg>
</div>

<div class="key">Every symbol ahead points at something here: a <strong>line</strong> is a weight, a <strong>circle</strong> holds a pre-activation $z$ and then an activation $a=\phi(z)$, and a <strong>bundle of lines</strong> is a weight matrix.</div>

<div class="source">Raschka & Mirjalili (2022), §11</div>
<div class="slide-no">6</div>

---
## Forward Pass — Matrix Form

<div class="flow-row shapes">
  <div class="flow-box"><b>X</b><small>(B, D)</small></div>
  <span class="arrow">→</span>
  <div class="flow-box accent"><b>Z<sup>(1)</sup></b><small>(B, H)</small></div>
  <span class="arrow">→</span>
  <div class="flow-box accent"><b>A<sup>(1)</sup></b><small>(B, H)</small></div>
  <span class="arrow">→</span>
  <div class="flow-box green"><b>Z<sup>(2)</sup></b><small>(B, K)</small></div>
  <span class="arrow">→</span>
  <div class="flow-box green"><b>P</b><small>(B, K)</small></div>
</div>

<div class="fragment eq">$$\underbrace{\mathbf{Z}^{(1)}}_{(B,H)}= \underbrace{\mathbf{X}}_{(B,D)} \underbrace{\mathbf{W}^{(1)}}_{(D,H)}+ \underbrace{\mathbf{b}^{(1)}}_{(H,)}$$</div>

<div class="fragment eq">$$\underbrace{\mathbf{A}^{(1)}}_{(B,H)}=\phi\!\left(\underbrace{\mathbf{Z}^{(1)}}_{(B,H)}\right)$$</div>

<div class="fragment eq">$$\underbrace{\mathbf{Z}^{(2)}}_{(B,K)}= \underbrace{\mathbf{A}^{(1)}}_{(B,H)} \underbrace{\mathbf{W}^{(2)}}_{(H,K)}+ \underbrace{\mathbf{b}^{(2)}}_{(K,)}$$</div>

<div class="fragment eq">$$\underbrace{\mathbf{P}}_{(B,K)}=\operatorname{softmax}\!\left(\underbrace{\mathbf{Z}^{(2)}}_{(B,K)}\right)$$</div>

<div class="source">Raschka & Mirjalili (2022), §11</div>
<div class="slide-no">7</div>

---
## Choosing an Activation

<div class="sketch">
  <figure><svg width="150" height="92" viewBox="0 0 150 92">
      <line x1="8" y1="46" x2="142" y2="46" stroke="#5a6080" stroke-width="1"/>
      <line x1="75" y1="6" x2="75" y2="86" stroke="#5a6080" stroke-width="1"/>
      <path d="M10,82 C45,82 60,10 140,10" fill="none" stroke="#8be9fd" stroke-width="2.4"/>
    </svg><figcaption>sigmoid — saturates both ends</figcaption></figure>
  <figure><svg width="150" height="92" viewBox="0 0 150 92">
      <line x1="8" y1="46" x2="142" y2="46" stroke="#5a6080" stroke-width="1"/>
      <line x1="75" y1="6" x2="75" y2="86" stroke="#5a6080" stroke-width="1"/>
      <path d="M10,84 C45,84 60,8 140,8" fill="none" stroke="#8be9fd" stroke-width="2.4"/>
    </svg><figcaption>tanh — zero-centred</figcaption></figure>
  <figure><svg width="150" height="92" viewBox="0 0 150 92">
      <line x1="8" y1="46" x2="142" y2="46" stroke="#5a6080" stroke-width="1"/>
      <line x1="75" y1="6" x2="75" y2="86" stroke="#5a6080" stroke-width="1"/>
      <path d="M10,46 L75,46 L140,10" fill="none" stroke="#8be9fd" stroke-width="2.4"/>
    </svg><figcaption>ReLU — no positive saturation</figcaption></figure>
  <figure><svg width="150" height="92" viewBox="0 0 150 92">
      <line x1="8" y1="46" x2="142" y2="46" stroke="#5a6080" stroke-width="1"/>
      <line x1="75" y1="6" x2="75" y2="86" stroke="#5a6080" stroke-width="1"/>
      <path d="M10,48 C50,52 62,40 140,10" fill="none" stroke="#8be9fd" stroke-width="2.4"/>
    </svg><figcaption>GELU — smooth gate</figcaption></figure>
</div>

| Activation | Strength | Failure mode |
|:--|:--|:--|
| Sigmoid | probabilistic gate | saturates; not zero-centered |
| Tanh | zero-centered | saturates for large $\lvert z\rvert$ |
| ReLU | cheap; strong gradients for $z>0$ | inactive “dead” units |
| GELU | smooth gate; common in Transformers | more computation |

**Default for an introductory MLP:** ReLU hidden units; task-specific output activation.

<div class="source">Nair & Hinton (2010); Hendrycks & Gimpel (2016)</div>
<div class="slide-no">8</div>

---
## From Logits to Probabilities

<div class="bar-pair">
  <div class="bar-group"><b>logits z (unbounded)</b>
    <div class="barset">
      <div class="bar" style="height:108px"><em>2.0</em><i>cat</i></div>
      <div class="bar" style="height:62px"><em>1.0</em><i>dog</i></div>
      <div class="bar" style="height:22px"><em>0.1</em><i>bird</i></div>
    </div>
  </div>
  <div class="bar-group"><b>probabilities p (sum to 1)</b>
    <div class="barset">
      <div class="bar hi" style="height:86px"><em>.66</em><i>cat</i></div>
      <div class="bar lo" style="height:31px"><em>.24</em><i>dog</i></div>
      <div class="bar lo" style="height:13px"><em>.10</em><i>bird</i></div>
    </div>
  </div>
</div>

<div class="fragment eq">$$p_{ik}=\frac{\exp(z_{ik}-m_i)}{\sum_{j=1}^{K}\exp(z_{ij}-m_i)},\qquad m_i=\max_j z_{ij}$$</div>

<div class="fragment eq">$$\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\sum_{k=1}^{K}y_{ik}\log(p_{ik}+\varepsilon)$$</div>

<div class="source">Goodfellow, Bengio & Courville (2016), §§6.2, 6.3</div>
<div class="slide-no">9</div>

---
## Empirical Checkpoint — Linear vs MLP

**Two-moons experiment:** same training split, same standardized inputs.

| Model | Representation | Expected boundary |
|:--|:--|:--|
| Logistic regression | original $(x\_1,x\_2)$ | one line |
| MLP $(2\rightarrow16\rightarrow2)$ | 16 learned ReLU features | piecewise nonlinear curve |

<div class="checkpoint">🧪 Run <code>02_neural_networks.ipynb</code>: compare test accuracy, decision boundaries, and training loss.</div>

**Question:** Is the MLP better because it optimizes harder—or because it can represent a different function class?

<div class="source">Raschka & Mirjalili (2022), Ch. 11</div>
<div class="slide-no">10</div>

---
## Training as a Feedback Loop

<div class="loop">
  <div>① Forward<br><small>$\mathbf{X}\rightarrow\mathbf{P}$</small></div>
  <div>② Loss<br><small>$\mathcal{L}(\mathbf{P},\mathbf{Y})$</small></div>
  <div>③ Backward<br><small>$\nabla_{\theta}\mathcal{L}$</small></div>
  <div>④ Update<br><small>$\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}$</small></div>
</div>

Repeat over mini-batches until validation performance stops improving.

<div class="key">Forward propagation computes values. Backpropagation computes vector–Jacobian products through the same graph in reverse.</div>

<div class="source">Rumelhart, Hinton & Williams (1986); Linnainmaa (1970)</div>
<div class="slide-no">11</div>

---
## Backpropagation = Repeated Chain Rule

For the scalar chain $x\rightarrow z\rightarrow a\rightarrow\mathcal{L}$:

$$\frac{\partial\mathcal{L}}{\partial x}
=\frac{\partial\mathcal{L}}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial x}$$

At each node, the backward pass receives an **upstream gradient**, multiplies by a **local derivative**, and sends a **downstream gradient**.

<div class="flow-row compact">
  <div class="flow-box">upstream<br>$\partial\mathcal{L}/\partial a$</div>
  <div class="arrow">×</div>
  <div class="flow-box accent">local<br>$\partial a/\partial z$</div>
  <div class="arrow">=</div>
  <div class="flow-box green">downstream<br>$\partial\mathcal{L}/\partial z$</div>
</div>

<div class="source">Rumelhart, Hinton & Williams (1986)</div>
<div class="slide-no">12</div>

---
## Softmax + Cross-Entropy Simplifies

<div class="fragment eq">$$\underbrace{\mathbf{G}^{(2)}}_{(B,K)}=\frac{\partial\mathcal{L}}{\partial\mathbf{Z}^{(2)}}=\frac{1}{B}\left(\underbrace{\mathbf{P}}_{(B,K)}-\underbrace{\mathbf{Y}}_{(B,K)}\right)$$</div>

<div class="compare2 fragment">
  <div><b>True class &nbsp;·&nbsp; p − 1 &lt; 0</b>
    gradient descent <strong>raises</strong> this logit
    <small>confidence was short of 1, so push it up</small></div>
  <div class="alt"><b>Every other class &nbsp;·&nbsp; p &gt; 0</b>
    gradient descent <strong>lowers</strong> this logit
    <small>any mass here is mass in the wrong place</small></div>
</div>

<div class="key">One subtraction is the entire error signal — this is why the softmax–cross-entropy pair is both clean to differentiate and stable to compute.</div>

<div class="source">Goodfellow, Bengio & Courville (2016), §6.2.2</div>
<div class="slide-no">13</div>

---
## Backpropagation — Matrix Form

<div class="flow-row shapes">
  <div class="flow-box green"><b>G<sup>(2)</sup></b><small>(B, K)</small></div>
  <span class="arrow">→</span>
  <div class="flow-box green"><b>∇W<sup>(2)</sup></b><small>(H, K)</small></div>
  <span class="arrow">→</span>
  <div class="flow-box accent"><b>G<sup>(1)</sup></b><small>(B, H)</small></div>
  <span class="arrow">→</span>
  <div class="flow-box"><b>∇W<sup>(1)</sup></b><small>(D, H)</small></div>
</div>

<div class="fragment eq">$$\underbrace{\nabla_{\mathbf{W}^{(2)}}\mathcal{L}}_{(H,K)}=\underbrace{\mathbf{A}^{(1)\top}}_{(H,B)}\underbrace{\mathbf{G}^{(2)}}_{(B,K)}$$</div>

<div class="fragment eq">$$\underbrace{\mathbf{G}^{(1)}}_{(B,H)}=\left(\underbrace{\mathbf{G}^{(2)}}_{(B,K)}\underbrace{\mathbf{W}^{(2)\top}}_{(K,H)}\right)\odot\phi'\!\left(\underbrace{\mathbf{Z}^{(1)}}_{(B,H)}\right)$$</div>

<div class="fragment eq">$$\underbrace{\nabla_{\mathbf{W}^{(1)}}\mathcal{L}}_{(D,H)}=\underbrace{\mathbf{X}^{\top}}_{(D,B)}\underbrace{\mathbf{G}^{(1)}}_{(B,H)}$$</div>

<div class="fragment"><small>Bias gradients sum each $\mathbf{G}$ over the batch axis.</small></div>

<div class="source">Raschka & Mirjalili (2022), §11</div>
<div class="slide-no">14</div>

---
## One Output-Layer Gradient, Worked

Suppose one example has hidden activation $\mathbf{a}=[2,1]$ and output error $\mathbf{g}=[-0.2,0.2]$.

$$\nabla\_{\mathbf{W}^{(2)}}\mathcal{L}
=\mathbf{a}^{\top}\mathbf{g}
=\begin{bmatrix}2\\1\end{bmatrix}
\begin{bmatrix}-0.2&0.2\end{bmatrix}
=\begin{bmatrix}-0.4&0.4\\-0.2&0.2\end{bmatrix}$$

Gradient descent subtracts this matrix:

- weights into the true class increase
- weights into the competing class decrease
- the more active hidden feature receives the larger update

<div class="source">Worked example adapted from Raschka & Mirjalili (2022), §11</div>
<div class="slide-no">15</div>

---
## Gradient Checking

<div class="compare2">
  <div><b>Analytic &nbsp;·&nbsp; backpropagation</b>
    one backward pass, all parameters at once
    <small>fast — what training actually uses</small></div>
  <div class="alt"><b>Numerical &nbsp;·&nbsp; finite difference</b>
    two loss evaluations <em>per parameter</em>
    <small>slow — a debugging check, never a training method</small></div>
</div>

<div class="fragment eq">$$g_j^{\text{num}}=\frac{\mathcal{L}(\theta_j+\epsilon)-\mathcal{L}(\theta_j-\epsilon)}{2\epsilon}$$</div>

<div class="fragment eq">$$\text{relative error}=\frac{\left\lVert\mathbf{g}^{\text{ana}}-\mathbf{g}^{\text{num}}\right\rVert_2}{\left\lVert\mathbf{g}^{\text{ana}}\right\rVert_2+\left\lVert\mathbf{g}^{\text{num}}\right\rVert_2+10^{-12}}$$</div>

<div class="key fragment">Double precision, a tiny network, and stay off ReLU kinks — then run it before any long training job.</div>

<div class="source">Goodfellow, Bengio & Courville (2016), §8.6</div>
<div class="slide-no">16</div>

---
## Mini-Batch Gradient Descent

For batch $\mathcal{B}\_t$:

$$\mathbf{g}\_t=\frac{1}{|\mathcal{B}\_t|}\sum\_{i\in\mathcal{B}\_t}\nabla\_{\theta}\ell\_i(\theta\_t),
\qquad
\theta\_{t+1}=\theta\_t-\eta\mathbf{g}\_t$$

| Batch size | Gradient noise | Memory | Updates / epoch |
|:--|:--:|:--:|:--:|
| small | higher | lower | more |
| large | lower | higher | fewer |

An **epoch** is one pass through the training set; an **iteration** is one parameter update.

<div class="source">Bottou (2010), Large-Scale Machine Learning with SGD</div>
<div class="slide-no">17</div>

---
## Learning Rate Controls the Dynamics

<div class="three-cards">
  <div><b>Too small</b><span>loss decreases slowly</span><i>underuses compute</i></div>
  <div class="good"><b>Useful range</b><span>steady descent</span><i>fast validation gain</i></div>
  <div class="bad"><b>Too large</b><span>oscillation or NaN</span><i>overshoots valleys</i></div>
</div>

Momentum smooths noisy directions:

$$\mathbf{v}\_{t+1}=\beta\mathbf{v}\_t+\mathbf{g}\_t,\qquad
\theta\_{t+1}=\theta\_t-\eta\mathbf{v}\_{t+1}$$

Adaptive optimizers help with scale differences, but they do not eliminate learning-rate tuning.

<div class="source">Polyak (1964); Kingma & Ba (2015)</div>
<div class="slide-no">18</div>

---
## Initialization Preserves Signal Scale

If weights are too small, activations and gradients vanish. If too large, they explode or saturate.

| Hidden activation | Common initialization | Variance target |
|:--|:--|:--|
| tanh / sigmoid | Xavier / Glorot | $\operatorname{Var}(W)\approx 2/(n\_{in}+n\_{out})$ |
| ReLU family | He / Kaiming | $\operatorname{Var}(W)\approx 2/n\_{in}$ |

All-zero weights are invalid: identical hidden units receive identical gradients and never specialize.

<div class="source">Glorot & Bengio (2010); He et al. (2015)</div>
<div class="slide-no">19</div>

---
## Normalize Inputs; Sometimes Normalize Layers

<div class="bar-pair">
  <div class="bar-group"><b>raw features — wildly different scales</b>
    <div class="barset">
      <div class="bar" style="height:14px"><i>age</i></div>
      <div class="bar" style="height:124px"><i>income</i></div>
      <div class="bar" style="height:30px"><i>visits</i></div>
    </div>
  </div>
  <div class="bar-group"><b>standardized — comparable, zero-centred</b>
    <div class="barset">
      <div class="bar hi" style="height:58px"><i>age</i></div>
      <div class="bar hi" style="height:70px"><i>income</i></div>
      <div class="bar hi" style="height:62px"><i>visits</i></div>
    </div>
  </div>
</div>

<div class="fragment eq">$$x'_d=\frac{x_d-\mu_d}{\sigma_d+\epsilon}\qquad\text{(inputs, once)}$$</div>

<div class="fragment eq">$$\hat z=\frac{z-\mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}}+\epsilon}},\qquad y=\gamma\hat z+\beta\qquad\text{(batch norm, inside)}$$</div>

<div class="key fragment">Related problems, not the same one — batch normalization does not replace input preprocessing. Modern view: it smooths optimization; “internal covariate shift” is not the full story.</div>

<div class="source">Ioffe & Szegedy (2015); Santurkar et al. (2018)</div>
<div class="slide-no">20</div>

---
## Regularization Controls Effective Capacity

| Method | Training-time action | Signal to watch |
|:--|:--|:--|
| $L\_2$ penalty | add $\lambda\lVert\mathbf{W}\rVert\_2^2$ to the loss | smoother, smaller parameters |
| Dropout | randomly mask activations | noisier training; robust features |
| Early stopping | keep best validation checkpoint | validation loss turns upward |
| Data augmentation | create label-preserving variations | invariance to realistic shifts |

$$\mathcal{L}\_{reg}=\mathcal{L}\_{data}+\lambda\lVert\mathbf{W}\rVert\_2^2$$

<div class="source">Srivastava et al. (2014); Prechelt (1998)</div>
<div class="slide-no">21</div>

---
## Read the Curves Before Changing the Model

| Pattern | Train metric | Validation metric | Likely response |
|:--|:--|:--|:--|
| Underfit | poor | poor | more capacity, better features, train longer |
| Overfit | strong | degrades / plateaus | regularize, augment, stop earlier |
| Unstable | oscillates / NaN | unreliable | lower $\eta$, normalize, clip/check gradients |
| Data bug | suspiciously perfect | suspiciously perfect | audit leakage and split logic |

<div class="key">A bigger network is only one possible intervention—and often the wrong first one.</div>

<div class="source">Raschka & Mirjalili (2022), Chs. 11–12</div>
<div class="slide-no">22</div>

---
## Applied Checkpoint — Handwritten Digits

**Task:** classify $8\times8$ grayscale images into $K=10$ digits.

<div class="network smallnet">
  <div class="layer"><b>Image</b><span>$(B,8,8)$</span><i>raw pixels</i></div>
  <div class="connector">flatten →</div>
  <div class="layer"><b>Vector</b><span>$(B,64)$</span><i>scaled to $[0,1]$</i></div>
  <div class="connector">MLP →</div>
  <div class="layer output"><b>Classes</b><span>$(B,10)$</span><i>probabilities</i></div>
</div>

<div class="checkpoint">🧪 Run <code>02b_neural_network_digit_recognition.ipynb</code>: baseline, MLP, error analysis, noise stress test, and confidence-based abstention.</div>

<div class="source">Dua & Graff, UCI Optical Recognition of Handwritten Digits</div>
<div class="slide-no">23</div>

---
## What an MLP Ignores About Images

Flattening maps $(B,8,8)\rightarrow(B,64)$ and discards explicit spatial structure.

| MLP | Convolutional network |
|:--|:--|
| every pixel connects independently | local receptive fields |
| no built-in translation structure | shared filters across positions |
| parameter count grows with image size | stronger image inductive bias |

An MLP can learn the task—but must learn spatial relationships from data rather than receiving them from the architecture.

<div class="source">LeCun et al. (1998); Raschka & Mirjalili (2022), Ch. 14</div>
<div class="slide-no">24</div>

---
## A Practical Debugging Order

1. **Data:** shapes, labels, split leakage, preprocessing fit only on train
2. **Forward pass:** finite logits, probabilities sum to one
3. **Loss:** can a tiny model overfit 10–20 examples?
4. **Backward pass:** gradient check; inspect zero / huge norms
5. **Optimization:** sweep learning rate; verify parameters change
6. **Generalization:** only then add regularization or more data

<div class="key">The tiny-batch overfit test is powerful: if a network cannot memorize a handful of examples, the training pipeline is probably broken.</div>

<div class="source">Goodfellow, Bengio & Courville (2016), Ch. 11</div>
<div class="slide-no">25</div>

---
## Takeaways

- An MLP alternates affine maps and nonlinear activations to learn representations
- Forward propagation computes predictions; cross-entropy measures their quality
- Backpropagation is reverse-mode chain rule with reusable local derivatives
- Tensor shapes make the derivation and implementation auditable
- Initialization, normalization, learning rate, and regularization shape training dynamics
- Evaluation must include error patterns and robustness—not accuracy alone

<div class="source">Raschka & Mirjalili (2022), Chs. 11–12</div>
<div class="slide-no">26</div>

---
## References

<div class="refs">

- Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS.
- He, K. et al. (2015). *Delving Deep into Rectifiers.* ICCV.
- Ioffe, S. & Szegedy, C. (2015). *Batch Normalization.* ICML.
- Rumelhart, D., Hinton, G. & Williams, R. (1986). *Learning representations by back-propagating errors.* Nature.
- Srivastava, N. et al. (2014). *Dropout.* JMLR.
- Raschka, S. & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*, Chs. 11–12.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, Chs. 6–8, 11.

</div>

<div class="source">Full URLs and additional sources are listed in the companion notebook.</div>
<div class="slide-no">27</div>
