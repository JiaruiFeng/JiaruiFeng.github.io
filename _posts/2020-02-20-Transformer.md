---
title: "Transformer"
categories:
  - Natural Language Processing
tags:
  - Learning note
classes: wide

---



RNN based model have achieved really good performance in many NLP field. However, one of the biggest drawback of RNN is that sequence structure make it hard to parallelelize. 

In 2017, Google came up with idea called Transformer, which utilize self-attention and other techniques to metain the functionality of RNN but imporve the parallelization performance of model.

### 1. Structure of Model

We will illustrate transformer by machine translation problem. Usually, a NMT model would always involve a encoder and decoder. the encoder is used to encode input sentence to a vector, the decoder is used to convert this vector to the output. The transformer also use this structure like follow:

<img src="/assets/images/image-20200405183435784.png" alt="image-20200405183435784" style="zoom:33%;" />

To understand the encoder and decoder in transformer, we first introduce self-attention mechanism.

#### 1.1 Self-Attention

The input of self-attention layer are the word embedding for sentences or the output from previous layer. Let the dimension of our word embedding be $d_{model}$. In the original paper, $d_{model}=512$. let the length of sentence be $l$. Thus, for one sentence, our input would be a matrix $X \in \mathbb{R}^{l \times d_{model}}$. Self-attention layer have three parameters, query, key and value matrix, denoted it as $W^{Q},W^K,W^V \in \mathbb{R}^{d_{model} \times d_k},\mathbb{R}^{d_{model} \times d_k}, \mathbb{R}^{d_{model} \times d_v}$ respectively, where $d_k$ is a hyperparameter define the length of hidden vector. In original paper, it be set as 64, and we will talk about reason latter. 

The first step of self-attention is to compute three vectors for each word use $W^Q,W^K,W^V$ :


$$
Q=XW^Q\\
K=XW^K\\
V=XW^V
$$


Then, we compute a score for every word against each word in the sentence use Q and K, for each word $i$ against word $j$, the score is:


$$
s_{ij}=Q_iK_j^T
$$


Write it as matrix form:


$$
S=QK^T
$$


Where $S\in \mathbb{R^{ l\times l}}$. Next, we divide $S$ by $\sqrt{d_k}$, which will be useful in training. Casue a large $d_k$ in dot product attention would result a small gradient. Thus we scale it by $\sqrt{d_k}$ before dot product attention.

Then, we use softmax to normalize score based on row of $S$. Finally, we perform weighted sum for value vetor $V$ to get output for each word in sentence, where weight is $S$ after normalization. The finally formula of self-attention is


$$
\text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$


In paper, it is also called scaled dot-production attention. Notice that the ouput matrix would have dimension $\mathbb{R}^{l \times d_v}$. 

#### 1.2  Multi-Head Attention

Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries matrix. Transformer linearly project the queries, keys and alues $h$ times with different, learned linear projection to $d_k$, $d_k$ and $d_v$ dimension, respectively. Thus, for each projection $i$ where $i=1,2,...,h$, we will get a output $\text { Attention }(Q_i, K_i, V_i) \in\mathbb{R}^{l \times d_v} $. Then, we concat these projection to get multi-head attention:


$$
\begin{aligned}
\text { MultiHead }(Q, K, V) &\left.=\text { Concat (head }_{1}, \ldots, \text { head }_{h}\right) W^{O} \\
\text { where head }_{i} &=\text { Attention }(Q_i, K_i, V_i)
\end{aligned}
$$


Where $Concat(head_1,...,head_h) \in \mathbb{R}^{l \times hd_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{model}}$. In the paper, $d_v=d_k=d_{model}/h=64$, and $h=8$. Thus, the output dimension for attention layer would be equal to dimension of input and also dimension of word embedding. This is also an advantage when we talk about residual part of transformer latter.

#### 1.3 Position-wise Feed-Forward Networks

In the structure of encoder, tranformer also contains fully connected feed-forward network layer, which is applied to each position separately and identically.This consists of two linear transformations with a ReLU activation in between.


$$
\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
$$


The dimension of input and output are $d_{model}=512$, but in between, the dimension is $d_{ff}=2048$.

#### 1.4 Structure of Encoder

Now, we can describe the structure of encoder of transformer. In each transformer, it contain two sub-layer, the one is self-attention, the other is feed forward network layer. for each sub-layer, transfromer also apply a layer normalization and residual mechanism. In the original paper, author state the formula is $LayerNorm (x+\text { Sublayer }(x))$. However, In code implementation, the formula is $(x+\text{sublayer}(LayerNorm(x)))$. Seem like it have better performance in practical experiment. Here I use the formula consistent with original paper. The structure of encoder is follow:

<img src="/assets/images/image-20200405213815765.png" alt="image-20200405213815765" style="zoom:67%;" />

since the dimension of input and dimension of output from self-attention and feed-forward layer are same, we can directly add it with each other. That why paper set $h=8$. 

For transformer, we don't only use one encoder layer to encoder input sentence. Instead, tranformer have 6 stacking identical encoder layers to compose encoder part of transformer.

#### 1.5 Structure of Decoder

The decoder of transformer also contain self-attention and feed-forward layer, However, it is slightly differently from encoder.

First, decoder has the third sub-layer called encoder-decoder attention. The formula of this is the sam as self-attention. But, we use query matrix $Q_i$ from the self-attention layer below it, and use key and value matrix $K_i,V_i$ from the output of encoder part. 

Meanwhile, In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to `-inf`) before the softmax step in the self-attention calculation.

Thus, the decoder structure look like this:

<img src="/assets/images/image-20200405214307920.png" alt="image-20200405214307920" style="zoom:67%;" />

Similarily, transformer have 6 identical decoder. However, decoder part is not parallel like encoder. We need to compute current word before we compute next word. Basically, during prediction,we first input set of matrix $K_i,V_i$ from the output of encoder part and a special start token embedding to the decoder to generate first output word, then input this word and matrix, so far so for until we get special end token.

#### 1.6 Final Ouput Layer

like many other NLP language model, after decoder, transformer apply a fully connected NN and softmax to get the final distribution of word.

#### 1.7 Positional Encoding

For the input, transformer also do a little change. Since in current structure, transformer don't have power to encoding position information for each word in sentence. Thus, transformer inject some information about relative or absolute position of the tokens in the sequence which is called "positional encodings" at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. In the paper, auther mention two possible way to do that, one is functional based, the other is learning based. they test both way and find they two got nearly identical result. Finally, they choose the first way because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training. the function of positional encodeing is:


$$
\begin{aligned}
P E_{(p o s, 2 i)} &=\sin \left(\text {pos} / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(p o s, 2 i+1)} &=\cos \left(\text {pos} / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
$$


Where $pos$ is the position of word, $i$ is the dimension. Visualization result seem like this:

![image-20200405224627736](/assets/images/image-20200405224627736.png)

Each row repersent a word, and each column represent one dimension.

#### 1.8 Structure of Transformer

Finally, we conclude transformer structure as follow:

![Transformer Structure](/assets/images/Transformer Structure.png)

#### 1.9 Other Detail

For the optimizer, transformer use Adam optimizer and varied the learning rate over the course of training, according to the formula:


$$
\text {lrate}=d_{\text {model }}^{-0.5} \cdot \min \left(\text {step}_{-} \text {num}^{-0.5}, \text {step}_{-} \text {num} \cdot \text {warmup}_{-} \text {steps}^{-1.5}\right)
$$


Where $\text{warmup_steps=4000}$.

Transformer also apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized. And also apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. dropout rate is 0.1.





### Reference

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. *NIPS*.

[2]CS 224N Lecture notes, Standford. http://web.stanford.edu/class/cs224n/index.html

[3]https://jalammar.github.io/illustrated-transformer/

[4]https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html