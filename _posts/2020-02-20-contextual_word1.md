---
title: "Contextual Word Representation(Pre-ELMo, ELMo and ULMfit)"
categories:
  - Natural Language Processing
tags:
  - Learning note
classes: wide


---



### 1.Introduction

So far, all the word representation strategies we talk about represent word as a fix window vector, which means, for one word, we only have one vector to represent them. This is convenient, but it makes some assumptions about language that do not fit with reality. Most importantly, words have different meanings in different contexts. For example:

<img src="/assets/images/image-20200616130902900.png" alt="image-20200616130902900" style="zoom:50%;" />

In the above example, bank in two context have different meaning. Although a long length vector may catch both meaning into it, what about words that have more than two meaning, like word *get*, which is mapped to over thirty different meanings.

Solution for this are quite simple: why not train word representation based on context word have? Actually, in RNN-language model, we will get word vetor in each step, which absorb information from context:

<img src="/assets/images/image-20200616133340036.png" alt="image-20200616133340036" style="zoom:50%;" />

Many models and strategies then be come up with to get a better contextual word embedding



### 2. Semi-Supervised Sequence Learning<sup>[3]</sup>

In paper, author come up with to methods of learning a sentence. the first it using LSTM autoencoder, the second is LSTM language model. Since we only focus on contextual word representation, we will only discuss the second method. Actually, they just using LSTM to pre-train a language model to predict the next word in sentence. after pre-training. using the weight of model as the initial weight to train the downstream task like text classification: 

<img src="/assets/images/image-20200616161628215.png" alt="image-20200616161628215" style="zoom:50%;" />

Author find that pre-train give model better performance in downstream task. Meanwhile, the output in each step of LSTM language model can be used as word representation which absorb contextual information.

### 3. Tag-LM(pre-ELMo)<sup>[4]</sup>

In tag-LM, author combine both fixed word embedding and contextual word representation as pre-trained word presentation. Then, use it in the downstream sequence labeling test:

<img src="/assets/images/image-20200616163634630.png" alt="image-20200616163634630" style="zoom:50%;" />

Meanwhile, in previous language model, people only use information from left context. However, in order to understand word better, the right information is also important, thus author use Bi-directional LSTM to get contextual word representation which absorb both left and right context information:

<img src="/assets/images/image-20200616163917585.png" alt="image-20200616163917585" style="zoom:50%;" />

For fixed token representation, model also apply both character-level and word level representation. Finally, concatanate all the vetcor as final word representation, then use it to do sequence tagging task. 



### 4.ELMo<sup>[5]</sup>

With the base of Tag-LM, same group come up with ELMo model, which is one of the milestone in contextual word representation. ELMo also use bi-directional LSTM(biLM) as language model. However, ELMo has two layers stacked LSTM and the final word representation is task specific combination of the intermediate layer representations in the biLM. The ELMo uses $L = 2$ biLSTM layers with 4096 units and 512 dimension projections and a residual connection from the first to second layer. The context insensitive type representation uses 2048 character n-gram convolutional filters followed by two highway layers and a linear projection down to a 512 representation.

For each token $t_k$, a $L-layer$  biLM compute a set of $2L+1$ representations:


$$
\begin{aligned}
R_{k} &=\left\{\mathbf{x}_{k}^{L M}, \overrightarrow{\mathbf{h}}_{k, j}^{L M}, \overleftarrow{\mathbf{h}}_{k, j}^{L M} | j=1, \ldots, L\right\} \\
&=\left\{\mathbf{h}_{k, j}^{L M} | j=0, \ldots, L\right\}
\end{aligned}
$$


Where $\mathbf{h}_{k, 0}^{L M}$ is the token layer and $$\mathbf{h}_{k, j}^{L M}=[\overrightarrow{\mathbf{h}}_{k, j}^{L M} ; \overleftarrow{\mathbf{h}}_{k, j}^{L M}]$$. 

For inclusion in a downstream model, ELMo combine all layers in $R_k$ into a single layer using:


$$
\mathbf{E} \mathbf{L} \mathbf{M} \mathbf{o}_{k}^{\operatorname{task}}=E\left(R_{k} ; \Theta^{\operatorname{task} }\right)=\gamma^{\operatorname{task} } \sum_{j=0}^{L} s_{j}^{\operatorname{task} } \mathbf{h}_{k, j}^{L M}
$$


Where $s^{task}$ are softmax-normalized weights and the scalar parameter $\gamma^{task}$ scalar overall usefulness of ELMo to task.

<img src="/assets/images/image-20200616221655187.png" alt="image-20200616221655187" style="zoom:50%;" />

For using ELMo in downstream task, we always freeze weights of ELMo after pre-training. Then, Concatenate ELMo weights into task specific model. An interseting thing finded during evaluation is that, the two biLM layers have differentiated meanings. Lower layer is better for lower-level synatic like NER or part-of-speech tagging, while higher layer is better for higher-level semantics.

### 5. ULM-fit<sup>[6]</sup>

ULM-fit also apply same general idea of transferring LM language but in slightly different way. The model is also multi-layer LSTM:

<img src="/assets/images/image-20200616224001591.png" alt="image-20200616224001591" style="zoom:50%;" />

First, train model on big general domain corpus, which is pre-training. Next, apply model to task specific data, fine-tune the model using the data. during fine-tune, ULM-fit introduce two ideas: *Discriminative fine-tuning* and *Slanted triangular learning rates*. Discriminative fine tuning is apply different learning rate in different layer. *Slanted triangular learning rate* is to adjust learning rate during fine-tuning process. We first increase the learning rate, then decrease the learning rate through fine-tuning:


$$
\begin{aligned}
c u t &=\left\lfloor T \cdot c u t_{-} f r a c\right\rfloor \\
p &=\left\{\begin{array}{ll}
t / c u t, & \text { if } t<c u t \\
1-\frac{t-c u t}{\operatorname{cut} \cdot(1 / c u t-f r a c-1)}, & \text { otherwise }
\end{array}\right.\\
\eta_{t} &=\eta_{\max } \cdot \frac{1+p \cdot(\text {ratio }-1)}{\text {ratio}}
\end{aligned}
$$


Where $T$ is the number of training iterations, $\text{cut_frac}$ is the fraction of iterations we increase the learning rate. $p$ is the fraction of the number of iterations we have increased or will decrease the learning rate respectively. $ratio$ specifies how much smaller the lowest learning rate is from the maximum learning rate $\eta_{max}$. $\eta_t$ is the learning rate in iteration $t$. In paper, $\text{cut_frac}=0.1$, $ratio=32$ and $\eta_{max}=0.01$. The learning rate through out fine-tuning iterations are:

<img src="/assets/images/image-20200616225217020.png" alt="image-20200616225217020" style="zoom:50%;" />

Finally, we fine-tune LM model in classification task.  Model augment the pretrained language model with two additional linear blocks. Each block uses batch normalization and dropout, with ReLU activations for the intermediate layer and a softmax activation that outputs a probability distribution over target classes at the last layer. For classification, model use the concatenation of all hidden state in the top of LM:


$$
\mathbf{h}_{c}=\left[\mathbf{h}_{T}, \operatorname{maxpool}(\mathbf{H}), \operatorname{meanpool}(\mathbf{H})\right]
$$


Where $\mathbf{H}=\{h_1,h_2,...,h_T\}$.



### Referenece

[1] Smith, N.A. (2019). Contextual Word Representations: A Contextual Introduction. *ArXiv, abs/1902.06006*.

[2] CS 224N Lecture notes, Standford. http://web.stanford.edu/class/cs224n/index.html

[3] Dai, A.M., & Le, Q.V. (2015). Semi-supervised Sequence Learning. *ArXiv, abs/1511.01432*.

[4] Peters, M.E., Ammar, W., Bhagavatula, C., & Power, R. (2017). Semi-supervised sequence tagging with bidirectional language models. *ArXiv, abs/1705.00108*.

[5]Peters, M.E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. *ArXiv, abs/1802.05365*.

[6]Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL*.