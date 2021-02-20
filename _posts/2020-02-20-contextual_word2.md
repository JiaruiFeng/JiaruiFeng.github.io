---
title: "Contextual Word Representation(GPT, BERT, XLNet)"
categories:
  - Natural Language Processing
tags:
  - Learning note
classes: wide



---



### 1. GPT<sup>[1]</sup>

In GPT, language model is trained using tranformer decoder structure. Like other pre-training model, GPT also invovled two stage training: pre-training in large scaled unlabeled data and fine-tuning in specific task labeled data.  In unsupervised training, Given an unsupervised corpus of tokens $\mathcal{U}=\left\{u_{1}, \ldots, u_{n}\right\}$, language model is to maximize following likelihood:


$$
L_{1}(\mathcal{U})=\sum_{i} \log P\left(u_{i} | u_{i-k}, \ldots, u_{i-1} ; \Theta\right)
$$


Where $\Theta$ is parameters in neural network. GPT applies a multi-headed self-attention operation over the input context tokens followed by position-wise feedforward layers to produce an output distribution over target tokens:


$$
\begin{aligned}
h_{0} &=U W_{e}+W_{p} \\
h_{l} &=\operatorname{transformer}_{-} \text {block }\left(h_{l-1}\right) \forall i \in[1, n] \\
P(u) &=\operatorname{softmax}\left(h_{n} W_{e}^{T}\right)
\end{aligned}
$$


Where $W_e$ is token embedding, $W_p$ is position embedding, $n$ is number of transformer layer. 

 Since there is no encoder in this set up, these decoder layers would not have the encoder-decoder attention sublayer that vanilla transformer decoder layers have. It would still have the self-attention layer, however (masked so it doesn’t peak at future tokens). To be more specific, GPT has 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads).position-wise feed-forward networks has 3072 dimensional inner states. For optimizer, GPT uses Adam with a max learning rate of $2.5e-4$. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule. The activation function of GPT is GELU. 

After pre-training, GPT is appiled to different tasks for fine-tuning. GPT design defferent scheme for different tasks:

<img src="/assets/images/image-20200617131017802.png" alt="image-20200617131017802" style="zoom:50%;" />

Meanwhile, to improve generalization of the supervised model, and accelerate convergence, GPT applies auxiliary objective besides task specific objective. Suppose the objective of task is:


$$
L_{2}(\mathcal{C})=\sum_{(x, y)} \log P\left(y | x^{1}, \ldots, x^{m}\right)
$$


Where $x^1,...,x^m$ are input tokens, $y$ is objective, the overall objective during fine-tuning is:


$$
L_{3}(\mathcal{C})=L_{2}(\mathcal{C})+\lambda * L_{1}(\mathcal{C})
$$


Where $L_1(\mathcal{C})$ is objective of language model we illustrated above.



### 2.BERT<sup>[3]</sup>

 There are one problem for normal language model assumption: standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training.  For example, GPT can only use transformer decoder, ELMo can only train bi-directional LSTM seperately. However, such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.

Meanwhile, directly use Transformer encoder to implement bi-directional contextual language model also have problem: words can see themselves indirectly:

<img src="/assets/images/image-20200617162805660.png" alt="image-20200617162805660" style="zoom:50%;" />

All such limitation lead to the BERT. BERT is also two-step framework: pre-training and fine-tunin. The model first using large scale  unlabeled data over different pre-training tasks. For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Here is the two-steps framework:

<img src="/assets/images/image-20200617163307891.png" alt="image-20200617163307891" style="zoom:50%;" />

The model architecture of BERT is a multi-layer bidirectional Transformer encoder. In paper, they introduce two size of model: one is *BERT-base* and another is *BERT-large*. Let the number of transformer block as $L$, the hidden size as $H$, and the number of self-attention heads as $A$. The *BERT-base* has $L=12$, $H=768$ and $A=12$, which result in total 110M parameters. For *BERT-large*, it has $L=24$, $h=1024$ and $A=16$, whihch result in total 340M parameters

#### Input/Output Representation

In order BERT can handle a variety of downstream tasks, the input representation is able to unambiguously represent both a single sentence and a pair of sentences. BERT using wordPiece embedding as token input. The first token of every sequence is always a special classification token [CLS], which is used for classification task. Meanwhile, for sentence pair, BERT integrate them to a single sequence with a [SEP] token between them. Then, BERT add a learned embedding to every token indicating whether it belongs to sentence *A* or sentence *B*. For a given token, its input representation is constructed by summing the corresponding token embedding, segment embedding, and position embeddings:

<img src="/assets/images/image-20200617164756451.png" alt="image-20200617164756451" style="zoom:50%;" />

#### Pre-training Tasks of BERT

To overcome problems that mentioned above, BERT introduce different tasks during pre-training:

##### Masked LM

In order to train a deep bidirectional representation, BERT simply mask some percentage of the input tokens at random, and then predict those masked tokens. To be more specific, for each sentence, BERT will randomly mask 15% of words with [MASK] in the sentence and the task is to predict those words. However, a downside is that it will create a mismatch between pre-training and fine-tuning since the [MASK] token does not appear during fine-tuning. To mitigate this, BERT don't mask all 15% words, instead, The final strategy is that:

* Firstly, randomly select 15% of words from sentence.
* For 80% of the time, we replace words with [MASK]
* For 10% of the time, we replace words with random words.
* For 10% of the time, we do nothing.

##### Next Sentence Prediction(NSP)

To accommodate downstream tasks like question answering, which rely on understanding of the relationship between two sentences, BERT introduce another task during pre-training called Next Sentence Prediction. Specifically, when choosing the sentences $A$ and $B$ for each pre-training example, 50% of the time $B$ is the actual next sentence that follows $A$ (labeled as *IsNext*), and 50% of the time it is a random sentence from the corpus (labeled as *NotNext*).  The final state of first token [CLS] is used to predict whether $A$ and $B$ are next sentence relationship.

The overall task would be illustrated in follow:

<img src="/assets/images/image-20200617171206307.png" alt="image-20200617171206307" style="zoom:50%;" />

#### Fine-tuning of BERT

Fine-tuning of BERT is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks—whether they involve single text or text pairs—by swapping out the appropriate inputs and outputs:

<img src="/assets/images/image-20200617170604638.png" alt="image-20200617170604638" style="zoom:50%;" />

#### BERT for Feature Extraction

The fine-tuning approach isn’t the only way to use BERT. Just like ELMo, you can use the pre-trained BERT to create contextualized word embeddings. Then you can feed these embeddings to your existing model. There all many combination way to use BERT word embedding. In paper, they examine six choices in NER task:

<img src="/assets/images/image-20200617171030007.png" alt="image-20200617171030007" style="zoom:50%;" />

Actually, for different tasks, the result may vary.



### 3.XLNet<sup>[4]</sup>

BERT prove that learning language model using contextual information from both side of word is very important. However, although BERT introduce particular strategy to mitigate discrepancy between pre-training and fine-tuning, it still cannot solve it totally. Meanwhile, training of BERT also introduce independence assumption between all masking words in one sequence, which is not good in some cases. Forthmore, BERT give up so called autoregressive(AR) language modeling, which is perdict the distribution of a word based on all the former words. XLNet introduce new strategy that can maintain the ability to capture bidirectional contexts but avoid discrepancy. Specifically, for a sequence $\mathbf{x}$ of length $T$, there are $T!$ different valid factorization orders. if model parameters are shared across all factorization orders, in expectation, the model will learn to gather information from all positions on both sides.

#### Objective

Let $\mathcal{Z}_{T}$ be the set of all possible permutations of the length-$T$ index sequence $[1,2,3,...,T]$. Using $z_t$ and $\mathbf{z}_{<t}$ to denote the $t$-th element and first $t-1$ elements of a permutation $\mathbf{z} \in \mathcal{Z}_{T}$. The objective for new language modeling is follow:


$$
\max _{\theta} \quad \mathbb{E}_{\mathbf{z} \sim \mathcal{Z}_{T}}\left[\sum_{t=1}^{T} \log p_{\theta}\left(x_{z_{t}} | \mathbf{x}_{\mathbf{z}_{<t}}\right)\right]
$$



#### Two-Stream Self-Attention for Target-Aware Representations

However, to implement it in normal Transformer framework is not so easy. A simple masking of proper positions in attention cannot solve the problem. To see this, assume we parameterize the next-token distribution $ p_{\theta}(X_{z_{t}} \lvert  \mathbf{x}_{\mathbf{z}_{<t}})$ using the standard softmax distribution:


$$
p_{\theta}(X_{z_t}=\left.x | \mathbf{x}_{\mathbf{z}<t}\right)=\frac{\exp \left(e(x)^{\top} h_{\theta}\left(\mathbf{x}_{\mathbf{z}<t}\right)\right)}{\sum_{x^{\prime}} \exp \left(e\left(x^{\prime}\right)^{\top} h_{\theta}\left(\mathbf{x}_{\mathbf{z}<t}\right)\right)}
$$


Where $$ h_{\theta}\left(\mathbf{x}_{\mathbf{z}<t}\right)$$ is the hidden representation of $\mathbf{x}_{z<t}$ produced by transformer with porper masking. However, the prediction of $\mathbf{X}_{z_t}$ don't conditioned on the $z_t$. Suppose we have two different permutations $\mathbf{z}^{(1)}$ and $\mathbf{z}^{(2)}$, satisfying the following relationship:


$$
\mathbf{z}_{<t}^{(1)}=\mathbf{z}_{<t}^{(2)}=\mathbf{z}_{<t} \quad \text { but } \quad z_{t}^{(1)}=i \neq j=z_{t}^{(2)}
$$


Then, the prediction of $X_{z_t}$ will be equal no matter it is $z_{t}^{(1)}$ or $z_{t}^{(2)}$:


$$
\underbrace{p_{\theta}\left(X_{i}=x | \mathbf{x}_{\mathbf{z}_{<t}}\right)}_{z_{t}^{(1)}=i, \mathbf{z}_{<t}^{(1)}=\mathbf{z}_{<t}}=\underbrace{p_{\theta}\left(X_{j}=x | \mathbf{x}_{\mathbf{z}_{<t}}\right)}_{z_{t}^{(1)}=j, \mathbf{z}_{<t}^{(2)}=\mathbf{z}_{<t}}=\frac{\exp \left(e(x)^{\top} h\left(\mathbf{x}_{\mathbf{z}_{<t}}\right)\right)}{\sum_{x^{\prime}} \exp \left(e\left(x^{\prime}\right)^{\top} h\left(\mathbf{x}_{\left.\mathbf{z}_{\langle t}\right)}\right)\right.}
$$


A way to solve it is that, we don't use $ h_{\theta}\left(\mathbf{x}_{\mathbf{z}<t}\right)$ but $g_{\theta}\left(\mathbf{x}_{\mathbf{Z}<t}, z_{t}\right)$, which incorporate information of $z_t$ into it. However how to design $g_{\theta}$ is not trivial. For the prediction of $X_{z_t}$, the $g_{\theta}\left(\mathbf{x}_{\mathbf{Z}<t}, z_{t}\right)$ must only contain information of $z_t$ but not the content of $x_{z_t}$. However, for the $j>t$, the $g_{\theta}\left(\mathbf{x}_{\mathbf{Z}<t}, z_{t}\right)$ must contain the information of $x_{z_t}$.

To overcome this, XLNet introduce an architecture called Two-Stream Self-Attention for Target-Aware Representations. That is, inside have only one hidden representation, XLNet have two hidden representation:

* The content representation $h_{\theta}\left(\mathbf{x}_{\mathbf{z}<t}\right)$, or abbreviated as $h_{z_t}$ , which serves a similar role to the standard hidden states in Transformer. This representation encodes *both* the context and $x_{z_t}$ itself.
* The query representation $g_{\theta}\left(\mathbf{x}_{\mathbf{z}_{<t}}, z_{t}\right)$ or abbreviated as $g_{z_t}$, which only has access to the contextual information $\mathbf{x}_{\mathbf{z}<t}$ and the position $z_t$, but not the content $x_{z_t}$.

The first layer query stream is initialized with a trainable vector $g^{(0)}_{i}=w$, while the content stream is set to the corresponding word embedding $h^{(0)}_i=e(x_i)$. Thus, for each self-attention layer $m=1,...,M$, the formulation of two stream are:


$$
\begin{array}{l}
g_{z_{t}}^{(m)} \leftarrow \text { Attention }\left(\mathrm{Q}=g_{z_{t}}^{(m-1)}, \mathrm{KV}=\mathbf{h}_{\mathrm{z}<t}^{(m-1)} ; \theta\right) \\
h_{z_{t}}^{(m)} \leftarrow \text { Attention }\left(\mathrm{Q}=h_{z_{t}}^{(m-1)}, \mathrm{KV}=\mathbf{h}_{\mathbf{z}<t}^{(m-1)} ; \theta\right)
\end{array}
$$


You the see the picture illustration in follow:

![image-20200617222148551](/assets/images/image-20200617222148551.png)

To reduce the optimization difficulty, XLNet don't predict every word in sentence. Instead, it only predict the last $1/K$ tokens, where $K$ is a hyperparameter. 

Meanwhile, XLNet also incorporate relative postional encoding and and segment recurrence mechanism. Actually, relative positional encoding is what we talked about. For segment reurrence mechanism, suppose we have two segment taken from a long sequence $\mathbf{s}$, name as $\tilde{\mathbf{x}}=\mathbf{s}_{1: T}$ and $\mathbf{x}=s_{T+1:2T}$. Let $\tilde{\mathbf{z}}$ and $\mathbf{z}$ be the permutations of $[1,2,...,T]$ and $[T+1,T+2,...,2T]$ respectively. Then, based on one permutation of $\tilde{\mathbf{z}}$, we get the hidden representation $\tilde{\mathbf{h}}^{(m)}$ for each layer $m$. The attention update with memory can be written as:


$$
h_{z_{t}}^{(m)} \leftarrow \text { Attention }\left(\mathrm{Q}=h_{z_{t}}^{(m-1)}, \mathrm{KV}=\left[\tilde{\mathbf{h}}^{(m-1)}, \mathbf{h}_{\mathbf{z} \leq t}^{(m-1)}\right] ; \theta\right)
$$


Where $[...]$ is concatenation along the sequence dimension.The detailed view of content stream and query stream flow is illustrated in follow:

<img src="/assets/images/image-20200617231812610.png" alt="image-20200617231812610" style="zoom:50%;" />

<img src="/Users/jiaruifeng/Library/Application Support/typora-user-images/image-20200617231831854.png" alt="image-20200617231831854" style="zoom:50%;" />

Actually, XLNet also including segment relative embedding, here we don't talk about it in detailed. Druing training, XLNet use larger dataset with bigger model, which result in better performance that BERT in many tasks.

### Reference

[1] Radford, A. (2018). Improving Language Understanding by Generative Pre-Training.

[2]The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) http://jalammar.github.io/illustrated-bert/

[3]Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *ArXiv, abs/1810.04805*.

[4]Yang, Z., Dai, Z., Yang, Y., Carbonell, J.G., Salakhutdinov, R., & Le, Q.V. (2019). XLNet: Generalized Autoregressive Pretraining for Language Understanding. *NeurIPS*.

[5]CS 224N Lecture notes, Standford. http://web.stanford.edu/class/cs224n/index.html