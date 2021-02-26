---
title: "Graph Neural Network-Model and Theory"
categories:
  - Deep Learning
  - Graph Neural Network
tags:
  - Learning note
  - Literature review
classes: wide
---

### 1. Introduction

Graph structure data are ubiquitous in the world, representing objects and their relationship in many fields, like bioinformatics, social network, e-commerce networks. It will bring huge impact if we can efficiently represent the graph using algorithm and model[<sup>1</sup>](#refer1). However, the intrinsic non-Euclidean structure make the most of ordinary algorithms and model unusable.  Recently, a new type of deep learning model named **graph neural network(GNN)** achieved enormous attention since it propose a potential outlet for modeling graph-structure data. In this series of posts, a brief summarization of GNN including model and theory, application and practical tricks will be discussed.  This post is the first post in the series talking about theory and several popular models of GNN. For the rest of parts, please see: 

[Graph Neural Network-Application](https://jiaruifeng.github.io/deep%20learning/graph%20neural%20network/GNN-application/)

[Graph Neural Network-Practical Tricks](https://jiaruifeng.github.io/deep%20learning/graph%20neural%20network/GNN-Practical-tricks/)

### 2. Notations and Preliminaries

A graph is defined by $$G=(V,E)$$, where $$V$$ is the node set with $$V={v_1,v_2,...,v_n}$$, $n$ is the number of node in graph, $E$ is the edge set with $$E \subseteq V \times V$$ and we set $m$ is the number of edge in the graph. $$A \in \mathbb{R}^{n \times n}$$ is the adjacency matrix. In GNN, a graph will also have node feature and edge feature, denoted as $$X^{V}\in \mathbb{R}^{n \times d^V}$$ and $$X^{E} \in \mathbb{R}^{n \times d^E}$$, where $$x_i^V$$ is the feature of $i$-$th$ node in graph and $$x_{ij}^{E}$$  is the feature of edge between node $i$ and $j$ .  

ßTo better illustrate the GNN, we also define $$\mathcal{N}_{k}(i)$$ as k-step neighbors of $$v_i$$ and $$\mathcal{N}(i)$$ as 1-step neighbors of $$v_i$$.  The $$D\in n \times n$$ is the degree matrix of graph, where $$D(i, i)=\sum_{j} A(i, j)$$.  $$H^l \in \mathbb{R^{n \times d^V}}$$ is the hidden representation of graph in $$l$$ layer of GNN and $$h_i^l$$ is the hidden representation of node $i$ in layer $$l$$.  The $$\sigma$$ is the non-linear function in neural network, like ReLu and sigmoid. $$\odot$$ is the point-wise multiplication of matrix.



### 3. Typical Graph Neural Network 

#### 3.1 General Framework

First, we define the general framework of GNN, the most type of GNN will fit into this general framework.  The goal of GNN is to learn a meaningful representation of each node in graph. We want the learned representation of each node will contain the information of its neighbors or local network.  Based on this intuitive thought, a typical GNN contains three different components: **Massage**, **Aggregation** and **Readout function**. The Massage is used to process the node representation in the current layer, the Aggregation is used to aggregate the information among the neighbors of node in current layer, the readout function is used to generate graph representation based on the final node representation. Here, we denote these three components as $MSG$, $AGG$ and $READOUT$, a GNN model with $L$ layer can be define as:



$$
\begin{aligned}
m_u^l&=MSG^l(h_u^{l-1},X^E)\\
h_v^l&=AGG^l(\{m_u^l,u \in \mathcal{N}(v)\},h_v^{l-1})\\
G&=READOUT(H^L)
\end{aligned}
$$



Where $l$ is the $l$-$th$ layer in GNN model. Different GNN will have different algorithm for each components. The iteration of $$MSG$$ and $AGG$ in different is called the **massage passing mechanism**. 



#### 3.2 Graph Convolutional Network(GCN)[<sup>2</sup>](#refer2)

GCN is the first popular GNN model in the field. In GCN, the $$MSG$$ is a non-linear transformation function and $$AGG$$ is point-wise mean operation among the neighbors of nodes. Here we derive the formula of GCN. Suppose in layer $$l$$, the learnable transformation matrix is $$W^l$$:



$$
H^l=H^{l-1}W^{l}
$$



In order to do point-wise mean operation, we first do summation over the neighbors of node, which can be done by simply multiply the representation with adjacency matrix:



$$
H^{l}=AH^{l-1}W^{l}
$$



Next, to add the representation of center node, we can add a self-cycle by define the $$\hat{A}=A+I$$:



$$
H^{l}=\hat{A}H^{l-1}W^{l}
$$



In GCN, the mean operation utilize the geometric mean. Define $\hat{D}$ be the degree matrix of $\hat{A}$, the geometric mean can be done by:



$$
H^l=\hat{D}^{\frac{1}{2}}\hat{A}\hat{D}^{\frac{1}{2}}H^{l-1}W^{l}
$$



Finally, we add non-linear function(ReLu in the paper):



$$
H^l=\sigma(\hat{D}^{\frac{1}{2}}\hat{A}\hat{D}^{\frac{1}{2}}H^{l-1}W^{l})
$$



This is the final formula of GCN.  The illustration of GCN is shown in the following figure:

<div align=center>![GCN](/assets/images/GCN.png)

The GCN can also be written as general framework we mentioned in the beginning, which is inductive version of GCN:



$$
h_v^l=ReLu(W^{l} ·Geometric\_mean\{h_u^{l-1},\forall u \in \mathcal{N}(v) \cup\{v\}\})
$$



Notice that the formula combine $MSG$ and $AGG$ together. However, in the original paper, the computation is done by the matrix formula, which limit the model to transudative and not suitable for inductive learning(We must retrain the model once the $A$ changed).



#### 3.3 GraphSAGE[<sup>3</sup>](#refer3)

Compare to GCN, GraphSAGE formalize the GNN to a more general framework and also make the GNN inductive to large-scale graph. In the $$l$$-$$th$$ layer, the algorithm of GraphSAGE:



$$
\begin{aligned}
m_u^l&=AGGREGATE^l(h_u^{l-1},X^E, \forall u\in\mathcal{N}(v))\\
h^{l}_v&=\sigma(W^{l}·CONCAT(h_v^{l-1},m_u^l)\\
h^l_v&=h^{l}_v/\lvert\lvert h^{l}_v \rvert\rvert_2,\ \forall v\in V

\end{aligned}
$$



Here the $$AGGREGATE$$ is the $$MSG$$ component we mentioned above and the $$AGG$$ component here is a concatenation operation with a non-linear transformation. Further, GraphSAGE propose 4 different $$AGGREGATE$$ functions:

##### Mean Aggregator

Mean aggregator is simple, where the point-wise mean of vectors in $$\{h_u^{l-1}, \forall u\in\mathcal{N}(v)\}$$ is taken. This is quite similar to the GCN. 

##### GCN Aggregator

Further, author come up with GCN Aggregator, which is equal to the inductive version of GCN：



$$
h_v^l=ReLu(W^{l} ·Geometric\_mean\{h_u^{l-1},\forall u \in \mathcal{N}(v) \cup\{v\}\})
$$



##### Pooling Aggregator

For pooling aggregator, we take point-wise max-pooling operation, which is also symmetric and trainable:



$$
\text { AGGREGATE }_{l}^{\text {pool }}=\max (\{\sigma\left(W_{\text {pool }}^l h_{u}^{l-1}+b\right), \forall u \in \mathcal{N}(v)\})
$$



##### LSTM Aggregator

Finally, author examined the LSTM aggregator. Compared to the mean aggregator, LSTMs have the advantage of larger expressive capability.
However, it is important to note that LSTMs are not inherently symmetric  (i.e., they are not permutation invariant), since they process their inputs in a sequential manner.  To mitigate this problem, the LSTM aggregator is adapted to operate on an unordered set by simply applying the LSTMs to a random  permutation of the node's neighbors. The formula of LSTM aggregator is:



$$
\text { AGGREGATE }_{l}^{\text {LSTM }}=LSTM (\{ h_{u}^{l}, \forall u \in \mathcal{N}(v)\})
$$



Another contribution of GraphSAGE are that it introduce graph sampling for mini-batch training and strategy unsupervised training. We will cover more detail of them in the application and practical trick part.



#### Graph Attention Network(GAT)[<sup>4</sup>](#refer4)

 ![GAT](/assets/images/GAT.png)

GAT introduce the attention mechanism[<sup>5,6</sup>](#refer7) into GNN.  In GCN, we can regards the aggregation operation as assign a equal weight to each neighbor of cent node when doing summation. However, different neighbors should have different importance when aggregate their information to center node. Therefore, the main attribute of GAT is that it automatically learn a weight when doing aggregation operation:



$$
H^{l}=\lvert\rvert_{k=1}^{K}(\alpha^{k} H^{l-1} W_{k}^{l})
$$



$$
\alpha_{v u}^{k}=\frac{\exp \left(\operatorname{LeakReLU}\left(\boldsymbol{H}_{v}^{l-1} W_{k}^{l} \| \boldsymbol{H}_{u}^{l-1} W_{k}^{l} a_{k}^{l}\right)\right.}{\sum_{u \in \mathcal{N}_{v}} \exp \left(\operatorname{LeakReLU}\left(\boldsymbol{H}_{v}^{l-1} W_{k}^{l} \| \boldsymbol{H}_{u}^{l-1} W_{k}^{l} a_{k}^{l}\right)\right)}
$$



Where $$\lvert\rvert$$ is the concatenation operation and $$K$$ is the number of head in multi-head attention. $$W_k^{l}$$ and $$a_k^l$$ are all trainable parameters in the model.



#### 3.4 Relational Graph Convolutional Network(R-GCN)[<sup>7</sup>](#refer7)

The R-GCN extend the GNN to multiple edge type. let $r\in \mathcal{R}$ be all the possible edge type. The update formula of R-GCN is:


$$
h_{v}^{(l)}=\sigma\left(\sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}(v)} \frac{1}{c_{v, r}} W_{r}^{(l)} h_{j}^{(l-1)}+W_{0}^{(l)} h_{v}^{(l-1)}\right)
$$


Where $$W_r^{(l)}$$ and $$W_{0}^{(l)}$$ are trainable parameters and 0 denote the edge type of self-loop. $$c_{v,r}$$ is a problem-specific normalization constant that can either be learned or chosen in advance.

However, one possible defect of this formula is that, the number of parameters will growth rapidly as the number of edge type increase, which will further introduce overfitting on rare edge types. To solve the problem, R-GCN proposed two type of regularization to tackle the problem.

##### basis decomposition

In basis decomposition, the $$W_r^{(l)}$$ is defined by:


$$
W_{r}^{(l)}=\sum_{b=1}^{B} a_{r b}^{(l)} V_{b}^{(l)}
$$


Where $$V_b \in \mathbb{R}^{d_{l}\times d_{l-1}}$$ and $$a_{rb}^{(l)}$$ only depend on $r$. The basis decomposition can be regarded as a form of effective weight sharing between different relation types .

##### block-diagonal decomposition

In block-diagonal decomposition, the $$W_r^{(l)}$$ is defined by:


$$
W_{r}^{(l)}=\bigoplus_{b=1}^{B} Q_{b r}^{(l)}
$$


Where $$W_r^{(l)}$$ are block-diagonal matrices: $$\operatorname{diag}(Q_{1 r}^{(l)}, \ldots, Q_{B r}^{(l)})$$ with $$Q_{b r}^{(l)} \in \mathbb{R}^{(d^{(l+1)} / B) \times(d^{(l)} / B)}$$. The block-diagonal decomposition can be regarded as a sparsity constraint on the weight matrices for each relation type. The block decomposition structure encodes an intuition that latent features can be grouped into sets of variables which are more tightly coupled within groups than across groups.  



#### 3.5 DiffPool [<sup>18</sup>](#refer18)

There are many different possible way for $$READOUT$$ function in GNNs, like simple mean operation/max-pooling operation, or more sophisticated function like[<sup>19,</sup>](#refer19)[<sup>20,</sup>](#refer20)[<sup>21,</sup>](#refer21). Since the $$READOUT$$ are more likely a application-specific operation, we will talk more about it in the second part of posts. But here is one work called DiffPool that is worth to mention.

In DiffPool, the author consider the GNN as a hierarchical $$READOUT$$ operation. In each layer of DiffPool, the model not only learn the representation of the nodes, but also pool the nodes to construct a new coarsened graph. This whole process is repeated for $$L$$ layers and we use the final output representation to classify the graph, as shown in the figure below.

![diffpool](/assets/images/diffpool.png)

The model achieve this by introducing a cluster assignment matrix $$S^{(l)} \in \mathbb{R}^{n_{l} \times n_{l+1}}$$.  Each row of $$S^{(l)}$$ corresponds to one of the $$n_l$$ nodes(or clusters) at layer $$l$$, and each column of $$S^{(l)}$$ corresponds to one of the $$n_{l+1}$$ clusters at the next layer $$l+1$$. Suppose we already know the $$S^{(l)}$$, the generation of new coarsened graph in DiffPool is:


$$
\begin{array}{l}
H^{(l+1)}=S^{(l)^{T}} Z^{(l)} \in \mathbb{R}^{n_{l+1} \times d} \\
A^{(l+1)}=S^{(l)^{T}} A^{(l)} S^{(l)} \in \mathbb{R}^{n_{l+1} \times n_{l+1}}
\end{array}
$$


Where $$Z^{(l)}$$ is the embedding in the $$l$$-th layer GNN and $$d$$ is the hidden dimension. The learning of $$S^{(l)}$$ is achieved by another GNN layer:


$$
S^{(l)}=\operatorname{softmax}\left(\mathrm{GNN}_{l, \mathrm{pool}}\left(A^{(l)}, H^{(l)}\right)\right)
$$


Where the softmax function is applied in a row-wise fashion. We denote this GNN as pooling GNN.

In practice, the training of pooling GNN is hard and may stuck in local minima. To solve the problem, DiffPool introduce auxiliary link prediction objective which encodes the intuition that nearby nodes should be pooled together. At each layer $$l$$, we minimize $$L_{\mathrm{LP}}=\lvert\lvert A^{(l)}, S^{(l)} S^{(l)^{T}}\rvert\rvert_{F}$$, where $$\lvert\lvert·\rvert\rvert_F$$ is frobenius norm. Moreover, the DiffPool constrain the $$S^{(l)}$$ that each node should be a one-hot vector by minimize the loss $$L_{\mathrm{E}}=\frac{1}{n} \sum_{i=1}^{n} H\left(S_{i}\right)$$.



### 4 The expressive limitation of GNN and Solutions

#### 4.1 Over-Smoothing

As we already know, the depth of deep neural network play a crucial role in the performance. What happen if we simply increase the depth of GNN? The results show that the performance will actually decrease instead of increase. The reason is lies in the problem of **Over-Smoothing**. 

##### Over-Smoothing[<sup>8,</sup>](#refer8)[<sup>9</sup>](#refer9)

The over-smoothing is that, the node activation will converge to a certain space given different situation. The GNN is mainly aggregate information of center node and its neighbors. With this intuition, the nodes in $L$ layer GNN will contain the information of $L$-step neighbors. Therefore, if we continue increase the number of layer, every node in same component of graph will contain exactly same amount of information. Further, it has be proved that [<sup>9</sup>](#refer9) the GNN will converge to the following space given different situation:

![oversmoothing](/assets/images/oversmoothing.png)

As a result, we must be very careful when increase the depth of GNN. There have been many works that try to tackle this problem. One possible solution is DropEdge.

##### DropEdge[<sup>9</sup>](#refer9),[<sup>10</sup>](#refer10)

The idea of DropEdge is simple. In each training epoch, the DropEdge technique drops out a certain rate of edges of the input graph by random. Let $$A_{drop}$$ denote the resulting adjacency matrix, then its relation with original $A$ becomes:



$$
A_{drop}=Unif(A,1-p)
$$



Where the $$Unif(A,1-p)$$ uniformly samples each edge in A with probability of $1-p$. DropEdge also perform the re-normalization trick on $$A_{drop}$$ and use it in propagation and training. When validation and testing, we use the original $$A$$.

In the paper, the author also prove the effeteness of DropEdge in alleviating over-smoothing. Moreover, the neighborhood aggregation can be understood as a weighted sum of the neighbor features (the weights are associated with the edges). As for DropEdge, it enables a random subset aggregation instead of the full aggregation during training. This random aggregation, statistically, only changes the expectation of the neighbor aggregation up to a multiplier $$1 - p$$ that will be actually removed after adjacency re-normalization. Therefore, DropEdge is unbiased and can be regarded as a data augmentation skill for training GCN by generating different random deformations of the input data. In this way, DropEdge is able to prevent over-fitting, similar to typical image augmentation skills. 

The performance of DropEdge can be seen in the following figure:

![DropEdge](/assets/images/DropEdge.png)

#### 4.2 Training Dynamics and Over-fitting

training dynamics and over-fitting are long-stand question not only in GNN but in many other deep learning model. A common solution is to add short path in the model. Based on the intuition, several variant of GNNs have been proposed.

##### ResGCN[<sup>2</sup>](#refer2)

Inspired by ResNet[<sup>11</sup>](#refer11),  the author of GCN also examine the performance of using residual connection in GCN. The formula of ResGCN is as following:



$$
H^{(l)}=\sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l-1)} W^{(l)})+\alpha H^{(l)}
$$



Where the $\alpha$ control the balance between the GCN propagation and residual connection.

##### Jumping Knowledge network(JKNet)[<sup>12</sup>](#refer12)

In JKNet, we not only consider residual connection between $l-1$ and $l$ layer. Instead, we pass the information from each layer to the last layer, and then adaptively learn an layer aggregation function to integrate the information from each layer, as shown in the figure:

![jknet](/assets/images/jknet.png)

Where the $N.A.$ in the figure represent the $MSG$ and $AGG$ component of GNN. The layer aggregation is independent to the nodes. Moreover, the JKNet can be integrated into the most of GNN architectures.

##### Approximate personalized propagation of neural predictions(APPNP)[<sup>13</sup>](#refer13)

APPNP combine the idea of massage passing and personalized page rank algorithm. In each GNN layer, the representation is updated by the following formula:



$$
\begin{aligned}
Z^{(l,0)} &=f_{\theta}(H^{l-1}) \\
Z^{(l,k+1)} &=(1-\alpha) \hat{A} Z^{(l,k)}+\alpha Z^{(l,0)} \\
Z^{(l,K)} &=\operatorname{softmax}((1-\alpha) \hat{A} Z^{(l,K-1)}+\alpha Z^{(l,0)})
\end{aligned}
$$



Where the $\alpha$ is the probability of restart, $f_{\theta}$ is a neural network for non-linear transformation.



#### 4.3 The representational ability of GNN

##### Weisfeiler-Lehman test(WL-test)[<sup>14</sup>](14)

Before we talk about the representational ability of GNN, we first introduce the WL-test and the close relationship between WL-test and massage passing framework of GNN. WL-test is used to solve the graph isomorphism problem, that is, whether two graph have same topological structure. For the 1-dimensional form, in each iteration, the WL-test update the label of nodes in the graph by:



$$
h_v^{l}=HASH(h_v,\mathcal{F}\{h_u,\ u\in\mathcal{N(v)}\})
$$



Where $$\mathcal{F}$$ is a aggregation function used to aggregate the label of all the neighbors, like simple concatenation. $$HASH$$ is a injective function that will have different output whenever the input is different. the iteration will continue until the label of each node in graph become stable. If two graph have same label for each node, they are potentially isomorphic (but not necessary).

We can see that the formula of WL-test is really close to the massage passing framework of GNN. But there is one mean difference: in GNN, we usually use a non-linear transformation to replace the $$HASH$$ function. However, this transformation is not injective function, which limited the representational power of GNN.

##### How Powerful are Graph Neural Networks?[<sup>15</sup>](15)

Ideally, a maximally powerful GNN could distinguish different graph structures by mapping them to different representations in the embedding space. This ability to map any two different graphs to different embeddings, however, implies solving the challenging graph isomorphism problem. That is, we want isomorphic graphs to be mapped to the same representation and non-isomorphic ones to different representations.   

In the paper[<sup>15</sup>](15), the author prove that for any typical GNN that follow the massage passing mechanism, the representational ability is as much as powerful as WL-test. In other words:  *let the $$G_1$$ and $$G_2$$ be any two non-isomorphic graphs. If a GNN maps $$G_1$$ and $$G_2$$ to different embeddings, the WL-test also decides $$G_1$$ and $$G_2$$ are not isomorphic.*

Further, the author proposed that: **If the neighbor aggregation and graph-level readout function are injective, the resulting GNN is as powerful as the WL-test**. 

##### Graph Isomorphism Network(GIN)

Next, the author shows that the mean and max aggregator commonly seen in GNN are not injective and sum aggregator have the maximum expressive power:

![GIN](/assets/images/GIN.png)

Based on the conclusion,  they propose the GIN, which has been proved to have the same ability as WL-test. In GIN, the node representation is:



$$
h_{v}^{(l)}=\mathrm{MLP}^{(l)}\left(\left(1+\epsilon^{(l)}\right) \cdot h_{v}^{(l-1)}+\sum_{u \in \mathcal{N}(v)} h_{u}^{(l-1)}\right)
$$



Where $MLP$ is a multi-layer perceptron. $$\epsilon^{(k)}$$ is a learnable parameter. To make the graph-level readout become injective, GIN proposed a new mechanism that combine different depth/iterations of the model, which is similar to JKNet:



$$
h_{G}=\operatorname{CONCAT}\left(\operatorname{READOUT}\left(\left\{h_{v}^{(l)} \mid v \in G\right\}\right) \mid l=0,1, \ldots, L\right)
$$






### 5. More powerful GNNs than WL-test

After we prove the representational ability of typical GNNs, one may asking, is there any method that can make GNNs more powerful? Recently, several  works try to tackle this question.

#### 5.1 Identity-aware Graph neural Network(ID-GNN)[<sup>16</sup>](16)

The main idea of ID-GNN is that, to embed a given node $$v$$ using $$K$$-layer GNN, we first extract the $$K$$-hop ego network around node $v$. Then, we assign the center node with different color and use different $MSG$ component to embed the node. Finally, we only use the embedding of node $v$ as the final embedding. The algorithm is described in below:

![IDGNN](/assets/images/IDGNN.png)

One big attribute of ID-GNN is that it can be incorporate into the most of massage passing architectures. In the paper, the author prove that the ID-GNN with GIN model are at least as powerful as original GIN and can discriminate several type of graphs that original GIN cannot. Here is one example:

![IDGNN2](/assets/images/IDGNN2.png)

In each case, if you use typical massage passing GNN, you will get same embedding of node $$A$$ and $$B$$ (suppose the features for each node in graph are identical), since the computational graph of two node are identical. However, if we use ID-GNN and assign the center node with different color, the computational graph become different and thus the embedding are not identical.



#### 5.2 Distance Encoding[<sup>17</sup>](17)

One type of graph that typical GNNs and WL-test cannot distinguish is $$r$$-regular graph. Here is a 3-regular graph with 8 nodes:

![DE](/assets/images/DE.png)

If we use typical GNNs, we will get same embedding for all the nodes(assume the node attribute is the same). However, nodes with different color should have different embedding, since as they are not structurally equivalent.  Further,  typical GNNs cannot distinguish node pair (like $$\{v_1,v_2\}$$ and $$\{v_4,v_7\}$$). However, if we use shortest-path distances (SPDs) between nodes as features we can distinguish blue nodes from green and red nodes because there is another node with SPD= 3 to a blue node of interest (SPD between $$v_3$$ and $$v_8$$), while all SPDs between other nodes to red/green nodes are less than 3.  

The idea of Distance Encoding is simple. By adding addition feature which characterize the distance of target node to the other node sets, we can potential distinguish more type of graph. In the paper, author proposed three different type of distance encoding: shortest-path-distance, generalized PageRank score and random walk. Then the distance encoding is added as extra feature into massage passing GNNs. 

Furthermore, the author use the distance encoding to control the aggregation procedure of GNN.



###  Reference

<div id="refer1">[1] Z. Zhang, P. Cui, and W. Zhu, “Deep Learning on Graphs: A Survey,” *IEEE Trans. Knowl. Data Eng.*, p. 1, 2020, doi: 10.1109/TKDE.2020.2981333.</div>

<div id="refer2">[2] T. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” Sep. 2016.</div>

<div id="refer3">[3] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive Representation Learning on Large Graphs,” Jun. 2017, Accessed: Oct. 09, 2020. [Online]. Available: https://arxiv.org/abs/1706.02216.</div>

<div id="refer4">[4] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, “Graph Attention Networks,” Oct. 2017, Accessed: Oct. 07, 2020. [Online]. Available: http://arxiv.org/abs/1710.10903.</div>

<div id="refer5">[5] Luong, M.-T.; Pham, H. & Manning, C. D. (2015), ‘Effective approaches to attention-based neural machine translation’, *arXiv preprint arXiv:1508.04025* .</div>

<div id="refer6">[6] A. Vaswani et al., “Attention Is All You Need,” Jun. 2017.</div>

<div id="refer7">[7] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling Relational Data with Graph Convolutional Networks BT - The Semantic Web,” 2018, pp. 593–607.</div>

<div id="refer8">[8] Q. Li, Z. Han, and X.-M. Wu, “Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning,” AAAI 2018.</div>

<div id="refer9">[9] W. Huang, Y. Rong, T. Xu, F. Sun, and J. Huang, “Tackling Over-Smoothing for General Graph Convolutional Networks,” Aug. 2020, Accessed: Nov. 16, 2020. [Online]. Available: https://arxiv.org/abs/2008.09864.</div>

<div id="refer10">[10] Y. Rong, W. Huang, T. Xu, and J. Huang, “DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. BT - 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020.” 2020, [Online]. Available: https://openreview.net/forum?id=Hkx1qkrKPr.</div>

<div id="refer11">[11] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” *CoRR*, vol. abs/1512.03385, 2015, [Online]. Available: http://arxiv.org/abs/1512.03385.</div>

<div id="refer12">[12] K. Xu, C. Li, Y. Tian, T. Sonobe, K. Kawarabayashi, and S. Jegelka, “Representation Learning on Graphs with Jumping Knowledge Networks,” *CoRR*, vol. abs/1806.0, 2018, [Online]. Available: http://arxiv.org/abs/1806.03536.</div>

<div id="refer13">[13] J. Klicpera, A. Bojchevski, and S. Günnemann, “Predict then Propagate: Graph Neural Networks meet Personalized PageRank,” Oct. 2018, Accessed: Feb. 25, 2021. [Online]. Available: https://arxiv.org/abs/1810.05997.</div>

<div id="refer14">[14] Boris Weisfeiler and AA Lehman. A reduction of a graph to a canonical form and an algebra arising
during this reduction. Nauchno-Technicheskaya Informatsia, 2(9):12–16, 1968.  </div>

<div id="refer15">[15] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How Powerful are Graph Neural Networks?,” Oct. 2018, Accessed: Jan. 04, 2021. [Online]. Available: https://arxiv.org/abs/1810.00826.</div>

<div id="refer16">[16] J. You, J. M. Gomes-Selman, R. Ying, and J. Leskovec, “Identity-aware Graph Neural Networks,” *ArXiv*, vol. abs/2101.10320, 2021.</div>

<div id="refer17">[17] P. Li, Y. Wang, H. Wang, and J. Leskovec, “Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning,” Aug. 2020, Accessed: Feb. 25, 2021. [Online]. Available: http://arxiv.org/abs/2009.00142.</div>

<div id="refer18">[18] R. Ying, J. You, C. Morris, X. Ren, W. Hamilton, and J. Leskovec, *Hierarchical Graph Representation Learning withDifferentiable Pooling*.NeurIPS 2018.</div>

<div id="refer19">[19] M. Zhang, Z. Cui, M. Neumann, and Y. Chen, “An end-to-end deep learning architecture for graph classification,” AAAI 2018.</div>

<div id="refer20">[20] J. Li, Y. Rong, H. Cheng, H. Meng, W. Huang, and J. Huang, “Semi-Supervised Graph Classification: A Hierarchical Graph Perspective,” Apr. 2019, Accessed: Oct. 09, 2020. [Online]. Available: https://arxiv.org/abs/1904.05003.</div>

<div id="refer21">[21] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural Message Passing for Quantum Chemistry,” in Proceedings of the 34th International Conference on Machine Learning - Volume 70, 2017, pp. 1263–1272.</div>