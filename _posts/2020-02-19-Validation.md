---
title: "Validation"
categories:
  - Machine Learning
tags:
  - Learning note
classes: wide

---



Validation is another technique used to fight overfitting. In regularization, we are actually trying to estimate follow:


$$
E_{\mathrm{out}}(h)=E_{\mathrm{in}}(h)+\underbrace{\text { overfit penalty }}_{\text {regularization estimates this quantity }}
$$


Validation, on the other hand, try to do follow thing:


$$
\underbrace{E_{\mathrm{out}}(h)}_{\text{validation estimates this quantity}}= E_{\mathrm{in}}(h)+\text { overfit penalty }
$$


To estimate out-of-sample error directly.

### 1. Test set

The main idea behind test set is to reserve some data from original dataset to estimate out-of-sample error:


$$
E_{o u t}(g)=E_{\vec{x}}[e(g(\vec{x}), y)]
$$


Where $\vec{x}$ should be out of sample. Typically, we would reserve $K$ data points used to estimate $E_{out}$, those data points cannot be involved in training:$D=D_{\text {train}} \cup D_{\text {test}}$.  Let test dataset be $$D_{\text {test}}=\left\{\left(\vec{x}_{1}, y_{1}\right), \ldots,\left(\vec{x}_{K}, y_{K}\right)\right\}$$, and $g^-$ be the hypothesis we learned use $$ D_{train}$$, We can calculate $$E_{test}(g)$$:


$$
E_{\text {test}}(g)=\frac{1}{K} \sum_{k=1}^{K} e\left(g\left(\vec{x}_{k}\right), y_{k}\right)
$$


$E_{test}(g)$ is an unbaised estimate of $E_{out}(g)$. since:


$$
\mathbb{E}\left[E_{\text {test}}(g)\right]=\frac{1}{K} \sum_{k=1}^{K} \mathbb{E}\left[e\left(g\left(\vec{x}_{k}\right), y_{k}\right)\right]=E_{\text {out}}(g)
$$


and we can applies Hoeffding bound to this:


$$
E_{o u t}(g) \leq E_{\text {test}}(g)+O(\sqrt{\frac{1}{K}})
$$


Since test data is choosen from original dataset, a large $K$ always means we have less data for training. The effect of the choice of $K$ could be like this. Usually, we would select $K=\frac{N}{5}$

<img src="/assets/images/image-20200223181707690.png" alt="image-20200223181707690" style="zoom:50%;" />

### 2. Validation set

What if we want to estimate $E_{out}$ for multiple times? like model selection. This is where validation set comes in. There is no difference for validation set and test set itself, since $D=D_{\text {train}} \cup D_{\text {val}}$. The difference is that we need use $D_{val}$ for multiple times and $D_{val}$ would affects the learning process.

For a single model $g^-$,  what we have in test set still hold:


$$
E_{\mathrm{out}}(g) \leq E_{\mathrm{out}}\left(g^{-}\right) \leq E_{\mathrm{val}}\left(g^{-}\right)+O\left(\frac{1}{\sqrt{K}}\right)
$$


where $g$ is hypothesis learned by $D$. Meanwhile, since $D_{val}$ is actually subset of $D$, we have $$\mathbb{E}\left[E_{v a l}\left(g_{m^{*}}^{-}\right)\right] \leq E_{\text {out}}\left(g_{m^{*}}^{-}\right)$$, where $g_{m^{*}}^{-}$ is a hypothesis we would explain in the next part.

<img src="/assets/images/image-20200223192438486.png" alt="image-20200223192438486" style="zoom:50%;" />

Typcially, the relationship between $$E_{\text {out}}\left(g_{m^{*}}^{-}\right)$$ and $$E_{v a l}\left(g_{m^{*}}^{-}\right)$$ is illsutrated in above plot.  When $K$ become large,  $$E_{v a l}\left(g_{m^{*}}^{-}\right)$$ would quickly approach to  $$E_{\text {out}}\left(g_{m^{*}}^{-}\right)$$.



When we use $D_{val}$ to do the validation, let $$\mathcal{H}_{1}, \ldots, \mathcal{H}_{M}$$ be the model we want to validate. Use the $D_{train}$ to train a $g^-_{m}$ for each model. Now, evaluate each model on the validation set to obtain the validation errors $$E_{1}, \cdots, E_{M}$$, where $$E_{m}=E_{\mathrm{val}}\left(g_{m}^{-}\right) ; \quad m=1, \ldots, M$$.

Now, a simple initutituion is select the model with lowest $$E_{val}(g^-_{m})$$, denote as model $$g^-_{m^*}$$, $$E_{v a l}\left(g_{m^{*}}^{-}\right) \leq E_{v a l}\left(g_{m}^{-}\right) \text {for all } m$$.

What can we say about the generalization of validation?  Consider a new model $\mathcal{H}_{val}$ contain all the final hypotheses,


$$
\mathcal{H}_{\mathrm{val}}=\left\{g_{1}^{-}, g_{2}^{-}, \ldots, g_{\mathrm{M}}^{-}\right\}
$$


Model selection is actually use $D_{val}$ to choose one of $$g^-_{m}$$ based on "in sample error" $$E_{val}(g^-_{m})$$.  Thus, we actually have this:


$$
E_{\mathrm{out}}\left(g_{m^{*}}\right) \leq E_{\mathrm{out}}\left(g_{m^{*}}^{-}\right) \leq E_{\mathrm{val}}\left(g_{m^{*}}^{-}\right)+O(\sqrt{\frac{\ln M}{K}})
$$



### 3. Cross Validation

The main idea behind validation is 


$$
E_{o u t}(g) \approx E_{o u t}\left(g^{-}\right) \approx E_{v a l}\left(g^{-}\right)
$$


However, we want to get a good estimation of $E_{out}(g^-)$ by  $E_{val}(g^-)$, which need a large $K$. Meanwhile, we also want to get a good estimation of $E_{out}(g)$ by $E_{out}(g^-)$, which expect a small $K$. This is a dilemma. That is where cross validation comes in.

#### Leave-one-out Cross Validation(LOOCV)

Every time, we use one one data point as validation set and other to train the model, thus we can get a estimation. then, repeat this for $N$ times with every time use different validation data. The final estimation of error is the avarage of all $N$ errors:


$$
\begin{aligned}
\mathrm{e}_{n}=&E_{\mathrm{val}}\left(g_{n}^{-}\right)=\mathrm{e}\left(g_{n}^{-}\left(\mathbf{x}_{n}\right), y_{n}\right)  \\
E_{\mathrm{cv}}=&\frac{1}{N} \sum_{n=1}^{N} \mathrm{e}_{n}
\end{aligned}
$$


There are several advantages to use LOOCV: first, it is a unbaised estimation of $E_{out}$. Second, LOOCV would have same result no matter how we choose train and validation data. However, LOOCV is computational expensive.

#### K-fold Cross Validation

K-fold cross validation try to addresses issue in LOOCV and normal validation approach. This time, we equally divide whole dataset to $K$ parts. everytime, we use $K-1$ parts to train the model and use the remain part as validation set. The estimation of $E_{out}$ is:


$$
E_{\mathrm{cv}}=\frac{1}{K} \sum_{k=1}^{K} \mathrm{E(g^{-})}_{k}
$$


Normally, LOOCV would have less bias but high variance than K-fold cross validation. When $K=N$, LOOCV and K-fold cross validation would be same. Usually, we will choose k as 5 or 10.

### Reference

[1] Learning From Data, Abu-Mostafa, Magdon-Ismail, and Lin.

[2] Lectures of CSE417T in Washington University in St.Louis, Chien-Ju Ho.http://chienjuho.com/courses/cse417t/

