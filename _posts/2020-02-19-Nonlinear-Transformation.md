---
title: "Nonlinear Transformation"
categories:
  - Machine Learning
tags:
  - Learning note
classes: wide

---



When we talk about PLA, we assume that data are linear separable. However, in real world, most time it is not true. How do we use linear method to deal with non-linear data? that's what nonlinear transformation does.

suppose we have following data:

<img src="/assets/images/image-20200220174236408.png" alt="image-20200220174236408" style="zoom:50%;" />

It is not linear separable. But what if we do a transformation like this:

<img src="/assets/images/image-20200220174523544.png" alt="image-20200220174523544" style="zoom:50%;" />

This time the linear function also change:
$$
w^Tx\rightarrow \tilde{w}^T\phi(x)
$$
However, notice that it is still linear function. What is amazing is that in the new space, the data become linear separable. If we map the function in the new space back to original space, we will get a quatric circle.

How does the feature transformation afffect the VC bound? assume the hypothesis sets after transformation be $H_{\phi}$, if we **decide the $\phi$ before we see the data**, the VC bound remain true by using $d_{vc}(H_{\phi})$. If we decide the $\phi$ after we see the data, the $d_{vc}$ for VC bound is no longer $d_{vc}(H_{\phi})$. For example, we first use linear classifier but failed. Then, we do the transformation, the total $d_{vc}$ for this would be $d_{vc}(H_{\phi} \cup H_{linear})$.

Decide the $\phi$ before we see the data don't means that we should choose $\phi$ blindly. Usually, we need to apply domian knowledge(feature engineering) , or we can use some common sets of feature transformation.



### Reference

[1] Learning From Data, Abu-Mostafa, Magdon-Ismail, and Lin.

[2] Lectures of CSE417T in Washington University in St.Louis, Chien-Ju Ho.http://chienjuho.com/courses/cse417t/

