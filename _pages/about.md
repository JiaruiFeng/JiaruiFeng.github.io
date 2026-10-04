---
permalink: /
title: "Jiarui Feng"
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<span class='anchor' id='about-me'></span>

I'm currently a Research Scientist at Meta MRS. Prior to that, I received my Ph.D. from the Department of Computer Science and Engineering, Washington University in St. Louis (WashU), where I was fortunately supervised by [Dr. Yixin Chen](https://www.cse.wustl.edu/~yixin.chen/) and also worked closely with [Dr. Fuhai Li](https://engineering.washu.edu/faculty/Fuhai-Li.html). 

My research spans **Graph Neural Networks (GNNs)**, **Large Language Models (LLMs)**, and **Generative Recommendation**. Specifically, I work on:

- **Expressiveness of GNNs** --- characterizing and improving the structure-learning capacity of message-passing architectures. 
- **LLMs and graphs** --- understanding the capabilities and limitations of LLMs on graph tasks, integrating GNNs with LLMs toward graph foundation models, and developing test-time training methods that adapt LLMs to graph reasoning. 
- **Scaling Mixture-of-Experts (MoE)** --- improving the quality and efficiency of sparsely activated models at scale. 
- **Diffusion models for sequential data** --- analyzing discrete and continuous diffusion models on language and recommendation data, and building large-scale generative recommenders based on masked diffusion.

# 🔥 News
- *2026.09*: &nbsp;🎉🎉 [TabDLM](https://arxiv.org/abs/2602.22586) is accepted by NeurIPS 2026! Congratulations to Donghong!
- *2026.08*: &nbsp;🎉🎉 [ReMix](https://arxiv.org/abs/2603.10160) is accepted by COLM 2026! Congratulations to Ruizhong!
- *2026.05*: &nbsp;🎉🎉 [GRIP](https://dl.acm.org/doi/abs/10.1145/3770855.3817895) is accepted by KDD 2026, research track!
- *2026.04*: &nbsp;🎉🎉 [DAG-MoE](https://arxiv.org/abs/2606.01062) is accepted by ICML 2026!
- *2026.04*: &nbsp;🎉🎉 Joined Meta MRS as a Research Scientist!
- *2026.02*: &nbsp;🎉🎉 Successfully passed the Ph.D. thesis defense!
- *2025.07*: &nbsp;🎉🎉 [GRIP](https://openreview.net/forum?id=WacW4lC4du) is accepted by PUT at ICML 2025!
- *2025.04*: &nbsp;🎉🎉 Passed the Ph.D. proposal!
- *2025.01*: &nbsp;🎉🎉 [GOFA](https://arxiv.org/abs/2407.09709) is accepted by ICLR 2025!
- *2024.09*: &nbsp;🎉🎉 [GNN4TaskPlan](https://arxiv.org/abs/2405.19119) is accepted by NeurIPS 2024! Congratulations to Xixi and Yifei!
- *2024.08*: Check out our newest work on joint modeling of graph and language ([paper](https://arxiv.org/abs/2407.09709), [code](https://github.com/JiaruiFeng/GOFA)). In this work, we propose GOFA, which interleaves GNN layers into LLMs to equip them with the ability to reason on graphs. We also design multiple novel large-scale unsupervised pretraining tasks for GOFA. GOFA achieves SOTA results across multiple benchmarking datasets!
- *2024.06*: We release [TAGLAS](https://github.com/JiaruiFeng/TAGLAS), an atlas of text-attributed graph datasets. We provide easy-to-use APIs for loading datasets, tasks, and evaluation metrics. The technical report is available on [arXiv](https://arxiv.org/abs/2406.14683). The project is still in development, and any suggestions are welcome.
- *2024.04*: &nbsp;🎉🎉 [PathFinder](https://www.biorxiv.org/content/10.1101/2024.01.13.575534v1) is accepted by Frontiers in Cellular Neuroscience! 
- *2024.01*: &nbsp;🎉🎉 [COLA](https://arxiv.org/abs/2309.10376) is accepted by WWW 2024! Congratulations to Hao!
- *2024.01*: &nbsp;🎉🎉 [OFA](https://arxiv.org/abs/2310.00149) is accepted by ICLR 2024 as a Spotlight (5%)!
- *2024.01*: &nbsp;🎉🎉 [sc2MeNetDrug](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011785) is accepted by PLOS Computational Biology!
- *2023.10*: We have developed a novel R Shiny application **sc2MeNetDrug** for the analysis of single-cell RNA-seq data. This application enables the identification of activated pathways, up-regulated ligands and receptors, cell-cell communication networks, and potential drugs to inhibit dysfunctional networks. Moreover, it provides user-friendly UI for easy usage! Check out our [GitHub repository](https://github.com/fuhaililab/sc2MeNetDrug) and [website](https://fuhaililab.github.io/sc2MeNetDrug/) for more details. This project is still ongoing, and we welcome any comments or suggestions!
- *2023.10*: Leveraging the power of language and LLMs, we propose **One-for-ALL (OFA)**, which is the first general framework that can use a single graph model to address (almost) all different graph classification tasks from different domains. Check out our [preprint](https://arxiv.org/abs/2310.00149) and [code](https://github.com/LechengKong/OneForAll)!
- *2023.09*: &nbsp;🎉🎉 [(k,t)-FWL+](https://arxiv.org/abs/2306.03266), [MAG-GNN](https://arxiv.org/abs/2310.19142), and [d-DRFWL2 (Spotlight)](https://arxiv.org/abs/2309.04941) are accepted by NeurIPS 2023!
- *2023.06*: &nbsp;🎉🎉 Passed the oral exam!
- *2022.09*: &nbsp;🎉🎉 Our paper "[How powerful are K-hop message passing graph neural networks](https://arxiv.org/abs/2205.13328)" is accepted by NeurIPS 2022. See you in New Orleans!
- *2022.08*: &nbsp;🎉🎉 Our paper "[Reward delay attacks on deep reinforcement learning](https://arxiv.org/abs/2209.03540)" is accepted by GameSec 2022.


# 📝 Selected Publications
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">KDD 2026</div><img src='images/grip.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[GRIP: In-Parameter Graph Reasoning through Fine-Tuning Large Language Models](https://dl.acm.org/doi/abs/10.1145/3770855.3817895)

**Jiarui Feng**, Donghong Cai, Yixin Chen, Muhan Zhang\\
<a href="https://dl.acm.org/doi/abs/10.1145/3770855.3817895"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/JiaruiFeng/GRIP"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://dl.acm.org/doi/abs/10.1145/3770855.3817895"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=KDD%2726&color=yellow"> </a>
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICML 2026</div><img src='images/dag_moe.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](https://arxiv.org/abs/2606.01062)

**Jiarui Feng**, Hanqing Zeng, Karish Grover, Ruizhong Qiu, Yinglong Xia, Qiang Zhang, Qifan Wang, Ren Chen, Dongqi Fu, Jiayi Liu, Zhoukai Zhao, Xiangjun Fan, Benyu Zhang, Yixin Chen\\
<a href="https://arxiv.org/abs/2606.01062"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/JiaruiFeng/DAG-MoE"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=yv1iRquxF9"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=ICML%2726&color=yellow"> </a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2025</div><img src='images/GOFA.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[GOFA: A Generative One-For-All Model for Joint Graph Language Modeling](https://arxiv.org/abs/2407.09709)

Lecheng Kong<sup>*</sup>, **Jiarui Feng**<sup>*</sup>, Hao Liu<sup>*</sup>, Chengsong Huang, Jiaxin Huang, Yixin Chen, Muhan Zhang (<sup>*</sup> Equal contribution)\\
<a href="https://arxiv.org/abs/2407.09709"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/JiaruiFeng/GOFA"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=mIjblC9hfm"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=ICLR%2725&color=yellow"> </a>
<a href="https://github.com/JiaruiFeng/GOFA"><img src="https://img.shields.io/github/stars/JiaruiFeng/GOFA?style=social" alt="Github stars"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024 Spotlight</div><img src='images/OFA.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[One for All: Towards Training One Graph Model for All Classification Tasks](https://arxiv.org/abs/2310.00149)

Hao Liu<sup>*</sup>, **Jiarui Feng**<sup>*</sup>, Lecheng Kong<sup>*</sup>, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang (<sup>*</sup> Equal contribution)\\
<a href="https://arxiv.org/abs/2310.00149"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/LechengKong/OneForAll"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=4IT2pgc9v6"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=ICLR%2724&color=yellow"> </a>
<a href="https://github.com/LechengKong/OneForAll"><img src="https://img.shields.io/github/stars/LechengKong/OneForAll?style=social" alt="Github stars"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">WWW 2024</div><img src='images/COLA.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks](https://arxiv.org/abs/2309.10376)

Hao Liu, **Jiarui Feng**, Lecheng Kong, Dacheng Tao, Yixin Chen, Muhan Zhang \\
<a href="https://arxiv.org/abs/2309.10376"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/Haoliu-cola/COLA/"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://arxiv.org/abs/2309.10376"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=WWW%2724&color=yellow"> </a>
<a href="https://github.com/Haoliu-cola/COLA/"><img src="https://img.shields.io/github/stars/Haoliu-cola/COLA?style=social" alt="Github stars"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2023</div><img src='images/neighborhood_tuple.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Extending the Design Space of Graph Neural Networks by Rethinking Folklore Weisfeiler-Lehman](https://arxiv.org/abs/2306.03266)

**Jiarui Feng**, Lecheng Kong, Hao Liu, Dacheng Tao, Fuhai Li, Muhan Zhang, Yixin Chen \\
<a href="https://arxiv.org/abs/2306.03266"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/JiaruiFeng/N2GNN"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=UlJcZoawgU"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2723&color=yellow"> </a>
<a href="https://github.com/JiaruiFeng/N2GNN"><img src="https://img.shields.io/github/stars/JiaruiFeng/N2GNN?style=social" alt="Github stars"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2023 Spotlight</div><img src='images/drfwl2.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Distance-Restricted Folklore Weisfeiler-Leman GNNs with Provable Cycle Counting Power](https://arxiv.org/abs/2309.04941)

Junru Zhou, **Jiarui Feng**, Xiyuan Wang, Muhan Zhang \\
<a href="https://arxiv.org/abs/2309.04941"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/zml72062/DR-FWL-2"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=94rKFkcm56"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2723&color=yellow"> </a>
<a href="https://github.com/zml72062/DR-FWL-2"><img src="https://img.shields.io/github/stars/zml72062/DR-FWL-2?style=social" alt="Github stars"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2023</div><img src='images/maggnn.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[MAG-GNN: Reinforcement Learning Boosted Graph Neural Network](https://arxiv.org/abs/2310.19142)

Lecheng Kong, **Jiarui Feng**, Hao Liu, Dacheng Tao, Yixin Chen, Muhan Zhang \\
<a href="https://arxiv.org/abs/2310.19142"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/LechengKong/MAG-GNN"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=K4FK7I8Jnl"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2723&color=yellow"> </a>
<a href="https://github.com/LechengKong/MAG-GNN"><img src="https://img.shields.io/github/stars/LechengKong/MAG-GNN?style=social" alt="Github stars"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2022</div><img src='images/khop.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[How powerful are K-hop message passing graph neural networks](https://arxiv.org/abs/2205.13328)

**Jiarui Feng**, Yixin Chen, Fuhai Li, Anindya Sarkar, Muhan Zhang \\
<a href="https://arxiv.org/abs/2205.13328"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/JiaruiFeng/KP-GNN"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=nN3aVRQsxGd"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2722&color=yellow"> </a>
<a href="https://github.com/JiaruiFeng/KP-GNN"><img src="https://img.shields.io/github/stars/JiaruiFeng/KP-GNN?style=social" alt="Github stars"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">GameSec 2022</div><img src='images/rewarddelay.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Reward Delay Attacks on Deep Reinforcement Learning](https://arxiv.org/abs/2209.03540)

Anindya Sarkar, **Jiarui Feng**, Yevgeniy Vorobeychik, Christopher Gill, Ning Zhang \\
<a href="https://link.springer.com/chapter/10.1007/978-3-031-26369-9_11"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/anindyasarkarIITH/Reward_Delay_Attack_DRL"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://link.springer.com/chapter/10.1007/978-3-031-26369-9_11"> <img alt="Publication venue" src="https://img.shields.io/static/v1?label=Pub&message=GameSec%2722&color=yellow"> </a>
</div>
</div>

You can browse my full publication list on [Google Scholar](https://scholar.google.com/citations?user=6CSGUR8AAAAJ).

# 🎖 Honors and Awards
- *2023.10*: NeurIPS 2023 Travel Award.
- *2021.07*: ICIBM 2021 Travel Award.


# 📖 Educations
- *2021.09 - 2026.02*: Ph.D., Washington University in St. Louis, MO, USA.
- *2019.09 - 2021.05*: M.S., Washington University in St. Louis, MO, USA.
- *2015.09 - 2019.06*: B.S., South China University of Technology, Guangzhou, China.


# 💻 Internships
- *2025.09 - 2025.12*: Student Researcher, Meta, Remote, US.
- *2025.05 - 2025.08*: Research Intern, Meta, Menlo Park, US.
- *2024.06 - 2024.09*: Lab Research Intern, Pinterest, Remote, US.
- *2019.06 - 2019.08*: SWE Intern, Alibaba Cloud, Hangzhou, China.
- *2018.12 - 2019.02*: Data Analytics Intern, Credit Card Center, Guangzhou Bank, Guangzhou, China.

# 🔬 Services
- **Conference reviewer**: CVPR23; NeurIPS23; ICLR24; CVPR24; NeurIPS24; ICLR25; ICML25; CVPR25; NeurIPS25; ICLR26.


# 🎮 Misc
- Crazy computer gamer: Overwatch, Apex Legends, World of Warcraft, PUBG, CS:GO...

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">theta!</div><img src='images/theta.jpeg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

- I have a cute ragdoll called $\theta$, and I love him!!!
</div>
</div>
