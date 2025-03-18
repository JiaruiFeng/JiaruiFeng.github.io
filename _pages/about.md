---
permalink: /
title: "Jiarui Feng"
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I'm a Fourth-year Ph.D. candidate in the Department of Computer Science and Engineering, Washington University in St.Louis (WashU). I'm fortunately supervised by [Dr. Yixin Chen](https://www.cse.wustl.edu/~yixin.chen/) and also closely work with [Dr. Fuhai Li](https://informatics.wustl.edu/research-lab-fuhai-li/). 
My research primarily focuses on graph representation learning. Particularly, most of my research targets an ultimate goal: to design and train a **graph foundation model (GFMs)** with **graph neural networks (GNNs)**. Specifically:

- Understanding and improving the **expressiveness** and **structure learning** ability of GNNs on graphs. 
- **Integrate GNNs with LLMs** to design the architecture of GFMs and enable a single model to solve graph tasks in multiple levels (node/link/graph) and domains (citation/molecule/e-commercial). 
- Design unified self-supervised pre-training tasks and graph prompting mechanism on graph domain to enable zero-shot transfer ability in GFMs.
- Collect and create large-scale graph datasets for the training of GFMs.

Besides that, I am also interested in the downstream application of GNNs in various domains like **recommendation, precision medicine, planning, and reasoning**. 

## I am actively looking for research internship in summer 2025. If you are interested in my work, feel free to chat or shoot me email for discussing any potential opportunities.

# 🔥 News
- *2025.01*: [GOFA](https://arxiv.org/abs/2407.09709) is accepted by ICLR 2025!
- *2024.09*: [GNN4TaskPlan](https://arxiv.org/pdf/2405.19119) is accepted by NeurIPS 2024, Congratulations to Xixi and Yifei!
- *2024.08*: Check out our newest work on joint modeling of graph and language ([paper](https://arxiv.org/abs/2407.09709),[code](https://github.com/JiaruiFeng/GOFA)). In this work, we propose GOFA, which interleave GNN layers into LLMs to enable LLM with the ability to reason on graph. We also design multiple novel large-scale unsupervised pretraining tasks for GOFA. GOFA achieves SOTA results across multiple benchmarking datasets!    
- *2024.06*: We release [TAGLAS](https://github.com/JiaruiFeng/TAGLAS), an atlas of text-attributed graph datasets. We provide easy-to-use APIs for loading datasets, tasks, and evaluation metrics. The technical report is available in [arxiv](https://arxiv.org/abs/2406.14683). The project is still in development, any suggestions are welcome.
- *2024.04*: &nbsp;🎉🎉 [PathFinder](https://www.biorxiv.org/content/10.1101/2024.01.13.575534v1) is accepted by Frontiers in Cellular Neuroscience! 
- *2024.01*: &nbsp;🎉🎉 [COLA](https://arxiv.org/abs/2309.10376) is accepted by WWW 2024! Congratulations to Hao!
- *2024.01*: &nbsp;🎉🎉 [OFA](https://arxiv.org/abs/2310.00149) is accepted by ICLR 2024 as a Spotlight(5%)!
- *2024.01*: &nbsp;🎉🎉 [sc2MeNetDrug](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011785) is accepted by PLOS Computational Biology!
- *2023.10*: We have developed a novel R Shiny application **sc2MeNetDrug** for the analysis of single-cell RNA-seq data. This application enables the identification of activated pathways, up-regulated ligands and receptors, cell-cell communication networks, and potential drugs to inhibit dysfunctional networks. Moreover, it provides user-friendly UI for easy usage! Check out our [GitHub repository](https://github.com/fuhaililab/sc2MeNetDrug) and [website](https://fuhaililab.github.io/sc2MeNetDrug/) for more details. This project is still ongoing, and we welcome any comments or suggestions!
- *2023.10*: Leveraging the power of language and LLMs, we propose **One-for-ALL (OFA)**, which is the first general framework that can use a single graph model to address (almost) all different graph classification tasks from different domains. Check out our [preprint](https://arxiv.org/abs/2310.00149) and [code](https://github.com/LechengKong/OneForAll)!
- *2023.09*: &nbsp;🎉🎉 [(k,t)-FWL+](https://arxiv.org/abs/2306.03266), [MAG-GNN](https://arxiv.org/pdf/2310.19142v1.pdf), and [d-DRFWL2(spotlight)](https://arxiv.org/pdf/2309.04941.pdf) are accepted by NeurIPS 2023!
- *2023.06*: &nbsp;🎉🎉 Passed the oral exam!
- *2022.09*: &nbsp;🎉🎉 Our paper "[How powerful are K-hop message passing graph neural networks](https://arxiv.org/abs/2205.13328)" is accepted by NeurIPS 2022. See you in New Orleans!
- *2022.08*: &nbsp;🎉🎉 Our paper "[Reward delay attacks on deep reinforcement learning](https://arxiv.org/abs/2209.03540)" is accepted by GameSec 2022.


# 📝 Selected Publications
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2025/div><img src='images/GOFA.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[GOFA: A Generative One-For-All Model for Joint Graph Language Modeling](https://arxiv.org/abs/2407.09709)

Lecheng Kong<sup>*</sup>, **Jiarui Feng**<sup>*</sup>, Hao Liu<sup>*</sup>, Chengsong Huang, Jiaxin Huang, Yixin Chen, Muhan Zhang (<sup>*</sup> Equal contribution)\\
<a href="https://arxiv.org/abs/2407.09709"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/JiaruiFeng/GOFA"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=mIjblC9hfm"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICLR%2725&color=yellow"> </a>
<a href="https://github.com/JiaruiFeng/GOFA"><img src="https://img.shields.io/github/stars/JiaruiFeng/GOFA?style=social" alt="Github"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024 Spotlight</div><img src='images/OFA.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[One for All: Towards Training One Graph Model for All Classification Tasks](https://arxiv.org/abs/2310.00149)

Hao Liu<sup>*</sup>, **Jiarui Feng**<sup>*</sup>, Lecheng Kong<sup>*</sup>, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang (<sup>*</sup> Equal contribution)\\
<a href="https://arxiv.org/abs/2310.00149"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/LechengKong/OneForAll"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=4IT2pgc9v6"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=ICLR%2724&color=yellow"> </a>
<a href="https://github.com/LechengKong/OneForAll"><img src="https://img.shields.io/github/stars/LechengKong/OneForAll?style=social" alt="Github"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">WWW 2024</div><img src='images/COLA.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks](https://arxiv.org/abs/2309.10376)

Hao Liu, **Jiarui Feng**, Lecheng Kong, Dacheng Tao, Yixin Chen, Muhan Zhang \\
<a href="https://arxiv.org/abs/2309.10376"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/Haoliu-cola/COLA/"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://arxiv.org/abs/2309.10376"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=WWW%2724&color=yellow"> </a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2023</div><img src='images/neighborhood_tuple.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Extending the Design Space of Graph Neural Networks by Rethinking Folklore Weisfeiler-Lehman](https://arxiv.org/abs/2306.03266)

**Jiarui Feng**, Lecheng Kong, Hao Liu, Dacheng Tao, Fuhai Li, Muhan Zhang, Yixin Chen \\
<a href="https://arxiv.org/abs/2306.03266"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/JiaruiFeng/N2GNN"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=UlJcZoawgU"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2723&color=yellow"> </a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2023 Spotlight</div><img src='images/drfwl2.png' alt="sym" width="65%" height="65%"></div></div>
<div class='paper-box-text' markdown="1">

[Distance-Restricted Folklore Weisfeiler-Leman GNNs with Provable Cycle Counting Power](https://arxiv.org/abs/2309.04941)

Junru Zhou, **Jiarui Feng**, Xiyuan Wang, Muhan Zhang \\
<a href="https://arxiv.org/abs/2309.04941"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/zml72062/DR-FWL-2"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=94rKFkcm56"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2723&color=yellow"> </a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2023</div><img src='images/maggnn.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[MAG-GNN: Reinforcement Learning Boosted Graph Neural Network](https://arxiv.org/abs/2310.19142)

Lecheng Kong, **Jiarui Feng**, Hao Liu, Dacheng Tao, Yixin Chen, Muhan Zhang \\
<a href="https://arxiv.org/abs/2310.19142"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/LechengKong/MAG-GNN"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=K4FK7I8Jnl"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2723&color=yellow"> </a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2022</div><img src='images/khop.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[How powerful are K-hop message passing graph neural networks](https://arxiv.org/abs/2205.13328)

**Jiarui Feng**, Yixin Chen, Fuhai Li, Anindya Sarkar, Muhan Zhang \\
<a href="https://arxiv.org/abs/2205.13328"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/JiaruiFeng/KP-GNN"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://openreview.net/forum?id=nN3aVRQsxGd&referrer=%5Bthe%20profile%20of%20Jiarui%20Feng%5D(%2Fprofile%3Fid%3D~Jiarui_Feng1)"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2722&color=yellow"> </a>
<a href="https://github.com/JiaruiFeng/KP-GNN"><img src="https://img.shields.io/github/stars/JiaruiFeng/KP-GNN?style=social" alt="Github"></a>
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">GameSec 2022</div><img src='images/rewarddelay.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[Reward Delay Attacks on Deep Reinforcement Learning](https://arxiv.org/abs/2209.03540)

Anindya Sarkar, **Jiarui Feng**, Yevgeniy Vorobeychik, Christopher Gill, Ning Zhang \\
<a href="https://link.springer.com/chapter/10.1007/978-3-031-26369-9_11"><img src="https://img.shields.io/badge/-Paper-grey?logo=gitbook&logoColor=white" alt="Paper"></a>
<a href="https://github.com/anindyasarkarIITH/Reward_Delay_Attack_DRL"><img src="https://img.shields.io/badge/-Github-blue?logo=github" alt="Github"></a>
<a href="https://link.springer.com/chapter/10.1007/978-3-031-26369-9_11"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=GameSec%2722&color=yellow"> </a>
</div>
</div>

You can browse my full publication list in [Google Scholar](https://scholar.google.com/citations?user=6CSGUR8AAAAJ&hl=zh-CN)

# 🎖 Honors and Awards
- *2023.10* NeurIPS 2023 Travel Award.
- *2021.07* ICIBM 2021 Travel Award. 


# 📖 Educations
- *2021.09 - Present*, Ph.D student, Washington University in St. Louis, MO, USA.
- *2019.09 - 2021.05*, Master, Washington University in St. Louis, MO, USA.
- *2015.09 - 2019.06*, Bachelor, South China Univerity of Technology, GuangZhou, China.


# 💻 Internships
- *2024.06 - 2024.09* Lab research intern, Pinterest, Remote, US. 
- *2019.06 - 2019.08*, SWE intern, Alibaba Cloud, HangZhou, China.
- *2018.12 - 2019.02*, Data analytics intern, Credit card center, GuangZhou Bank, GuangZhou, China.

# 🔬Services
- **Conference reviewer**: CVPR23; NeurIPS23; ICLR24; CVPR24; NeurIPS24


# 🎮 Misc
- Crazy computer gamer: Overwatch, APEX, World of warcraft, PUBG, CS:GO...

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">theta!</div><img src='images/theta.jpeg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
- I have a cute ragdoll called $\theta$, I love him!!!!!!!!!!!!!
</div>
</div>
