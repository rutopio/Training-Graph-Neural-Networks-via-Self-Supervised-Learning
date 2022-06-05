# Training-Graph-Neural-Networks-via-Self-Supervised-Learning-Experiments-and-Analysis

- Letest Update: June 6th, 2022



![](https://miro.medium.com/max/1400/1*YXiAuTJvZyHgIpl8s_D6mg.gif)

***

## Contrastive Learning

![](https://miro.medium.com/max/911/1*pwIufoZNu2wanqtBQdIrQQ.gif)

- e.g. **SimCLR**, MoCo

  - Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020, November). A simple framework for contrastive learning of visual representations. In International conference on machine learning (pp. 1597-1607). PMLR.

  - https://arxiv.org/pdf/2002.05709.pdf

  - https://github.com/google-research/simclr


![](https://i.imgur.com/1jxcSPW.png)

- Model:
    - $f(\cdot)$: Encoder
    - $g(\cdot)$: Projection Head

- Cosine Similarity: 
    - $$\textbf{sim}(\textbf{u},\textbf{v})=\textbf{u}^T\textbf{v}/ (\|\textbf{u}\|_2\|\textbf{v}\|_2)$$

- Contrastive Loss Function: NT-Xent
  - $$\mathcal{loss}(i,j)=-\log\frac{\exp\big(\textbf{sim}(\textbf{z}_{i},\textbf{z}_{j})/\tau\big)}{\sum^{2N}_{k=1}\mathbb{1}_{[k\neq i]}\exp\big(\textbf{sim}(\textbf{z}_{i},\textbf{z}_{j}/\tau)\big)}$$

***

## Clustering Learning

- e.g. **SwAV**, SeLA, DeepClustering

  - Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., & Joulin, A. (2020). Unsupervised learning of visual features by contrasting cluster assignments. arXiv preprint arXiv:2006.09882.

  - https://arxiv.org/pdf/2006.09882.pdf

  - https://github.com/facebookresearch/swav


- Loss Function:

  - $$\mathcal{loss}(\mathbf{z}_t,\mathbf{q}_s)=-\sum_k\mathbf{q}^{(k)}_s\log\mathbf{p}_t^{(k)}$$

  - $$\mathbf{p}_t^{(k)}=\frac{\exp\big(\frac{1}{\tau}\mathbf{z}^T_t\cdot\mathbf{c}_k\big)}{\sum_k \exp\big(\frac{1}{\tau}\mathbf{z}^T_t\cdot\mathbf{c}_k^{'}\big)}$$

***

## Distillation Learning

- e.g. BYOL, **SimSiam**

  - Chen, X., & He, K. (2021). Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 15750-15758).

  - https://arxiv.org/pdf/2011.10566.pdf

  - https://github.com/facebookresearch/simsiam


![](https://i.imgur.com/Sy8MLRC.png)

- Model:
    - $f(\cdot)$: Encoder
    - $h(\cdot)$: **Predictor**

- Denote:
    - $p_1=h(f(x_1))$
    - $z_2=f(x_2)$

- Negative Cosine similarity:

    - $$\mathcal{D}(p_1, z_2)=-\bigg(\frac{p_1}{\|p_1\|_2}\cdot\frac{z_2}{\|z_2\|_2}\bigg)$$

    - $$\mathcal{D}\in\{-1(\text{absoulty the same}),0(\text{absoulty different})\}$$

***

## Redundancy reduction

- e.g. **Barlow Twins**

  - Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow twins: Self-supervised learning via redundancy reduction. arXiv preprint arXiv:2103.03230.

  - https://arxiv.org/pdf/2103.03230.pdf

  - https://github.com/facebookresearch/barlowtwins


![](https://user-images.githubusercontent.com/14848164/120419539-b0fab900-c330-11eb-8536-126ce6ce7b85.png)


​    
- Model:
    - $f(\cdot)$: Encoder
    - $g(\cdot)$: Projection Head
    
- Cross-correlation Matrix 交互相關矩陣
    - $b$: indexes batch samples
    - $i, j$: index the vector dimension of the networks’ output
    
    - $$\mathcal{C}_{ij}=\frac{\sum_b\big(z^A_{b,i}\big)\cdot\big(z^B_{b,j}\big)}{\sqrt{\sum_b\big(z^A_{b,i}\big)}\cdot\sqrt{\sum_b\big(z^B_{b,j}\big)}}$$
    
    - $$\mathcal{C}\in\{-1(\text{perfect correlation}),1(\text{perfect anti-correlation})\}$$



- Innovative Loss Function  
    - $\lambda$: a positive constant trading off the importance of the first and second terms of the loss
    
    - $$\mathcal{D}=\underbrace{\sum_i(1-\mathcal{C}_{ii})^2}_{\text{invariance term}}+\lambda\underbrace{\sum_i\sum_{i\neq j}\mathcal{C}_{ij}^2}_{\text{redundancy reduction term}}$$
