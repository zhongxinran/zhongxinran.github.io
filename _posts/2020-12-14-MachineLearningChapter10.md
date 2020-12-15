---
title: 西瓜书 | 第十章 降维与度量学习
author: 钟欣然
date: 2020-12-13 04:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

# k近邻学习
k近邻（k-Nearest Neighbor, kNN）学习是一种常用的监督学习方法，给定测试样本，基于某种距离度量找出训练集中与其最靠近的k个样本，然后基于这k个邻居的信息进行预测，分类任务中可使用投票法，回归任务中可使用平均法，还可基于距离远近进行加权平均或加权投票，距离越近权重越大；k取不同值时，分类结果会有显著不同

- 懒惰学习（lazy learning）：在训练阶段仅仅是把样本保存起来，训练时间开销为零，待收到测试样本后在再进行处理
	- kNN没有显式的训练过程，属于懒惰学习
- 急切学习（eager learning）：在训练阶段就对样本进行学习处理

**下面对最近邻分类器（1NN）在二分类问题上的性能做一个简单的讨论**

给定测试样本$\pmb x$，若其最近邻样本为$\pmb z$，则最近邻分类器出错的概率就是二者类别标记不同的概率

$$
P(e r r)=1-\sum_{c \in \mathcal{Y}} P(c \vert \pmb{x}) P(c \vert \pmb{z})
$$

假设样本独立同分布，且对任意小正数$\delta$，在$\pmb x$附近$\delta$距离范围内总能找到一个训练样本，令$c^{*}=\arg \max _{c \in \mathcal{Y}} P(c \vert \pmb{x})$表示贝叶斯最优分类器的结果，有

$$
\begin{aligned} P(e r r) &=1-\sum_{c \in \mathcal{Y}} P(c | \pmb{x}) P(c | \pmb{z}) \\ & \simeq 1-\sum_{c \in \mathcal{Y}} P^{2}(c | \pmb{x}) \\ & \leqslant 1-P^{2}\left(c^{*} | \pmb{x}\right) \\ &=\left(1+P\left(c^{*} | \pmb{x}\right)\right)\left(1-P\left(c^{*} | \pmb{x}\right)\right) \\ & \leqslant 2 \times\left(1-P\left(c^{*} | \pmb{x}\right)\right) \end{aligned}
$$

即最近邻分类器虽然简单，但它的泛化错误率不超过贝叶斯最优分类器错误率的两倍

# 2. 低维嵌入

## 维数灾难与降维
**密采样**：

上节讨论基于一个重要的假设，在任意测试样本$\pmb x$附近任意小的$\delta$距离范围内总能找到一个训练样本，即训练样本的采样密度足够大，或称为密采样（dense sample），然而这个假设在现实任务中很难满足

**维数灾难**：

- 现实应用中属性维数经常成千上万，要满足密采样条件所需的样本数目是无法达到的天文数字
- 许多学习方法涉及距离计算，高维空间会给距离计算带来很大的麻烦

高维情形下出现的数据样本稀疏、距离计算困难等问题，是所有机器学习方法共同面临的严重障碍，被称为维数灾难（curse of dimensionality）

**降维**：

缓解维数灾难的一个重要途径就是降维（dimension reduction），也称为维数约简，即通过某种数学变换将原始高维属性空间转变为一个低维子空间（subspace），在这个子空间样本密度大幅提高，距离计算也更为容易

为什么能进行降维？很多时候人们观测或收集到的数据样本虽然是高维的，但与学习任务密切相关的也许仅是某个地位分布，即高维空间中的一个低维嵌入

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118175254163.png#pic_center)

## 多维缩放
若降维时要求原始样本空间中样本之间的距离在低维空间中得以保持，则为多维缩放（Multiple Dimensional Scaling, MDS）

假定m个样本在原始空间的距离矩阵为$\mathbf{D} \in \mathbb{R}^{m \times m}$，元素$dist_{ij}$表示样本$\pmb x_i$到$\pmb x_j$的距离，目标是获得样本在$d'$维空间的表示$\mathbf{Z} \in \mathbb{R}^{d' \times m},d'\leq d$，且任意两个样本在$d'$维空间中的欧氏距离等于原始空间中的距离，即$\left\|z_{i}-z_{j}\right\|=d i s t_{i j}$

令$\mathbf{B}=\mathbf{Z}^{\mathrm{T}} \mathbf{Z} \in \mathbb{R}^{m \times m}$，其中$\mathbf{B}$为降维后样本的内积矩阵，$b_{i j}=z_{i}^{\mathrm{T}} z_{j}$，因此有

$$
\begin{aligned} d i s t_{i j}^{2} &=\left\|z_{i}\right\|^{2}+\left\|z_{j}\right\|^{2}-2 z_{i}^{\mathrm{T}} z_{j} \\ &=b_{i i}+b_{j j}-2 b_{i j} \end{aligned}
$$

为便与讨论，令降维后的样本$\mathbf{Z}$被中心化，即$\sum_{i=1}^{m} \pmb{z}\_{i}=\mathbf{0}$，则有矩阵$\mathbf{B}$的行和、列和均为零，即$\sum_{i=1}^{m} b\_{i j}=\sum_{j=1}^{m} b\_{i j}=0$，因此有

$$
\sum_{i=1}^{m} d i s t_{i j}^{2}=\operatorname{tr}(\mathbf{B})+m b_{j j}$$ $$\sum_{i=1}^{m} d i s t_{i j}^{2}=\operatorname{tr}(\mathbf{B})+m b_{j j}$$ $$\sum_{i=1}^{m} \sum_{j=1}^{m} d i s t_{i j}^{2}=2 m \operatorname{tr}(\mathbf{B})
$$

其中，$\operatorname{tr}(\mathbf{B})=\sum_{i=1}^{m}\left\|\pmb{z}_{i}\right\|^{2}$，因此有

$$
\operatorname{dist}_{i .}^{2}=\frac{1}{m} \sum_{j=1}^{m} d i s t_{i j}^{2}$$ $$d i s t_{\cdot j}^{2}=\frac{1}{m} \sum_{i=1}^{m} d i s t_{i j}^{2}$$ $$\text {dist.}=\frac{1}{m^{2}} \sum_{i=1}^{m} \sum_{j=1}^{m} d i s t_{i j}^{2}
$$

综上，可得

$$
b_{i j}=-\frac{1}{2}\left(d i s t_{i j}^{2}-d i s t_{i}^{2}-d i s t_{\cdot j}^{2}+d i s t_{. .}^{2}\right)
$$

由此即可通过降维前后保持不变的距离矩阵$\mathbf{D}$求取内积矩阵$\mathbf{B}$

对矩阵B做特征值分解（eigenvalue decomposition），$\mathbf{B}=\mathbf{V} \mathbf{\Lambda} \mathbf{V}^{\mathrm{T}}$，其中$\pmb{\Lambda}=\operatorname{diag}\left(\lambda_{1}, \lambda_{2}, \dots, \lambda_{d}\right)$为特征值构成的对角矩阵，$\lambda_{1} \geqslant \lambda_{2} \geqslant \ldots \geqslant \lambda_{d}$，$\mathbf{V}$为特征向量矩阵，假定其中有$d^\*$个非零特征值，它们构成对角矩阵$\pmb{\Lambda}\_\*=\operatorname{diag}\left(\lambda\_{1}, \lambda\_{2}, \dots, \lambda\_{d^\*}\right)$，令$\mathbf{V}\_\*$表示相应的特征向量矩阵，则

$$
\mathbf{Z}=\mathbf{\Lambda}_{*}^{1 / 2} \mathbf{V}_{*}^{\mathrm{T}} \in \mathbb{R}^{d^{*} \times m}
$$

现实应用中为了有效降维，往往仅需降维后的距离与原始空间中的距离尽可能接近，而不必严格相等，此时可取$d^{\prime} \ll d$个最大特征值构成对角矩阵，$\tilde{\mathbf{\Lambda}}=\operatorname{diag}\left(\lambda_{1}, \lambda_{2}, \ldots, \lambda_{d^{\prime}}\right)$，令$\tilde{\mathbf{V}}$表示相应的特征向量矩阵，则

$$
\mathbf{Z}=\tilde{\mathbf{\Lambda}}^{1 / 2} \tilde{\mathbf{V}}^{\mathrm{T}} \in \mathbb{R}^{d^{\prime} \times m}
$$

算法描述如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118184947787.png#pic_center)

## 线性变换
欲获得低维子空间，最简单的是对原始高维空间进行线性变换，给定d维空间中的样本$\mathbf{X}=\left(\pmb{x}\_{1}, \pmb{x}\_{2}, \ldots, \pmb{x}\_{m}\right) \in \mathbb{R}^{d \times m}$，变换后得到$d^{\prime} \leq d$维空间中的样本

$$
\mathbf{Z}=\mathbf{W}^{\mathrm{T}} \mathbf{X}
$$

其中$\mathbf{W} \in \mathbb{R}^{d \times d^{\prime}}$是变换矩阵，$\mathbf{Z} \in \mathbb{R}^{d^{\prime} \times m}$是样本在新空间中的表达

变换矩阵$\mathbf{W}$可视为$d'$个$d$维基向量，$\pmb{z}\_{i}=\mathbf{W}^{\mathrm{T}} \pmb{x}\_{i}$是第$i$个样本与这$d'$个基向量分别做内积得到的$d'$个属性向量，换言之，$\pmb{z}\_{i}$是原属性向量$\pmb{x}\_{i}$在新坐标系$\left\\{\pmb{w}\_{1}, \pmb{w}\_{2}, \cdots, \pmb{w}\_{d^{\prime}}\right\\}$中的坐标向量，若$\pmb{w}\_{i}$与$\pmb{w}\_{j}(i\neq j)$正交，则新坐标系是一个正交坐标系，此时$\pmb{W}$为正交变换，新空间中的属性是原空间中属性的线性组合

基于线性变换来进行降维的方法称为线性降维方法，不同之处是对低维子空间的性质有不同的要求，相当于对$\pmb{W}$施加了不同的约束

对降维效果的评估，通常是比较降维前后学习器的性能，若性能有所提高则认为降维起到了作用，若将位数将至二维或三维，则可通过可视化技术来直观地判断降维效果

# 3. 主成分分析

## 两种等价推导

如何用一个超平面（直线的高维推广）对所有样本进行恰当的表达？

- 最近重构性：样本点到这个超平面的距离都足够近
- 最大可分性：样本点在这个超平面上的投影都尽可能分开

基于重构性和最大可分性，能分别得到主成分分析（Principal Component Analysis, PCA）这一常用的降维方法的两种等价推导

**基于重构性的推导**：

假设数据样本进行了中心化，即$\sum_{i} \pmb x\_{i}=0$，再假定投影变换后得到的新坐标系为$\left\\{\pmb{w}\_{1}, \pmb{w}\_{2}, \ldots, \pmb{w}\_{d}\right\\}$，其中$\pmb{w}\_{i}$是标准正交基向量，$\left\|\pmb{w}\_{i}\right\|\_{2}=1, \pmb{w}\_{i}^{\mathrm{T}} \pmb{w}\_{j}=0(i\neq j)$，若丢弃新坐标系中的部分坐标，即将维度降低到$d^{\prime}<d$，则样本点$\pmb x\_i$在低维坐标系中的投影是$\pmb{z}\_{i}=\left(z\_{i 1} ; z\_{i 2} ; \ldots ; z\_{i d^{\prime}}\right)$，其中$z\_{i j}=\pmb{w}\_{j}^{\mathrm{T}} \pmb{x}\_{i}$是$\pmb{x}\_{i}$在低维坐标系下第$j$维的坐标，若基于$\pmb{z}\_{i}$来重构$\pmb{x}\_{i}$，则$\hat{\pmb{x}}\_{i}=\sum_{j=1}^{d^{\prime}} z\_{i j} \pmb{w}\_{j}$

若考虑整个训练集，原样本点$\pmb{x}\_{i}$与基于投影重构的样本点$\hat{\pmb{x}}\_{i}$之间的距离为

$$
\begin{aligned} \sum_{i=1}^{m}\left\|\sum_{j=1}^{d^{\prime}} z_{i j} \pmb{w}_{j}-\pmb{x}_{i}\right\|^{2} &=\sum_{i=1}^{m} \pmb{z}_{i}^{\mathrm{T}} \pmb{z}_{i}-2 \sum_{i=1}^{m} \pmb{z}_{i}^{\mathrm{T}} \mathbf{W}^{\mathrm{T}} \pmb{x}_{i}+\mathrm{const} \\ & \propto-\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}}\left(\sum_{i=1}^{m} \pmb{x}_{i} \pmb{x}_{i}^{\mathrm{T}}\right) \mathbf{W}\right) \end{aligned}
$$

其中，$\mathbf{W}=\\{\pmb w_1,\pmb w_j,\dots ,\pmb w_d\\}$，根据重构性，上式应被最小化，考虑到$\pmb w_j$是标准正交基，$\sum_i \pmb{x}\_{i} \pmb{x}\_{i}^{\mathrm{T}}$是协方差矩阵，有

$$
\begin{array}{cl}\underset{\mathbf{W}}{\min } & {-\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{X} \mathbf{X}^{\mathrm{T}} \mathbf{W}\right)} \\ {\text { s.t. }} & {\mathbf{W}^{\mathrm{T}} \mathbf{W}=\mathbf{I}}\end{array}
$$

**基于最大可分性的推导**：

样本点$\pmb x\_i$在新空间中超平面上的投影是$\mathbf{W}^{\mathrm{T}} \pmb{x}\_{i}$，若所有样本点的投影能尽可能分开，则应该使投影后样本点的方差最大化，投影后样本点的协方差矩阵是$\sum_{i} \mathbf{W}^{\mathrm{T}} \pmb{x}\_{i} \pmb{x}\_{i}^{\mathrm{T}} \mathbf{W}$，于是优化目标可写为

$$
\begin{array}{cl}\underset{\mathbf{W}}{\max } & {\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{X} \mathbf{X}^{\mathrm{T}} \mathbf{W}\right)} \\ {\text { s.t. }} & {\mathbf{W}^{\mathrm{T}} \mathbf{W}=\mathbf{I}}\end{array}
$$

对优化目标使用拉格朗日乘子法，可得

$$
\mathbf{X} \mathbf{X}^{\mathrm{T}} \mathbf{w}_i=\lambda \mathbf{w}_i
$$

只需对协方差矩阵$\mathbf{X} \mathbf{X}^{\mathrm{T}}$进行特征值分解，将求得的特征值排序：$\lambda\_{1} \geqslant \lambda\_{2} \geqslant \ldots \geqslant \lambda\_{d}$，再取前$d'$个特征值对应的特征向量构成$\mathbf{W}=\left(\pmb{w}\_{1}\right.,\left.\pmb{w}\_{2}, \ldots, \pmb{w}\_{d^{\prime}}\right)$，即为主成分分析的解

算法如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118192442821.png#pic_center)

## 其他说明

降维后的维数$d'$：

- 通常由用户事先指定
- 通过在$d'$值不同的低维空间中对k近邻分类器（或其他开销较小的学习器）进行交叉验证来选取
- 对PCA还可从重构的角度设置一个重构阈值，例如$t=95\%$，然后选取使下式成立的最小$d'$值$$\frac{\sum_{i=1}^{d^{\prime}} \lambda_{i}}{\sum_{i=1}^{d} \lambda_{i}} \geqslant t$$

PCA仅需保留$\mathbf{W}^*$与样本的均值向量即可通过简单的向量减法和矩阵-向量乘法将新样本投影至低维空间中，降维导致最小的$d'-d$个特征值的特征向量被舍弃了，但舍弃这部分信息往往是必要的：

- 舍弃这部分信息之后能使样本的采样密度增大，这正是降维的重要动机
- 当数据受到噪声影响时，最小的特征值所对应的特征向量往往与噪声相关，将他们舍弃能在一定程度上起到去噪的效果

# 4. 核化线性降维
在不少现实任务中，可能需要非线性映射才能找到合适的低维嵌入，例如下图，样本点从二维空间中的矩形区域采样后以S形曲面嵌入到三维空间，若直接使用线性降维方法，则将丢失原本的低维结构。为了区分原本采样的低维空间与降维后的低维空间加以区分，称前者为本真（intrinsic）低维空间

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119172716111.png#pic_center)

非线性降维的一种常用方法，是基于核技巧对线性降维方法进行核化（kernelized）

**核主成分分析**（Kernelized PCA, KPCA）

假定我们将在高维特征空间中把数据投影到由$\mathbf{W}=(\pmb w_1,\pmb w_2,\dots ,\pmb w_d)$确定的超平面上，则对于$\pmb w_j$，有

$$
\left(\sum_{i=1}^{m} z_{i} z_{i}^{\mathrm{T}}\right) \pmb w_j=\lambda \pmb w_j
$$

其中，$\pmb z_i$是样本点$\pmb x_i$在高维特征空间中的像，进一步有

$$
\begin{aligned} \pmb w_j &=\frac{1}{\lambda}\left(\sum_{i=1}^{m} z_{i} \pmb{z}_{i}^{\mathrm{T}}\right) \pmb w_j=\sum_{i=1}^{m} \pmb{z}_{i} \frac{\pmb{z}_{i}^{\mathrm{T}} \pmb w_j}{\lambda_j} \\ &=\sum_{i=1}^{m} \pmb{z}_{i} \alpha_{i}^j \end{aligned}
$$

其中$\alpha\_{i}^j=\frac{\pmb{z}\_{i}^{\mathrm{T}} \pmb w\_j}{\lambda\_j}$是$\pmb \alpha\_i$的第$j$个分量，假定$\pmb z\_i$是样本点$\pmb x\_i$通过映射$\phi$产生的，即$\pmb{z}\_{i}=\phi\left(\pmb{x}\_{i}\right), i=1,2, \ldots, m$，若$\phi$能被显式表达出来，则通过它将样本映射到高维特征空间，再在特征空间中实施PCA即可


$$
\left(\sum_{i=1}^{m} \phi\left(\pmb{x}_{i}\right) \phi\left(\pmb{x}_{i}\right)^{\mathrm{T}}\right) \pmb w_j=\lambda \pmb w_j$$ $$\pmb w_j=\sum_{i=1}^{m} \phi\left(\pmb{x}_{i}\right) {\alpha}_{i}^j
$$


一般情形下，我们不清楚$\phi$的具体形式，于是引入核函数

$$
\kappa\left(\pmb{x}_{i}, \pmb{x}_{j}\right)=\phi\left(\pmb{x}_{i}\right)^{\mathrm{T}} \phi\left(\pmb{x}_{j}\right)
$$

将上两式代入$\left(\sum_{i=1}^{m} \phi\left(\pmb{x}\_{i}\right) \phi\left(\pmb{x}\_{i}\right)^{\mathrm{T}}\right) \pmb w\_j=\lambda \pmb w\_j$，化简可得

$$
\mathbf{K}\pmb \alpha_j=\lambda_j\pmb \alpha_j
$$

其中$\mathbf{K}$为$\kappa$对应的核矩阵，$\mathbf{K}_{ij}=\kappa (\pmb x_i,\pmb x_j),\pmb \alpha_j=(\alpha_1^j;\alpha_2^j;\dots ;\alpha_m^j)$（分号代表是列向量），显然，上式是特征值分解问题，取$\mathbf{K}$最大的$d'$个特征值对应的特征向量即可

对新样本$\pmb x$，其投影后的第$j(j=1,2,\dots ,d')$维坐标为

$$
\begin{aligned} z_{j} &=\pmb{w}_{j}^{\mathrm{T}} \phi(\pmb{x})=\sum_{i=1}^{m} \alpha_{i}^{j} \phi\left(\pmb{x}_{i}\right)^{\mathrm{T}} \phi(\pmb{x}) \\ &=\sum_{i=1}^{m} \alpha_{i}^{j} \kappa\left(\pmb{x}_{i}, \pmb{x}\right) \end{aligned}
$$

其中，$\pmb \alpha_i$已经过规范化，上式表明，为获得投影后的坐标，KPCA需对所有样本求和，因此计算开销较大

# 5. 流形学习

流形学习是一类借鉴了拓扑流形概念的降维方法，流形是在局部与欧式空间同胚的空间，即在局部具有欧式空间的性质，能用欧氏距离来进行距离计算，这给降维方法带来了很大的启发：若低维流形嵌入到高维空间，则数据样本在高维空间中看起来非常复杂，但在局部上仍具有欧式空间的性质，因此可以容易地在局部建立降维映射关系，然后再设法将局部映射关系推广到全局，当维数将至二维或三维时，能对数据进行可视化展示

## 等度量映射

等度量映射（Isometric Mapping, Isomap）认为低维流形嵌入到高维空间之后，直接在高维空间中计算直线距离有误导性，因为高维空间中的直线距离在低维嵌入流形上不可达的，低维嵌入流形上两点间的距离是测地线（geodesic）距离（红色线），测地线距离是两点间的本真距离

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191121204224759.png#pic_center)

计算测地线距离时，我们可利用流形在局部上与欧式空间同胚的性质，对每个点基于欧氏距离找出其近邻点，建立近邻连接图，图上近邻点之间有连接，非近邻点没有连接，将问题转化为计算近邻连接图上两点之间的最短路径问题，可采用著名的Dijkstra算法或Floyd算法，得到两点间的距离后，就可通过MDS方法（10.2节）来获得样本点在低维空间中的坐标。Isomap算法如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122140722898.png#pic_center)

Isomap仅得到了训练样本在低维空间的坐标，对于新样本如何将其映射到低维空间呢？常用方法是将训练样本的高维空间坐标作为输入，低维空间坐标作为输出，训练一个回归学习器来对新样本的低维空间坐标进行预测

对近邻图的构建通常有两种做法，一种是指定近邻点个数，例如欧氏距离最近的k个点为近邻点，称为k近邻图，另一种是指定距离阈值$\epsilon$，距离小于$\epsilon$的点为近邻点，称为$\epsilon$近邻图。两种方法均有不足，若近邻范围指定得较大，则距离较远的点可能被误认为近邻，会出现短路问题，近邻范围指定得较小，则图中有些区域可能与其他区域不存在连接，会出现断路问题，二者都会给后续的最短路径计算造成误导

## 局部线性嵌入

与Isomap试图保持近邻样本之间的距离不同，局部线性嵌入（Locally Linear Embedding）试图保持邻域内样本之间的线性关系，假定样本点$\pmb x_i$能通过邻域样本$\pmb x_j,\pmb x_k,\pmb x_l$的坐标通过线性组合重构出来，即

$$
\pmb{x}_{i}=w_{i j} \pmb{x}_{j}+w_{i k} \pmb{x}_{k}+w_{i l} \pmb{x}_{l}
$$

LLE希望上述关系能在低维空间中得以保持

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122142323902.png#pic_center)

LLE先为每个样本$\pmb x_i$找到其近邻下标集合$Q_i$，然后计算出线性重构的系数$\pmb w_i$ 

$$
\begin{aligned} \underset{\pmb{w}_{1}, \pmb{w}_{2}, \ldots, \pmb{w}_{m}}{\min} & \sum_{i=1}^{m}\left\|\pmb{x}_{i}-\sum_{j \in Q_{i}} w_{i j} \pmb{x}_{j}\right\|_{2}^{2} \\ \text { s.t. } & \sum_{j \in Q_{i}} w_{i j}=1\end{aligned}
$$

其中$\pmb x\_i,\pmb x\_j$均为已知，令$C\_{j k}=\left(\pmb{x}\_{i}-\pmb{x}\_{j}\right)^{\mathrm{T}}\left(\pmb{x}\_{i}-\pmb{x}\_{k}\right)$，$w\_{ij}$有闭式解

$$
w_{i j}=\frac{\sum_{k \in Q_{i}} C_{j k}^{-1}}{\sum_{l, s \in Q_{i}} C_{l s}^{-1}}
$$

LLE在低维空间中保持$\pmb w_i$不变，于是$\pmb x_i$在对应的低维空间坐标$\pmb z_i$可通过下式求解

$$
\min _{\pmb{z}_{1}, \pmb{z}_{2}, \ldots, \pmb{z}_{m}} \sum_{i=1}^{m}\left\|\pmb{z}_{i}-\sum_{j \in Q_{i}} w_{i j} \pmb{z}_{j}\right\|_{2}^{2}
$$

令$\mathbf{Z}=\left(\pmb{z}\_{1}, \pmb{z}\_{2}, \ldots, \pmb{z}\_{m}\right) \in \mathbb{R}^{d^{\prime} \times m},(\mathbf{W})\_{i j}=w\_{i j},\mathbf{M}=(\mathbf{I}-\mathbf{W})^{\mathrm{T}}(\mathbf{I}-\mathbf{W})$，则上式可重写为

$$
\begin{array}{l}\underset{\mathbf{z}}{\min} & \operatorname{tr}\left(\mathbf{Z} \mathbf{M} \mathbf{Z}^{\mathrm{T}}\right) \\ \text { s.t. } & \mathbf{Z} \mathbf{Z}^{\mathrm{T}}=\mathbf{I}\end{array}
$$

理解：$\mathbf{Z}(1-\mathbf{W})^\mathrm{T}$是$d\times m$维的，其第一行乘以其转秩的第一列是上上式里面的每个$\pmb{z}\_{i}-\sum_{j \in Q\_{i}} w\_{i j} \pmb{z}\_{j}$的第一个分量求平方和

上式可通过特征值分解求解：$\mathbf{M}$最小的$d'$个特征值对应的特征向量组成的矩阵即为$\mathbf{Z}^\mathrm{T}$，算法如下所示，对于不在样本$\pmb x_i$邻域区域的样本$\pmb x_j$，无论其如何变化都对$\pmb x_i$和$\pmb z_i$没有任何影响，这种将变动限制在局部的思想在很多地方都有用

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122144656811.png#pic_center)

# 6. 度量学习

对高维数据进行降维的主要目的是希望找到一个合适的低维空间，在此空间中进行学习能比原始空间性能更好，事实上每个空间对应了样本属性上定义的一个距离度量，而寻找合适的空间实质上就是在寻找一个合适的距离度量，度量学习（metric learning）的基本动机即为直接学习出一个合适的距离度量

## 从加权欧氏距离引入度量学习

对两个d维样本$\pmb x_i,\pmb x_j$，假定不同属性的重要性不同，则可引入属性权重，其平方加权欧氏距离为

$$
\begin{aligned} \operatorname{dist}_{\mathrm{wed}}^{2}\left(\pmb{x}_{i}, \pmb{x}_{j}\right) &=\left\|\pmb{x}_{i}-\pmb{x}_{j}\right\|_{2}^{2}=w_{1} \cdot d i s t_{i j, 1}^{2}+w_{2} \cdot d i s t_{i j, 2}^{2}+\ldots+w_{d} \cdot d i s t_{i j, d}^{2} \\ &=\left(\pmb{x}_{i}-\pmb{x}_{j}\right)^{\mathrm{T}} \mathbf{W}\left(\pmb{x}_{i}-\pmb{x}_{j}\right) \end{aligned}
$$

其中，$w_{i} \geqslant 0, \mathbf{W}=\operatorname{diag}(\pmb{w})$是一个对角阵，可通过学习确定

进一步，$\mathbf{W}$的非对角元素均为零，这意味着坐标轴是正交的，即属性之间无关，但现实任务中往往不是这样，如西瓜的重量和体积正相关，为此，将$\mathbf{W}$替换为一个普通的半正定矩阵$\mathbf{M}$（保持距离非负且对称，即必有正交基$\mathbf{P}$使得$\mathbf{M}=\mathbf{P} \mathbf{P}^{\mathrm{T}}$），可得到马氏距离（Mahalanobis distance）：

$$
\operatorname{dist}_{\operatorname{mah}}^{2}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)=\left(\pmb{x}_{i}-\pmb{x}_{j}\right)^{\mathrm{T}} \mathbf{M}\left(\pmb{x}_{i}-\pmb{x}_{j}\right)=\left\|\pmb{x}_{i}-\pmb{x}_{j}\right\|_{\mathrm{M}}^{2}
$$

其中，$\mathbf{M}$称为度量矩阵，度量学习则是对$\mathbf{M}$进行学习

## 近邻成分分析

对$\mathbf{M}$学习要设置一个目标，假定我们希望提高近邻分类器的性能，则可将$\mathbf{M}$嵌入到其评价指标中去，通过优化该性能指标求得，下面以近邻成分分析为例进行讨论

近邻成分分析在进行判别时通常采用多数投票法，不妨将其替换为概率投票法，对任意样本$\pmb x_j$，它对$\pmb x_i$分类结果影响的概率为

$$
p_{i j}=\frac{\exp \left(-\left\|\pmb{x}_{i}-\pmb{x}_{j}\right\|_{\mathrm{M}}^{2}\right)}{\sum_{l} \exp \left(-\left\|\pmb{x}_{i}-\pmb{x}_{l}\right\|_{\mathrm{M}}^{2}\right)}
$$

显然，距离越大影响越小，自身的影响最大，若以留一法（LOO）正确率的最大化为目标，则可计算$\pmb x_i$的留一法正确率，即它被自身之外的所有样本正确分类的概率为

$$
p_{i}=\sum_{j \in \Omega_{i}} p_{i j}
$$

其中$\Omega_i$表示与$\pmb x_i$属于相同类别的样本的下标集合，于是整个样本集上的留一法正确率为

$$
\sum_{i=1}^{m} p_{i}=\sum_{i=1}^{m} \sum_{j \in \Omega_{i}} p_{i j}
$$

即NCA的优化目标为

$$
\min _{\mathbf{P}} 1-\sum_{i=1}^{m} \sum_{j \in \Omega_{i}} \frac{\exp \left(-\left\|\mathbf{P}^{\mathrm{T}} \pmb{x}_{i}-\mathbf{P}^{\mathrm{T}} \pmb{x}_{j}\right\|_{2}^{2}\right)}{\sum_{l} \exp \left(-\left\|\mathbf{P}^{\mathrm{T}} \pmb{x}_{i}-\mathbf{P}^{\mathrm{T}} \pmb{x}_{l}\right\|_{2}^{2}\right)}
$$

从而可得到最大化近邻分类器LOO正确率的距离度量矩阵$\mathbf{M}$

## 引入领域知识

若已知某些样本相似，某些样本不相似，则可定义必连（must-link）约束集合$\mathcal{M}$和勿连（cannot-link）约束集合$\mathcal{C}$，$\left(\pmb{x}\_{i}, \pmb{x}\_{j}\right) \in \mathcal{M}$表示二者相似，$\left(\pmb{x}\_{i}, \pmb{x}\_{j}\right) \in \mathcal{C}$表示二者不相似，显然我们希望相似的样本之间距离小，不相似的样本之间距离大，因此优化目标为

$$
\begin{array}{cl}\underset{\mathbf{M}}{\min} & {\sum_{\left(\pmb{x}_{i}, \pmb{x}_{j}\right) \in \mathcal{M}}\left\|\pmb{x}_{i}-\pmb{x}_{j}\right\|_{\mathrm{M}}^{2}} \\ {\text { s.t. }} & {\sum_{\left(\pmb{x}_{i}, \pmb{x}_{k}\right) \in \mathcal{C}}\left\|\pmb{x}_{i}-\pmb{x}_{k}\right\|_{\mathrm{M}}^{2} \geqslant 1} \\ & {\mathbf{M} \succeq 0}\end{array}
$$

其中，${\mathbf{M} \succeq 0}$表明$\mathbf{M}$必须是半正定的，上式要求在不相似样本间的距离不小于1的前提下相似样本间的距离尽可能小

不同的度量学习方法针对不同目标获得好的半正定对称距离度量矩阵$\mathbf{M}$，若$\mathbf{M}$是一个低秩矩阵，则通过对$\mathbf{M}$进行特征值分解，总能找到一组正交基，其正交基数目为矩阵$\mathbf{M}$的秩$rank(\mathbf{M})$，小于原属性$d$，于是度量学习学得的结果可衍生出一个降维矩阵$\mathbf{P} \in \mathbb{R}^{d \times \operatorname{rank}(\mathbf{M})}$，能用于降维目的
