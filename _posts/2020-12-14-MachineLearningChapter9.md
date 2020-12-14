---
title: 西瓜书 | 第九章 聚类
author: 钟欣然
date: 2020-12-13 03:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

# 1. 聚类任务

**无监督学习**：

在无监督学习（unsupervised learning）中，训练样本的标记信息是未知的，目标是通过对无标记训练样本的学习来揭示数据的内在性质及规律，为进一步的数据分析提供基础，其中研究最多、应用最广的即为聚类（clustering）

**聚类**：

聚类试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个簇（cluster），每个簇可能对应一些潜在的概念/类别，如“浅色瓜”，但这些概念聚类前是未知的，聚类后由使用者来把握和命名

**符号表示**：

- 样本集$D=\left\\{\pmb{x}\_{1}, \pmb{x}\_{2}, \ldots, \pmb{x}\_{m}\right\\}$包含m个无标记样本，每个样本$\pmb{x}\_{i}=\left(x\_{i 1} ; x\_{i 2} ; \ldots ; x\_{i n}\right)$是一个n维特征向量
- 聚类算法将样本集划分为k个不相交的簇$\left\\{C\_{l} \vert l=1,2 ; \ldots, k\right\\}$，满足$C\_{l}^{\prime} \cap_{l^{\prime} \neq l} C_{l}=\varnothing,D=\bigcup_{l=1}^{k} C_{l}$
- 样本$\pmb x_j$的簇标记（cluster label）记为$\lambda\_{j} \in\{1,2, \ldots, k\}$，即$\pmb{x}\_{j} \in C_{\lambda\_{j}}$，聚类的结果可用包含m个元素的簇标记向量$\pmb{\lambda}=\left(\lambda\_{1} ; \lambda\_{2} ; \ldots ; \lambda\_{m}\right)$表示

**意义**：

聚类既可以作为一个单独过程，用于寻找数据内在的分布结构，也可用作分类等其他学习任务的先驱，如先聚类再将聚类结果每个簇定义为一个类，再基于这些类训练分类模型

# 2. 性能度量

聚类性能度量也叫作聚类有效性指标（validity index），必要性体现在两个方面：

- 我们需要通过某种性能度量来评价聚类的好坏
- 若明确了最终将要使用的性能度量，可直接将其作为聚类过程的优化目标，从而得到符合要求的聚类结果

性能度量的标准是同一簇的样本尽可能彼此相似，不同簇的样本尽可能不同，即簇内相似度（intra-cluster similarity）高且簇间相似度（inter-cluster similarity）低，方法大致有外部指标和内部指标两类

## 外部指标

外部指标（external index）将聚类结果与某个参考模型（reference model）进行比较，参考模型例如领域专家给出的划分结果

对数据集$D=\left\\{\pmb{x}\_{1}, \pmb{x}\_{2}, \ldots, \pmb{x}\_{m}\right\\}$，假定聚类给出的簇划分为$\mathcal{C}=\left\\{C\_{1}\right.\left.C\_{2}, \ldots, C\_{k}\right\\}$，参考模型给出的簇划分为$\mathcal{C}^{\*}=\left\\{C\_{1}^{\*}, C\_{2}^{\*}, \ldots, C\_{s}^{\*}\right\\}$，相应地令$\pmb \lambda,\pmb \lambda^\*$分别表示二者对应的簇标记向量，考虑样本两两配对，定义

$$
\begin{array}{l}{\left.a=|S S|, \quad S S=\left\{\left(\pmb{x}_{i}, \pmb{x}_{j}\right) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right)\right\}} \\ {\left.b=|S D|, \quad S D=\left\{\left(\pmb{x}_{i}, \pmb{x}_{j}\right) | \lambda_{i}=\lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\right\}\right\}} \\ {\left.c=|D S|, \quad D S=\left\{\left(\pmb{x}_{i}, \pmb{x}_{j}\right) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*}=\lambda_{j}^{*}, i<j\right)\right\}} \\ {\left.d=|D D|, \quad D D=\left\{\left(\pmb{x}_{i}, \pmb{x}_{j}\right) | \lambda_{i} \neq \lambda_{j}, \lambda_{i}^{*} \neq \lambda_{j}^{*}, i<j\right)\right\}}\end{array}
$$

集合SS包含了在$\mathcal{C}$中隶属于相同簇且在$\mathcal{C}^\*$中也隶属于相同簇的样本对，集合SD包含了在$\mathcal{C}$中隶属于相同簇但在$\mathcal{C}^\*$中也隶属于不同簇的样本对……且有$a+b+c+d=\frac{m(m-1)}2$

常用的外部指标有：

- Jaccard系数（Jaccard Coefficient, JC）：值在$[0,1]$区间，越大越好

$$
\mathrm{JC}=\frac{a}{a+b+c}
$$

- FM指数（Fowlkes and Mallows Index, FMI）：值在$[0,1]$区间，越大越好

$$
\mathrm{FMI}=\sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}
$$

- Rand指数（Rand Index, RI）：值在$[0,1]$区间，越大越好

$$
\mathrm{RI}=\frac{2(a+d)}{m(m-1)}
$$

## 内部指标

内部指标（internal index）直接考察聚类结果而不利用任何参考模型

考虑聚类结果的簇划分$\mathcal{C}=\left\\{C\_{1}\right.\left.C\_{2}, \ldots, C\_{k}\right\\}$，令$dist(\cdot ,\cdot )$表示两个样本之间的距离，$\pmb \mu=\frac{1}{\vert C\vert} \sum_{1 \leqslant i \leqslant\vert C\vert} \pmb{x}\_{i}$表示簇C的中心点，则有

- 簇C内样本间的平均距离

$$
\operatorname{avg}(C)=\frac{2}{|C|(|C|-1)} \sum_{1 \leqslant i<j \leqslant|C|} \operatorname{dist}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)
$$

- 簇C内样本间的最远距离

$$
\operatorname{diam}(C)=\max _{1 \leqslant i<j \leqslant|C|} \operatorname{dist}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)
$$

- 簇$C_i$和簇$C_j$最近样本间的距离

$$
d_{\min }\left(C_{i}, C_{j}\right)=\min _{\pmb{x}_{i} \in C_{i}, \pmb{x}_{j} \in C_{j}} \operatorname{dist}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)
$$

- 簇$C_i$和簇$C_j$中心点间的距离

$$
d_{\mathrm{cen}}\left(C_{i}, C_{j}\right)=\operatorname{dist}\left(\pmb{\mu}_{i}, \pmb{\mu}_{j}\right)
$$

常用的内部指标：

- DB指数（Davies-Bouldin Index, DBI）：值越小越好

$$
\mathrm{DBI}=\frac{1}{k} \sum_{i=1}^{k} \max _{j \neq i}\left(\frac{\operatorname{avg}\left(C_{i}\right)+\operatorname{avg}\left(C_{j}\right)}{d_{\operatorname{cen}}\left(\pmb{\mu}_{i}, \pmb{\mu}_{j}\right)}\right)
$$

- Dunn指数（Dunn Index, DI）：值越大越好

$$
\mathrm{DI}=\min _{1 \leqslant i \leqslant k}\left\{\min _{j \neq i}\left(\frac{d_{\min }\left(C_{i}, C_{j}\right)}{\max _{1 \leqslant l \leqslant k} \operatorname{diam}\left(C_{l}\right)}\right)\right\}
$$

# 3. 距离计算

**距离度量$dist(\cdot,\cdot)$的基本性质**：

- 非负性：$\operatorname{dist}\left(\pmb{x}\_{i}, \pmb{x}\_{j}\right) \geqslant 0$
- 同一性：$\operatorname{dist}\left(\pmb{x}\_{i}, \pmb{x}\_{j}\right)=0$当且仅当$\pmb{x}\_{i}=\pmb{x}\_{j}$
- 对称性：$\operatorname{dist}\left(\pmb{x}\_{i}, \pmb{x}\_{j}\right)=\operatorname{dist}\left(\pmb{x}\_{j}, \pmb{x}\_{i}\right)$
- 直递性：$\operatorname{dist}\left(\pmb{x}\_{i}, \pmb{x}\_{j}\right) \leqslant \operatorname{dist}\left(\pmb{x}\_{i}, \pmb{x}\_{k}\right)+\operatorname{dist}\left(\pmb{x}\_{k}, \pmb{x}\_{j}\right)$

**属性分类**：
- 有序属性（ordinal attribute）：包含连续属性（continuous attribute）和如{1,2,3}这样的离散属性（categorical attribute）
- 无序属性（non-ordinal attribute）：如{飞机，火车，轮船}这样的离散属性

**距离度量**：

- 闵可夫斯基距离（Minkowski distance）用于有序属性

$$
\operatorname{dist}_{\mathrm{mk}}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)=\left(\sum_{u=1}^{n}\left|x_{i u}-x_{j u}\right|^{p}\right)^{\frac{1}{p}}
$$

p=2时即为欧氏距离（Euclidean distance）

$$
\operatorname{dist}_{\mathrm{ed}}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)=\left\|\pmb{x}_{i}-\pmb{x}_{j}\right\|_{2}=\sqrt{\sum_{u=1}^{n}\left|x_{i u}-x_{j u}\right|^{2}}
$$

p=1时即为曼哈顿距离（Manhattan distance）

$$
\operatorname{dist}_{\operatorname{man}}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)=\left\|\pmb{x}_{i}-\pmb{x}_{j}\right\|_{1}=\sum_{u=1}^{n}\left|x_{i u}-x_{j u}\right|
$$

- VDM（Value Difference Metric）用于无序离散属性，令$m_{u,a}$表示在属性u上取值为a的样本数，$m_{u,a,i}$表示在第i个样本簇中在属性u上取值为a的样本数，k为样本簇数，则属性u上两个离散值a与b的VDM距离为

$$
\operatorname{VDM}_{p}(a, b)=\sum_{i=1}^{k}\left|\frac{m_{u, a, i}}{m_{u, a}}-\frac{m_{u, b, i}}{m_{u, b}}\right|^{p}
$$

- 混合距离：假定有$n_c$个有序属性，$n-n_c$个无序属性，且有序属性排在前面，则

$$
\operatorname{MinkovDM}_{p}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)=\left(\sum_{u=1}^{n_{c}}\left|x_{i u}-x_{j u}\right|^{p}+\sum_{u=n_{c}+1}^{n} \operatorname{VDM}_{p}\left(x_{i u}, x_{j u}\right)\right)^{\frac{1}{p}}
$$

- 加权距离（weighted distance）：当样本空间中不同属性的重要性不同时，可使用及安全距离，以加权闵可夫斯基距离为例

$$
\operatorname{dist}_{\mathrm{wmk}}\left(\pmb{x}_{i}, \pmb{x}_{j}\right)=\left(w_{1} \cdot\left|x_{i 1}-x_{j 1}\right|^{p}+\ldots+w_{n} \cdot\left|x_{i n}-x_{j n}\right|^{p}\right)^{\frac{1}{p}}
$$

其中，$w_i$为权重，满足$w_i\geq 0,\sum_{i=1}^{n} w_{i}=1$

**其他说明**：

- 通常我们是基于某种形式的距离来定义相似度度量，距离越大，相似度越小，按时用于相似度度量的距离未必一定要满足距离度量的所有基本性质，尤其是直递性，这样的距离成为非度量距离

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116175518875.png#pic_center)

- 在不少现实任务中有必要基于数据样本来确定合适的距离计算式，可通过距离度量学习（distance metric learning）实现

# 4. 原型聚类

原型聚类也称为基于原型的聚类（prototype-based clustering），此类算法假设聚类结构能通过一组原型刻画，很常用。通常情形下先对原型进行初始化，然后对原型进行迭代更新求解。采用不同的原型表示、不同的求解方式，将产生不同的算法

## k均值算法

算法最小化平方误差

$$
E=\sum_{i=1}^{k} \sum_{\pmb{x} \in C_{i}}\left\|\pmb{x}-\pmb{\mu}_{i}\right\|_{2}^{2}
$$

其中，$\pmb{\mu}\_{i}=\frac{1}{\vert C\_{i}\vert } \sum\_{\pmb{x} \in C\_{i}} \pmb{x}$是簇$C\_i$的均值向量，直观来看，上式在一定程度上刻画了簇内样本围绕簇均值向量的紧密程度，E值越小簇内样本相似度越高

最小化上式并不统一，找到它的最优解需考察样本集D中所有可能的簇划分，这是一个NP难问题，k均值算法采用了贪心策略，通过迭代优化近似求解上式，算法如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116182442681.png#pic_center)

西瓜数据集上的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116182556757.png#pic_center)

## 学习向量量化
学习向量量化（Learning Vector Quantization, LVQ）试图找到一组原型向量来刻画聚类结构，但与一般聚类方法不同，LVQ要求数据样本带有类别标记，学习过程利用样本的这些监督信息来辅助聚类

给定样本集$D=\left\\{\left(\pmb{x}\_{1}, y\_{1}\right),\left(\pmb{x}\_{2}, y\_{2}\right), \ldots,\left(\pmb{x}\_{m}, y\_{m}\right)\right\\}$，$y\_i\in \mathcal{Y}$是样本$\pmb x\_j$的类标记，LVQ的目标是学得一组n维向量$\left\\{\pmb{p}\_{1}, \pmb{p}\_{2}, \dots, \pmb{p}\_{q}\right\\}$，每个原型向量代表一个聚类簇，簇标记$t\_i\in \mathcal{Y}$，算法如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116184130858.png#pic_center)

停止条件可能是达到最大迭代轮数或原型向量更新很小甚至不再更新，算法的关键是如何更新原型向量直观上看，对样本$\pmb x\_j$，若最近的原型向量$\pmb p\_{i^\*}$与$\pmb x\_j$类别相同，则令$\pmb p\_{i^\*}$向$\pmb x\_j$的方向靠拢

$$
\pmb{p}^{\prime}=\pmb{p}_{i^{*}}+\eta \cdot\left(\pmb{x}_{j}-\pmb{p}_{i^{*}}\right)
$$

此时$\pmb{p}^{\prime}$与$\pmb x\_j$的距离为

$$
\begin{aligned}\left\|\pmb{p}^{\prime}-\pmb{x}_{j}\right\|_{2} &=\left\|\pmb{p}_{i^{*}}+\eta \cdot\left(\pmb{x}_{j}-\pmb{p}_{i^{*}}\right)-\pmb{x}_{j}\right\|_{2} \\ &=(1-\eta) \cdot\left\|\pmb{p}_{i^{*}}-\pmb{x}_{j}\right\|_{2} \end{aligned}
$$

若最近的原型向量$\pmb p\_{i\*}$与$\pmb x\_j$类别不同，则令$\pmb p\_{i^\*}$远离$\pmb x\_j$的方向

$$
\pmb{p}^{\prime}=\pmb{p}_{i^{*}}-\eta \cdot\left(\pmb{x}_{j}-\pmb{p}_{i^{*}}\right)
$$

此时$\pmb{p}^{\prime}$与$\pmb x\_j$的距离为

$$
\begin{aligned}\left\|\pmb{p}^{\prime}-\pmb{x}_{j}\right\|_{2} &=\left\|\pmb{p}_{i^{*}}-\eta \cdot\left(\pmb{x}_{j}-\pmb{p}_{i^{*}}\right)-\pmb{x}_{j}\right\|_{2} \\ &=(1+\eta) \cdot\left\|\pmb{p}_{i^{*}}-\pmb{x}_{j}\right\|_{2} \end{aligned}
$$

学得一组原型向量后，即可实现对样本空间的簇划分，每个样本$\pmb x$都被划入与其距离最近的原型向量所代表的的簇中；换言之，每个原型向量$\pmb p\_i$定义了与之相关的一个区域$R\_i$，该区域中每个样本在所有原型向量中与$\pmb p\_i$的距离最小

$$
R_{i}=\left\{\pmb{x} \in \mathcal{X} |\left\|\pmb{x}-\pmb{p}_{i}\right\|_{2} \leqslant\left\|\pmb{x}-\pmb{p}_{i}^{\prime}\right\|_{2}, i^{\prime} \neq i\right\}
$$

由此形成了对样本空间的簇划分$\left\\{R\_{1}, R\_{2}, \ldots, R\_{q}\right\\}$，该划分通常称为Voronoi划分（Voronoi tessellation）

西瓜数据集上的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116190318893.png#pic_center)

## 高斯混合聚类

高斯混合（Mixture-of-Gaussian）聚类采用概率模型来表达聚类原型

**背景知识**：

多元高斯分布密度函数如下，为明确显示分布与参数的依赖关系，记为$p(\pmb{x} \vert \pmb{\mu}, \mathbf{\Sigma})$

$$
p(\pmb{x})=\frac{1}{(2 \pi)^{\frac{n}{2}}|\mathbf{\Sigma}|^{\frac{1}{2}}} e^{-\frac{1}{2}(\pmb{x}-\pmb{\mu})^{\mathrm{T}} \mathbf{\Sigma}^{-1}(\pmb{x}-\pmb{\mu})}
$$

高斯混合分布由k个混合成分组成，$\alpha_i>0$为相应的混合系数（mixture coefficient），满足$\sum_{i=1}^{k} \alpha_{i}=1$ 

$$
p_{\mathcal{M}}(\pmb{x})=\sum_{i=1}^{k} \alpha_{i} \cdot p\left(\pmb{x} | \pmb{\mu}_{i}, \mathbf{\Sigma}_{i}\right)
$$

**后验概率**：

假设样本的生成过程由高斯混合分布给出，首先根据$\alpha_{1}, \alpha_{2}, \dots, \alpha_{k}$定义的先验分布选择高斯混合成分，其中$\alpha_i$为选择第$i$个混合成分的概率，然后根据被选择的混合成分的概率密度函数进行采样，从而生成相应的样本

若训练集$D=\left\\{\pmb{x}\_{1}, \pmb{x}\_{2}, \ldots, \pmb{x}\_{m}\right\\}$由上述过程生成，令随机变量$z\_j\in \{1,2,\dots ,k\}$表示生成样本$\pmb x\_j$的高斯混合成分，取值未知，其先验概率为$\alpha_i$，后验概率为

$$
\begin{aligned} p_{\mathcal{M}}\left(z_{j}=i | \pmb{x}_{j}\right) &=\frac{P\left(z_{j}=i\right) \cdot p_{\mathcal{M}}\left(\pmb{x}_{j} | z_{j}=i\right)}{p_{\mathcal{M}}\left(\pmb{x}_{j}\right)} \\ &=\frac{\alpha_{i} \cdot p\left(\pmb{x}_{j} | \pmb{\mu}_{i}, \pmb{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\pmb{x}_{j} | \pmb{\mu}_{l}, \mathbf{\Sigma}_{l}\right)} \end{aligned}
$$

将其记为$\gamma_{ji}(i=1,2,\dots ,k)$

高斯混合分布已知时，高斯混合聚类将把样本集划分为k个簇，每个样本$\pmb x_j$的簇标记$\lambda_j$为

$$
\lambda_{j}=\underset{i \in\{1,2, \ldots, k\}}{\arg \max } \gamma_{j i}
$$

从原型聚类的角度来看，高斯混合聚类是采用概率模型（高斯分布）对原型进行刻画，簇划分则由原型对应后验概率确定

**模型参数求解**：

给定样本集D，采用极大似然估计，利用EM算法进行迭代优化求解

$$
\begin{aligned} L L(D) &=\ln \left(\prod_{j=1}^{m} p_{\mathcal{M}}\left(\pmb{x}_{j}\right)\right) \\ &=\sum_{j=1}^{m} \ln \left(\sum_{i=1}^{k} \alpha_{i} \cdot p\left(\pmb{x}_{j} | \pmb{\mu}_{i}, \mathbf{\Sigma}_{i}\right)\right) \end{aligned}
$$

- $\mu_{i}$：由$\frac{\partial L L(D)}{\partial \pmb{\mu}_{i}}=0$，有

$$
\sum_{j=1}^{m} \frac{\alpha_{i} \cdot p\left(\pmb{x}_{j} | \pmb{\mu}_{i}, \mathbf{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\pmb{x}_{j} | \pmb{\mu}_{l}, \mathbf{\Sigma}_{l}\right)}\left(\pmb{x}_{j}-\pmb{\mu}_{i}\right)=0
$$

进一步有

$$
\pmb{\mu}_{i}=\frac{\sum_{j=1}^{m} \gamma_{j i} \pmb{x}_{j}}{\sum_{j=1}^{m} \gamma_{j i}}
$$
即各混合成分的均值可通过样本加权平均来估计，样本权重是每个样本属于该成分的后验概率

- $\Sigma_i$：由$\frac{\partial L L(D)}{\partial \pmb{\Sigma}_{i}}=0$，有

$$
\pmb{\Sigma}_{i}=\frac{\sum_{j=1}^{m} \gamma_{j i}\left(\pmb{x}_{j}-\pmb{\mu}_{i}\right)\left(\pmb{x}_{j}-\pmb{\mu}_{i}\right)^{\mathrm{T}}}{\sum_{j=1}^{m} \gamma_{j i}}
$$

- $\alpha_i$：除了要最大化$LL(D)$，还需满足$\alpha_{i} \geqslant 0, \sum_{i=1}^{k} \alpha_{i}=1$，考虑$LL(D)$的拉格朗日形式

$$
L L(D)+\lambda\left(\sum_{i=1}^{k} \alpha_{i}-1\right)
$$

令上式对$\alpha_i$的导数为0，有

$$
\sum_{j=1}^{m} \frac{p\left(\pmb{x}_{j} | \pmb{\mu}_{i}, \pmb{\Sigma}_{i}\right)}{\sum_{l=1}^{k} \alpha_{l} \cdot p\left(\pmb{x}_{j} | \pmb{\mu}_{l}, \mathbf{\Sigma}_{l}\right)}+\lambda=0
$$

两边同乘以$\alpha_i$，又$\lambda=-m$（$\forall i$，上式两边同乘$\alpha_i$仍成立，对所有i求和可得），有

$$
\alpha_{i}=\frac{1}{m} \sum_{j=1}^{m} \gamma_{j i}
$$

即每个高斯成分的混合系数由样本属于该成分的平均后验概率确定

至此，可采用EM算法：
- E步：根据当前参数计算每个样本属于每个高斯成分的后验概率$\gamma_{ji}$
- M步：根据上述三式更新模型参数$\left\\{\left(\alpha\_{i}, \pmb{\mu}\_{i}, \mathbf{\Sigma}\_{i}\right) \vert 1 \leqslant i \leqslant k\right\\}$

算法如下所示，停止条件为达到最大迭代次数或似然函数增长很少或不再增长

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116194743440.png#pic_center)

西瓜数据集上的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116194850669.png#pic_center)

# 5. 密度聚类

密度聚类也称为基于密度的聚类（density-based clustering），此类算法假设聚类结构能通过样本分布的紧密程度确定，密度聚类算法从样本密度的角度考察样本之间的可连接性，并基于可连接样本不断拓展聚类簇以获得最终的聚类结果

著名的DBSCAN基于一组邻域（neighborhood）参数（$\epsilon, M i n P t s$）来刻画样本分布的紧密程度，定义

- $\epsilon$-邻域：对$\pmb x_j\in D$，其$\epsilon$-邻域包含样本集D中与$\pmb x_j$的距离不大于$\epsilon$的样本，即$N\_{\epsilon}\left(\pmb{x}\_{j}\right)=\left\\{\pmb{x}\_{i} \in D \vert \operatorname{dist}\left(\pmb{x}\_{i}, \pmb{x}\_{j}\right) \leqslant \epsilon\right\\}$
- 核心对象（core object）：若$\pmb x_j$的$\epsilon$-邻域至少包含$MinPts$个样本，即$\vert N_{\epsilon}\left(\pmb{x}_{j}\right)\vert \geqslant M i n P t s$，则$\pmb x_j$是一个核心对象
- 密度直达（directly density-reachable）：若$\pmb x_j$位于$\pmb x_i$的$\epsilon$-邻域中，且$\pmb x_i$是核心对象，则称$\pmb x_j$由$\pmb x_i$密度直达
- 密度可达（density-reachable）：对$\pmb x_i$与$\pmb x_j$，若存在样本序列$\pmb{p}\_{1}, \pmb{p}\_{2}, \ldots, \pmb{p}\_{n}$，其中$\pmb{p}\_{1}=\pmb{x}\_{i}, \pmb{p}\_{n}=\pmb{x}\_{j}$且$\pmb{p}\_{i+1}$由$\pmb{p}\_{i}$密度直达，则称$\pmb x\_j$由$\pmb x\_i$密度可达
- 密度相连（density-connected）：对$\pmb x_i$与$\pmb x_j$，若存在$\pmb x_k$使得$\pmb x_i$与$\pmb x_j$均由$\pmb x_k$密度可达，则称$\pmb x_i$与$\pmb x_j$密度相连

基于以上概念，DBSCAN将簇定义为由密度可达导出的最大的密度相连样本集合，即给定领域参数，簇C是满足以下性质的非空样本子集：

- 连接性（connectivity）：$\pmb{x}\_{i} \in C, \pmb{x}\_{j} \in C \Rightarrow \pmb{x}\_{i} ,\pmb{x}\_{j}$密度相连
- 最大性（maximality）：$\pmb{x}\_{i} \in C$，$\pmb{x}\_{j}$由$\pmb{x}\_{j}$密度可达$\Rightarrow\pmb{x}\_{j} \in C$

如何从训练集D中找出满足以上性质的聚类簇呢？实际上，若$\pmb x$为核心对象，由$\pmb x$密度可达的所有样本组成的集合记为$X=\left\\{x^{\prime} \in D \vert x^{\prime}由x密度可达\right\\}$，则不难证明X即为满足连接性与最大性的簇，算法如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116204717393.png#pic_center)

在西瓜数据集上的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116204816348.png#pic_center)

# 6. 层次聚类

层次聚类（hierarchical clustering）试图在不同层次对数据集进行划分，形成树形的聚类结构，数据集的划分可采用自底而上的聚合策略，也可采用自顶而下的分拆策略

AGNES是一种自底向上聚合策略的层次聚类算法，它先将数据集中的每个样本看作一个初始聚类簇，在算法运行的每一步找出距离最近的两个聚类簇进行合并，该过程不断重复，直至达到预设的聚类簇个数

这里的关键是如何计算聚类簇之前的距离，给定聚类簇$C_i,C_j$：

- 最小距离：$d_{\min }\left(C\_{i}, C\_{j}\right)=\min\ _{\pmb{x} \in C\_{i}, \pmb{z} \in C\_{j}} \operatorname{dist}(\pmb{x}, \pmb{z})$
- 最大距离：$d_{\max }\left(C\_{i}, C\_{j}\right)=\max\ _{\pmb{x} \in C\_{i}, \pmb{z} \in C\_{j}} \operatorname{dist}(\pmb{x}, \pmb{z})$
- 平均距离：$d_{\mathrm{avg}}\left(C\_{i}, C\_{j}\right)=\frac{1}{\vert C_{i}\vert \vert C_{j}\vert } \sum\_{\pmb{x} \in C\_{i}} \sum\_{z \in C\_{j}} \operatorname{dist}(\pmb{x}, \pmb{z})$

使用上述三个距离时AGNES算法相应地称为单链接（single-linkage）、全链接（complete-linkage）、均链接（average-linkage），算法如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116224233522.png#pic_center)

西瓜数据集上的结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116224324793.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191116224358902.png#pic_center)
