---
title: 西瓜书 | 第七章 贝叶斯分类器
author: 钟欣然
date: 2020-12-13 01:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

# 1. 贝叶斯决策论
贝叶斯决策论（Bayesian decision theory）是在概率框架下实施决策的基本方法，对分类任务来说，在所有相关概率都已知的理想情形下，贝叶斯决策论考虑如何基于这些概率和误判损失来选择最优的类别标记。

## 最小化总体风险
假设有N种可能的类别标记，即$\mathcal{Y}=\left\\{c_{1}, c_{2}, \ldots, c_{N}\right\\}$，$\lambda_{ij}$是将真实标记为$c_j$的样本误分类为$c_j$产生的损失。基于后验概率$P(x_i\vert\pmb x)$可获得将样本$\pmb x$分类为$c_i$所产生的期望损失（expected loss），即在样本$\pmb x$上的条件风险（conditional risk）

$$
R\left(c_{i} \vert \pmb{x}\right)=\sum_{j=1}^{N} \lambda_{i j} P\left(c_{j} \vert \pmb{x}\right)
$$

我们的任务是寻找一个判定准则$h: \mathcal{X} \mapsto \mathcal{Y}$以最小化总体风险

$$
R(h)=\mathbb{E}_{\pmb{x}}[R(h(\pmb{x}) \vert \pmb{x})]
$$

**贝叶斯判定准则**（Bayes decision rule）：

在每个样本$\pmb x$上选择最小化条件风险$R(h(\pmb{x}) \vert \pmb{x})$的$h$，则总体风险$R(h)$也将被最小化

$$
h^{*}(\pmb{x})=\underset{c \in \mathcal{Y}}{\arg \min } R(c \vert \pmb{x})
$$

此时，$h^\*$称为贝叶斯最优分类器（Bayes optimal classifier），与之对应的总体风险$R(h^*)$称为贝叶斯风险（Bayes risk），$1-R(h^\*)$反映了分类器所能达到的最好性能，即通过机器学习所能产生的模型精度的理论上限

## 最小化分类错误率

若目标是最小化分类错误率，或者说错判损失相等，有

$$
\lambda_{i j}=\left\{\begin{array}{ll}{0,} & {\text { if } i=j} \\ {1,} & {\text { otherwise }}\end{array}\right.
$$

此时条件风险

$$
R(c \vert \pmb{x})=1-P(c \vert \pmb{x})
$$

进一步，贝叶斯最优分类器为

$$
h^{*}(\pmb{x})=\underset{c \in \mathcal{Y}}{\arg \max } P(c \vert \pmb{x})
$$

即对每个样本选择使后验概率最大的类别标记

## 后验概率$P(c \vert \pmb{x})$的计算

**两种方式**：

- 给定$\pmb x$，可通过直接建模$P(c \vert \pmb{x})$预测c，得到判别式模型（discriminative models），前面介绍的决策树、BP神经网络、支持向量机都可归入判别式模型
- 先对联合概率分布$P(\pmb{x}, c)$建模，再由此求得$P(c \vert \pmb{x})$，得到生成式模型（generative models）

$$
P(c \vert\pmb{x})=\frac{P(\pmb{x}, c)}{P(\pmb{x})}=\frac{P(c) P(\pmb{x} \vert c)}{P(\pmb{x})}
$$

其中，$P(c)$是先验（prior）概率；$P(\pmb{x} \vert c)$是样本相对于类标记的类条件概率（class-conditional probability），或称为似然（likelihood）;$P(\pmb x)$是用于归一化的证据（evidence）因子

**生成式模型**：

对给定样本，证据因子与类标记无关，此时$P(c\vert\pmb{x})$的估计问题就转化为如何估计先验概率和似然。

- 类先验概率表达了样本空间中各类样本所占的比例，根据大数定律，当训练集包含充足的独立同分布的样本时，$P(c)$可以通过各类样本出现的频率进行估计
- 类条件概率涉及关于$\pmb x$的所有属性的联合概率，直接根据样本出现的频率来估计很困难。假设样本的d个属性都是二值的，则样本空间将有$2^d$种可能的取值，这个值往往大于训练样本数m，即很多样本取值在训练集中根本没有出现，但未被观测到与出现概率为零是不同的，下面的几个小节将详细介绍估计类条件概率的方法

# 2. 极大似然估计

估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计，我们可假设这种确定的形式可以被参数向量$\pmb \theta_c$唯一确定，则可以将$P\left(\pmb{x} \vert {c}\right)$记为$P\left(\pmb{x} \vert \pmb{\theta}_{c}\right)$

**参数估计**（parameter estimation）：

- 频率学派（Frequentist）：参数虽然未知，但却是客观存在的固定值，可通过优化似然函数等准则来确定参数值
- 贝叶斯学派（Bayesian）：参数是未观察到的随机变量，其本身也可有分布，因此可假定参数服从一个先验分布，然后基于观测到的数据计算参数的后验分布

**极大似然估计过程**：

令$D_c$表示训练集D中第c类样本组成的集合，假设这些样本是独立同分布的，则有

$$
P\left(D_{c} \vert \pmb{\theta}_{c}\right)=\prod_{\pmb{x} \in D_{c}} P\left(\pmb{x} \vert \pmb{\theta}_{c}\right)
$$

由于连乘易造成下溢，通常采用对数似然

$$
\begin{aligned} L L\left(\pmb{\theta}_{c}\right) &=\log P\left(D_{c} \vert \pmb{\theta}_{c}\right) \\ &=\sum_{\pmb{x} \in D_{c}} \log P\left(\pmb{x} \vert \pmb{\theta}_{c}\right) \end{aligned}
$$

则参数$\pmb{\theta}_{c}$的最大似然估计为

$$
\hat{\pmb{\theta}}_{c}=\underset{\pmb{\theta}_{c}}{\arg \max } L L\left(\pmb{\theta}_{c}\right)
$$

这种参数化的方法虽能使类条件概率估计变得相对简单，但估计结果的准确性严重依赖于所假设的概率分布形式是否符合潜在的真实数据分布。

# 3. 朴素贝叶斯分类器

## 属性条件独立性假设

朴素贝叶斯分类器（naive Bayes classifier）采用属性条件独立性假设（attribute conditional independence assumption）：对已经类别假设所有属性相互独立，即每个属性独立地对分类结果产生影响

$$
P(c \vert \pmb{x})=\frac{P(c) P(\pmb{x} \vert c)}{P(\pmb{x})}=\frac{P(c)}{P(\pmb{x})} \prod_{i=1}^{d} P\left(x_{i} \vert c\right)
$$

此时贝叶斯判定准则为

$$
h_{n b}(\pmb{x})=\underset{c \in \mathcal{Y}}{\arg \max } P(c) \prod_{i=1}^{d} P\left(x_{i} \vert c\right)
$$

用样本估计先验概率

$$
P(c)=\frac{\left\vert D_{c}\right \vert }{\vert D\vert }
$$

对离散属性

$$
P\left(x_{i} \vert c\right)=\frac{\vert D_{c, x_{i}}\vert }{\vert D_{c}	\vert }
$$

对连续属性，假定$p\left(x_{i} \vert c\right) \sim \mathcal{N}\left(\mu_{c, i}, \sigma_{c, i}^{2}\right)$，则有

$$
p\left(x_{i} \vert c\right)=\frac{1}{\sqrt{2 \pi} \sigma_{c, i}} \exp \left(-\frac{\left(x_{i}-\mu_{c, i}\right)^{2}}{2 \sigma_{c, i}^{2}}\right)
$$

## 平滑
若某个属性值在训练集种没有与某个类同时出现过，则直接进行概率估计时类条件概率为0，为避免其他属性携带的信息被训练集中未出现的属性值抹去，在估计概率值时通常要进行平滑（smoothing），常用拉普拉斯修正（Laplacian correction）

令$N$表示训练集D中的类别数，$N_i$表示第i个属性的取值数，则修正如下

$$
\begin{aligned} \hat{P}(c) &=\frac{\vert D_{c}\vert +1}{\vert D\vert +N} \\ \hat{P}(x_{i} \vert c) &=\frac{\vert D_{c, x_{i}}\vert +1}{\vert D_{c}\vert +N_{i}} \end{aligned}
$$

拉普拉斯修正避免了因训练集样本不充分而导致概率估值为零的问题，并且在训练集变大时，修正过程引入的影响也逐渐变得可忽略，使得估值趋向于实际概率值

## 不同场景下的朴素贝叶斯

- 若对预测速度要求较高，则对给定训练集，将所有概率估值事先计算好存储起来，预测时只需查表进行判别
- 若任务数据更替频繁，则可采用懒惰学习（lazy learning），先不进行任何训练，收到预测请求时再根据当前数据集进行概率估值；若有数据增加，则可在现有估值基础上，仅对新增样本的属性值涉及的概率估值进行计数修正即可实现增量学习

# 4.  半朴素贝叶斯分类器

现实任务中属性条件独立性假设很难成立，对此假设进行一定程度的放松，适当考虑一部分属性间的相互依赖信息，从而既不需进行完全联合概率计算，又不至于彻底忽略了比较强的属性依赖关系，即为半朴素贝叶斯分类器。

**独依赖估计**（One-Dependent Estimator）：

假设每个属性在类别之外最多仅依赖于一个其他属性：

$$
P(c \vert \pmb{x}) \propto P(c) \prod_{i=1}^{d} P\left(x_{i} \vert c, p a_{i}\right)
$$

其中$pa_i$为属性$x_i$所依赖的属性，称为$x_i$的父属性，假设对每个属性$x_i$，若其父属性$pa_i$已知，则可采用7.3节所述的方式估计概率值$P\left(x_{i} \vert c, p a_{i}\right)$，接下来，问题的关键转化为如何确定每个属性的父属性

## 超父

假设所有属性都依赖于同一个属性，称为超父（super-parent），然后通过交叉验证等模型选择方法来确定超父属性，由此形成了SPODE（Super-Parent ODE）方法

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191110220233237.png#pic_center)

## TAN
TAN（Tree Augmented naive Bayes）在最大带权生成树（maximum weighted spanning tree）算法的基础上，通过以下步骤将属性间依赖关系约减为上图c的树形结构：

- 计算任意两个属性之间的条件互信息（conditional mutual information）：

$$
I\left(x_{i}, x_{j} \vert y\right)=\sum_{x_{i}, x_{j} ; c \in \mathcal{Y}} P\left(x_{i}, x_{j} \vert c\right) \log \frac{P\left(x_{i}, x_{j} \vert c\right)}{P\left(x_{i} \vert c\right) P\left(x_{j} \vert c\right)}
$$

- 以属性为结点构建完全图，任意两个结点之间边的权重设为$I\left(x_{i}, x_{j} \vert y\right)$
- 构建此完全图的最大带权生成树，挑选根变量，将边设置为有向
- 加入类别结点y，增加从y到每个属性的有向边

条件互信息刻画了两个属性在已经类别情况下的相关性，通过最大生成树算法，TAN实际上仅保留了强相关属性之间的依赖性

## AODE
AODE（Averaged One-Dependent Estimator）是一种基于集成学习机制、更为强大的独依赖分类器，它尝试将每个属性作为超父来构建SPODE，然后讲那些具有足够训练数据支撑的SPODE集成起来作为最终结果

$$
P(c \vert \pmb{x}) \propto \sum_{i=1 \atop D_{x_{i}} \vert \geqslant m^{\prime}} P\left(c, x_{i}\right) \prod_{j=1}^{d} P\left(x_{j} \vert c, x_{i}\right)
$$

其中$D_{x_i}$是在第i个属性上取值为$x_i$的样本的集合，$m'$为阈值常数，且有

$$
\begin{aligned} \hat{P}\left(c, x_{i}\right) &=\frac{\vert D_{c, x_{i}}\vert+1}{\vert D\vert+N_{i}} \\ \hat{P}\left(x_{j} \vert c, x_{i}\right) &=\frac{\vert D_{c, x_{i}, x_{j}}\vert+1}{\vert D_{c, x_{i}}\vert +N_{j}} \end{aligned}
$$

AODE无需模型选择，既能通过预计算节省预测时间，也能采取懒惰学习方式在预测时再进行计数，并且易于实现增量学习

## 高阶依赖
考虑属性间的高阶依赖即将上述的$pa_i$替换为包含k个属性的集合$\pmb p\pmb a_i$，从而将ODE拓展为kDE

需注意的是，随着k的增加，准确估计概率$P(a_i\vert y,\pmb p\pmb a_i)$所需的训练样本数量将以指数级增加，因此，若训练数据非常充分，泛化性能有可能提升，但在有限样本条件下，则又陷入估计高阶联合概率的泥沼

# 5. 贝叶斯网
贝叶斯网（Bayesian network）也称信念网（belief network），借助有向无环图（Directed Acyclic Graph, DAG）来刻画属性间的依赖关系，并使用条件概率表（Conditional Probability Table, CPT）

一个贝叶斯网B由结构G和参数$\Theta$两部分构成，即$B=\langle G, \Theta\rangle$
- G是一个有向无环图，每个结点对应一个属性，若两个属性有直接依赖关系，则由一条边连接起来
- $\Theta$定量描述这种依赖关系，假设属性$x_i$在G中的父结点集为$\pi_i$，则$\Theta$包含了每个属性的条件概率表，$\theta_{x_{i} \vert \pi_{i}}=P_{B}\left(x_{i} \vert \pi_{i}\right)$

例，西瓜问题的一种贝叶斯网结构以及属性“根蒂”的条件概率表

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191112221945195.png#pic_center)

## 结构
贝叶斯网结构有效地表达了属性间的条件独立性，它假设每个属性与它的非后裔属性独立，因此有

$$
P_{B}\left(x_{1}, x_{2}, \ldots, x_{d}\right)=\prod_{i=1}^{d} P_{B}\left(x_{i} \vert \pi_{i}\right)=\prod_{i=1}^{d} \theta_{x_{i} \vert \pi_{i}}
$$

如上图示例，有

$$
P\left(x_{1}, x_{2}, x_{3}, x_{4}, x_{5}\right)=P\left(x_{1}\right) P\left(x_{2}\right) P\left(x_{3} \vert x_{1}\right) P\left(x_{4} \vert x_{1}, x_{2}\right) P\left(x_{5} \vert x_{2}\right)
$$

显然，$x_3,x_4$在给定$x_1$时独立，可简记为$x_{3} \perp x_{4} \vert x_{1}$

**三个变量间的典型依赖关系**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191112222811802.png#pic_center)

- 同父（common parent）结构：给定$x_1$时，$x_3,x_4$条件独立，$x_{3} \perp x_{4} \vert x_{1}$
- V型（V-structure）结构：也称冲撞结构，给定$x_4$时，$x_1,x_2$不条件独立，但若$x_4$的取值完全未知，则$x_1,x_2$相互独立，这样的独立性称为边际独立性（marginal independence），记为$x_{1} \perp\!\!\!\perp x_{2}$ 

$$
\begin{aligned} P\left(x_{1}, x_{2}\right) &=\sum_{x_{4}} P\left(x_{1}, x_{2}, x_{4}\right) \\ &=\sum_{x_{4}} P\left(x_{4} \vert x_{1}, x_{2}\right) P\left(x_{1}\right) P\left(x_{2}\right) \\ &=P\left(x_{1}\right) P\left(x_{2}\right) \end{aligned}
$$

- 顺序结构：给定$x$的值，$y,z$条件独立，$y \perp z \vert x$

**关于边际独立性**

一个变量的取值确定与否能对另外两个变量间的独立性发生影响，在V型结构中，$x_{1} \perp x_{2} \vert x_{4}$不成立，但有$x_{1} \perp\!\!\!\perp x_{2}$；在同父结构和顺序结构中，分别有$x_{3} \perp x_{4} \vert x_{1},y \perp z \vert x$，但$x_{3} \perp\!\!\!\perp x_{4},y \perp\!\!\!\perp z$不成立

**有向分离**

为了分析有向图中变量间的条件独立性，可使用有向分离（D-separation）

- 将有向图转变为无向图，称为道德图（moral graph）
	- 在图中V型结构的两个父结点间加上一条无向边，此过程称为道德化（moralization）
	- 将所有有向边改成无向边

		例如，西瓜问题贝叶斯网对应的道德图

![在这里插入图片描述](https://img-blog.csdnimg.cn/201911122312546.png#pic_center)

- 基于道德图直观、迅速地找到变量间的条件独立性：
	- 假定道德图中有变量$x,y$和变量集合$\mathbf{z}=\left\{z_{i}\right\}$，若变量$x,y$在道德图中能被$\mathbf{z}$分开，即从图中去掉$\mathbf{z}$后，$x,y$分属两个连通分支，则$x,y$被$\mathbf{z}$有向分离，$x \perp y \vert \mathbf{z}$成立
	- 
## 学习

**评分函数**

贝叶斯网的首要任务就是根据训练数据集找出结构最恰当的贝叶斯网，常用办法是评分搜索。具体来说，定义一个评分函数（score function），以此来评估贝叶斯网与训练数据的契合程度，然后基于这个评分函数来寻找最优的贝叶斯网。贝叶斯网引入了归纳偏好。

常用的评分函数基于信息论准则，将学习问题看作一个数据压缩任务，学习的目标是找到一个能以最短编码长度描述训练数据的模型，此时编码的长度包括了描述模型自身所需的编码位数和使用该模型描述数据所需的编码位数。对于贝叶斯网学习而言，模型就是贝叶斯网，同时每个贝叶斯网描述了一个在训练数据上的概率分布，自有一套编码机制能使哪些经常出现的样本有更短的编码。我们应选择哪个综合编码长度（包括描述网络和编码数据）最短的贝叶斯网，即最小描述长度（Minimal Description Length, MDL）准则。

给定训练集$D=\left\{\pmb{x}_{1}, \pmb{x}_{2}, \ldots, \pmb{x}_{m}\right\}$，贝叶斯网$B=\langle G, \Theta\rangle$在D上的评分函数可写为

$$
s(B \vert D)=f(\theta)\vert B\vert -L L(B \vert D)
$$

其中，$\vert B\vert$是贝叶斯网的

$$
L L(B \vert D)=\sum_{i=1}^{m} \log P_{B}\left(\pmb{x}_{i}\right)
$$

显然，第一项计算编码贝叶斯网B所需的编码位数，第二项计算B对应的概率分布对D描述得有多好，学习认为转化为优化任务，即寻找一个贝叶斯网B使评分函数$s(B \vert D)$最小

- AIC（Akaike Information Criterion）评分函数：$f(\theta)=1$，即每个参数用1编码位描述

$$
\operatorname{AIC}(B \vert D)=\vert B\vert-L L(B \vert D)
$$

- BIC（Bayesian Information Criterion）评分函数：$f(\theta)=\frac12\log m$，即每个参数用$\frac12\log m$编码位描述

$$
\operatorname{BIC}(B \vert D)=\frac12\log m\vert B\vert-L L(B \vert D)
$$

- 负对数似然：$f(\theta)=0$，即不计算对网络进行编码的长度

不难发现，若网络结构G固定，则评分函数的第一项为常数，最小化评分函数等价于对参数$\Theta$的最大似然估计，可直接在训练数据D上通过经验估计获得

$$
\theta_{x_{i} \vert \pi_{i}}=\hat{P}_{D}\left(x_{i} \vert \pi_{i}\right)
$$

因此，为了最小化评分函数，只需对网络结构进行搜索，而候选结构的最优参数可直接在训练集上计算得到

**网络结构搜索**

从所有可能的网络结构空间搜索最优结构是NP难问题

> - NP类问题（Nondeterminism Polynomial）：在多项式时间内“可验证”的问题。也就是说，不能判定这个问题到底有没有解，而是猜出一个解来在多项式时间内证明这个解是否正确。即该问题的猜测过程是不确定的，而对其某一个解的验证则能够在多项式时间内完成。
> - 多项式时间（Polynomial）：对于规模为n的输入，它们在最坏的情况下的运行时间为$O(n^k)$，其中k为某个常数，则该算法为多项式时间的算法
————————————————
原文链接：[https://blog.csdn.net/u014295667/article/details/47090639](https://blog.csdn.net/u014295667/article/details/47090639)

有两种常用策略能在有限时间内求得近似解：

- 贪心法：从某个网络结构出发，每次调整一条边（增加、删除或调整方向），直到评分函数不再降低为止
- 通过给网络结构施加约束来消减搜索空间，如将网络结构限定为树形结构

## 推断
贝叶斯网训练好后就能用来回答查询（query），即通过一些属性变量的观测值来推断其他属性变量的取值，这个过程称为推断（inference），已知变量观测值称为证据（evidence）

- 精确推断：直接根据贝叶斯网定义的联合概率分布来精确计算后验概率，但已经被证明是NP难的，当网络结点较多、连接稠密时，难以进行精确推断
- 近似推断：通过降低精度要求，在有限时间内求得近似解，常用吉布斯采样（Gibbs sampling）

**符号表示**

- $\mathbf{Q}=\left\\{Q_{1}, Q_{2}, \ldots, Q_{n}\right\\}$表示待查询变量
- $\mathbf{E}=\left\\{E_{1}, E_{2}, \ldots, E_{k}\right\\}$是证据变量，已知其取值为$\mathbf{e}=\left\\{e_{1}, e_{2}, \ldots, e_{k}\right\\}$
- 目标是计算后验概率$P(\mathbf{Q}=\mathbf{q} \vert \mathbf{E}=\mathbf{e})$，其中$\mathbf{q}=\left\\{q_{1}, q_{2}, \ldots, q_{n}\right\\}$是待查询变量的一组取值

**过程**

吉布斯采样先随机产生一个与证据$\mathbf{E}=\mathbf{e}$一致的样本$\mathbf{q}^0$作为初始点，每一步从当前样本出发产生下一个样本，具体过程为，在第t次采样中，先假设$\mathbf{q}^{t}=\mathbf{q}^{t-1}$，然后对非证据变量逐个进行采样改变其取值，采样概率根据贝叶斯网B和其他变量的当前取值（$\mathbf{Z}=\mathbf{z}$）计算获得，假定经过T次采样得到的与$\mathbf{q}$一致的样本共有$n_q$个，则可估算出后验概率

$$
P(\mathbf{Q}=\mathbf{q} \vert \mathbf{E}=\mathbf{e}) \simeq \frac{n_{q}}{T}
$$

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019111314173095.png#pic_center)

**实质**

吉布斯采样是在贝叶斯网所有变量的联合状态空间与证据$\mathbf{E}=\mathbf{e}$一致的子空间中进行随机漫步（random walk），每一步仅依赖于前一步的状态，这是一个马尔科夫链（Markov chain），在一定条件下，无论从什么初始状态开始，马尔科夫链第t步的状态分布在$t\rightarrow \infty$时必收敛于一个平稳分布（stationary distribution）

对吉布斯采样来说，这个分布恰好是$P(\mathbf{Q} \vert \mathbf{E}=\mathbf{e})$，因此，在T很大时，吉布斯采样相当于根据$P(\mathbf{Q} \vert \mathbf{E}=\mathbf{e})$采样

需注意的是，由于马尔科夫链通常需要很长时间才能趋于平稳分布，因此吉布斯采样算法的收敛速度较慢，若贝叶斯网中存在极端概率0或1，则不能保证马尔科夫链存在平稳分布，此时吉布斯采样会给出错误的估计结果

# 6. EM算法

## 隐变量情形下贝叶斯网的参数估计

隐变量（latent variable）即为未观测变量，令$\mathbf{X}$表示已观测变量集，$\mathbf{Z}$表示未观测变量集，$\Theta$表示模型参数，若欲对$\Theta$进行极大似然估计，则应最大化对数似然

$$
L L(\Theta \vert \mathbf{X}, \mathbf{Z})=\ln P(\mathbf{X}, \mathbf{Z} \vert \Theta)
$$

由于$\mathbf{Z}$是隐变量，上式无法直接求解，此时可以通过对$\mathbf{Z}$计算期望，最大化已观测数据的对数边际似然（marginal likelihood）

$$
L L(\Theta \vert \mathbf{X})=\ln P(\mathbf{X} \vert \Theta)=\ln \sum_{\mathbf{Z}} P(\mathbf{X}, \mathbf{Z} \vert \Theta)
$$

## EM算法

EM（Expectation-Maximization）是常用的估计参数隐变量的迭代式方法，以初始值$\Theta^0$为起点，执行以下步骤直至收敛：

- E步：若参数$\Theta$已知，则可根据训练数据推断出最优因变量$\mathbf{Z}$的值，即基于$\Theta^t$推断隐变量$\mathbf{Z}$的期望，记为$\mathbf{Z}^t$
- M步：若$\mathbf{Z}$的值已知，则可对参数$\Theta$做极大似然估计，即基于已观测变量$\mathbf{X},\mathbf{Z}$对参数$\Theta$作极大似然估计，记为$\Theta^{t+1}$

进一步，若我们不是取$\mathbf{Z}$的期望，而是基于$\Theta^t$计算隐变量$\mathbf{Z}$的概率分布$P\left(\mathbf{Z} \vert \mathbf{X}, \Theta^{t}\right)$，则EM算法的步骤是：

- E步：以当前参数$\Theta^t$推断隐变量分布$P\left(\mathbf{Z} \vert \mathbf{X}, \Theta^{t}\right)$，并计算对数似然关于$\mathbf{Z}$的期望

$$
Q\left(\Theta \vert \Theta^{t}\right)=\mathbb{E}_{\mathbf{Z} \vert \mathbf{X}, \mathbf{\theta}^{t}} L L(\Theta \vert \mathbf{X}, \mathbf{Z})
$$

- M步：寻找参数最大化期望似然

$$
\Theta^{t+1}=\underset{\Theta}{\arg \max } Q\left(\Theta \vert \Theta^{t}\right)
$$

事实上，隐变量估计问题也可通过梯度下降等优化算法求解，但由于求和的项数将随着隐变量的数目以指数级上升，会给梯度计算带来麻烦，而EM算法则可看作一种非梯度优化方法
