---
title: 西瓜书 | 第十一章 特征选择与稀疏学习
author: 钟欣然
date: 2020-12-13 05:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

# 1. 子集搜索与评价

将属性称为特征（feature），对当前学习任务有用的属性称为相关特征（relevant feature）、没什么用的属性称为无关特征（irrelevant feature），从给定的特征集合中选择出相关特征子集的过程，称为特征选择（feature selection），是一个重要的数据预处理（data preprocessing）过程

**特征选择的必要性**：

- 大为减轻位数暂难问题，与降维有相似动机，是处理高维数据的两大主流技术
- 去除不相关特征往往会降低学习任务的难度

**冗余特征**：

特征选择中所谓的无关特征是指与当前学习任务无关，而冗余特征（redundant feature）是指包含的信息能从其他特征中推演出来的特征。很多时候冗余特征不起作用，去除它们会减轻学习过程的负担，但有冗余特征会降低学习任务的难度，即若某个冗余特征恰好对应了完成学习任务所需的中间概念则是有益的。本章暂且假定数据中不涉及冗余特征，并且假定初始的特征集合包含了所有的重要信息

**特征选择的步骤**：

产生一个候选子集，评价出它的好坏，基于评价结果产生下一个候选自己，再对其进行评价，这个过程持续进行下去，直至无法找到更好的候选子集为止。

- 子集搜索（subset search）：即如何根据评价结果获取下一个候选特征子集
	- 前向（forward）搜索：给定特征集合$\\{a_1,a_{2}, \dots, a_{d}\\}$，将每个特征看作一个候选子集，对这d个候选单特征子集进行评价，假定$\\{a_{2}\\}$最优，于是将其作为第一轮的选定集，然后在上一轮的选定集中加入一个特征，构成包含两个特征的候选子集，假定在这d-1个候选两特征子集中$\\{a_{2}，a_4\\}$最优，且由于$\\{a_{2}\\}$，于是将$\\{a_{2}，a_4\\}$作为本轮的选定集，以此类推，直至第k+1轮最优的候选子集不如上一轮的选定集，则停止生成候选子集，并将上一轮的k特征集合作为特征选择结果
	- 后向（backward）搜索：从完整的特征集合开始，每次尝试去掉一个无关特征
	- 双向（bidirectional）搜索：将前向和后向搜索结合起来，每一轮逐渐增加选定相关特征（这些特征在后续轮中将确定不会被取出）、同时减少无关特征
	- 显然上述策略都是贪心的，因为它们仅考虑了使本轮选定集左右，例如假设在第三轮选择了$\left\\{a_{2}, a_{4}, a_{5}\right\\}$，在第四轮却是$\left\\{a_{2}, a_{4}, a_{6}, a_{8}\right\\}$比所有的$\left\\{a_{2}, a_{4}, a_{5}, a_{i}\right\\}$都更优，遗憾的是若不穷举，上述问题无法避免
- 子集评价（subset evaluation）：即如何评价候选特征子集的好坏
	- 属性子集A的信息增益：给定数据集D，假定D中第i类样本所占的比例为$p_{i}(i=1,2, \dots,\vert\mathcal{Y}\vert)$，假定样本属性均为离散值，对属性子集A，假定根据其取值将D分成了V个子集$\left\\{D^{1}, D^{2}, \ldots, D^{V}\right\\}$，每个子集中的样本在A上取值相同，于是属性子集A的信息增益见（1）式，其中信息熵定义见（2）式。信息增益越大，意味着特征子集A实际上包含的有助于分类的信息越多，因此可以对每个候选特征子集计算其信息增益作为评价准则
	- 更一般的，特征子集A实际上确定了对数据集D的一种划分，每个划分区域对应着A上的一个取值，而样本标记信息Y对应着对D的真是划分，通过估算这两个划分的差异，就能对A进行评价，差异越小A越好，能判断两个划分差异的机制都能用于特征子集评价，如信息熵

$$
\operatorname{Gain}(A)=\operatorname{Ent}(D)-\sum_{v=1}^{V} \frac{\left|D^{v}\right|}{|D|} \operatorname{Ent}\left(D^{v}\right)\tag 1
$$

$$
\operatorname{Ent}(D)=-\sum_{i=1}^{| \mathcal{Y |}} p_{k} \log _{2} p_{k}\tag 2
$$

将特征子集搜索机制与子集评价机制相结合，即可得到特征选择方法，如将前向搜索与信息熵相结合，与决策树的算法非常相似，其它的特征选择方法未必像决策树特征选择这么明显，但本质上都是显式或隐式地结合了某种或多种子集搜索和评价机制

常见的特征选择方法大致可分为三类：过滤式（2.）、包裹式（3.）和嵌入式（4.）

# 2. 过滤式选择

过滤式（filter）方法先对数据集进行特征选择，然后再训练学习器，特征选择过程与后续学习器无关，这相当于先用特征选择过程对初始特征进行过滤，再用过滤后的特征来训练模型

Relief（Relevant Feature）是一种著名的过滤式特征选择方法，该方法设计了一个相关统计量来度量特征的重要性。该统计量是一个向量，每一个分量对应一个初始特征，特征子集的重要性则是由子集中每个特征对应的相关统计量分量之和来决定，因此只需设置一个阈值$\tau$，选择比其大的相关统计量对应的特征或指定欲选取的特征个数k然后选择相关统计量分量最大的k个统计量

Relief的关键是确定相关统计量，给定训练集$\left\\{\left(\pmb{x}\_{1}, y\_{1}\right)\right.,\left.\left(\pmb{x}\_{2}, y\_{2}\right), \dots,\left(\pmb{x}\_{m}, y\_{m}\right)\right\\}$，对每个示例$\pmb x\_i$，先在其同类样本中寻找其最近邻$\pmb{x}\_{i, \mathrm{nh}}$，称为猜中近邻（near-hit），再从其异类样本中寻找其最近邻$\pmb{x}\_{i, \mathrm{nm}}$，称为猜错近邻（near-miss），相关统计量对应属性j的分量为

$$
\delta^{j}=\sum_{i}-\operatorname{diff}\left(x_{i}^{j}, x_{i, \mathrm{nh}}^{j}\right)^{2}+\operatorname{diff}\left(x_{i}^{j}, x_{i, \mathrm{nm}}^{j}\right)^{2}
$$

其中，$x_a^j$表示样本$\pmb x_a$在属性j上的取值，$\operatorname{diff}\left(x_{a}^{j}, x_{b}^{j}\right)$取决于属性j的类型，若属性j为离散型时，二者取值相同则为0，不同则为1，若为连续型，则为$\vert x_{a}^{j}-x_{b}^{j}\vert$，注意$x_{a}^{j},x_{b}^{j}$已规范到[0,1]区间

直观理解为，若样本与其猜中近邻在属性j上的距离小于与其猜错近邻的距离，说明该属性在区分同类和异类样本时是有益的，增大其分量，反之亦然，最后对不同样本求平均值，分量值越大，对应属性的分类能力越强

实际上Relief只需在数据集上采样而不必在整个数据集上估计相关统计量，其时间开销随采样次数和原始特征数线性增长，是一个运行效率很高的过滤式特征选择算法，但只适用于二分类问题，其拓展变体Relief-F可处理多分类问题，对一个第k类的示例，猜错近邻在除第k类之外每一类中都找到一个最近邻作为猜错近邻，记为$\pmb{x}_{i, l, \mathrm{nm}}(l=1,2, \ldots,\vert\mathcal{Y}vert ; l \neq k)$，于是相关统计量对应属性j的分量为

$$
\delta^{j}=\sum_{i}-\operatorname{diff}\left(x_{i}^{j}, x_{i, \mathrm{nh}}^{j}\right)^{2}+\sum_{l \neq k}\left(p_{l} \times \operatorname{diff}\left(x_{i}^{j}, x_{i, l, \mathrm{nm}}^{j}\right)^{2}\right)
$$

其中，$p_l$为第$l$类样本在数据集D中占的比例

# 3. 包裹式选择
包裹式（wrapper）特征选择直接把最终将要使用的学习器的性能作为特征子集的评价准则，即为给定学习器选择最有利于其性能、量身定做的特征子集

一般而言，包裹式特征选择的最终学习器性能好过过滤式特征选择，但由于在特征选择过程中需多次训练学习器，计算开销比过滤式特征选择大得多

LVW（Las Vegas Wrapper）是一个典型的特征选择方法，在拉斯维加斯方法（Las Vegas method）框架下使用随机策略来进行子集搜索，并以最终分类器的误差为特征子集评价准则，算法如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191123104527200.png#pic_center)

若初始特征数很多、T设置较大，可能算法运行很长时间都达不到停止条件，若有运行时间限制，则有可能给不出解

# 4. 嵌入式选择与$L_1$正则化
嵌入式（embedding）特征选择将特征选择过程与学习器的训练过程融为一体，二者在一个优化过程中完成，即在学习器训练过程中自动地进行了特征选择，而在过滤式和包裹式中二者有明显的分别，先完成一个再完成另一个或两者交替进行

## 岭回归与LASSO

给定数据集$D=\left\\{\left(\pmb{x}\_{1}, y\_{1}\right),\left(\pmb{x}\_{2}, y\_{2}\right), \ldots,\left(\pmb{x}\_{m}, y\_{m}\right)\right\\}$，考虑线性回归模型，以平方误差为损失函数，则优化目标为

$$
\min _{\pmb{w}} \sum_{i=1}^{m}\left(y_{i}-\pmb{w}^{\mathrm{T}} \pmb{x}_{i}\right)^{2}
$$

当样本特征很多而样本数相对较小时，很容易陷入过拟合，为了缓解这一问题，引入正则化项

- 岭回归（ridge regression）：若使用$L_2$范数正则化，则有

$$
\min _{\pmb{w}} \sum_{i=1}^{m}\left(y_{i}-\pmb{w}^{\mathrm{T}} \pmb{x}_{i}\right)^{2}+\lambda\|\pmb{w}\|_{2}^{2}
$$

其中，正则化参数$\lambda>0$

- LASSO（Least Absolute Shrinkage and Selection Operator）：使用$L_1$范数正则化

$$
\min _{\pmb{w}} \sum_{i=1}^{m}\left(y_{i}-\pmb{w}^{\mathrm{T}} \pmb{x}_{i}\right)^{2}+\lambda\|\pmb{w}\|_{1}
$$

其中，正则化参数$\lambda>0$

二者都有助于降低过拟合风险，但LASSO更易于获得稀疏（sparse）解，即它求得的$\pmb w$会有更少的非零分量，直观例子如下图，假定$\pmb x$有两个属性，因此两个方法解出的$\pmb w$也会有两个分量，即$w_1,w_2$，我们将其作为两个坐标轴，然后再图中绘出两个优化目标的第一项的等值线，即在$(w_1,w_2)$空间中平方误差项取值相同的点的连线，再分别绘制出$L_1$范数和$L_2$范数的等值线，要在平方误差项和正则化项之间折中，即出现在图中平方误差项等值线和正则化项等值线的相交处，不难看出，$L_1$范数的交点常出现在坐标轴上，即$w_1$或$w_2$为0，$L_2$范数的交点常出现在某个象限中，即$w_1,w_2$均非0，即$L_1$范数更易于得到稀疏解

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191123110130248.png#pic_center)

$\pmb w$取得稀疏解意味着初始的特征中只有对应着$\pmb w$的非零分量的特征才会出现在最终模型中，于是求解$L_1$范数郑泽华的结果是得到了仅采用一部分初始特征的模型，即特征选择过程与学习器训练过程融为一体，同时完成

## 求解过程
$L_1$正则化问题的求解可使用近端梯度下降（Proximal Gradient Descent, PGD），对优化目标

$$
\min _{\pmb{x}} f(\pmb{x})+\lambda\|\pmb{x}\|_{1}
$$

若$f(\pmb{x})$可导，且$\nabla f$满足L-Lipschitz条件，即存在常数$L>0$使得

$$
\left\|\nabla f\left(\pmb{x}^{\prime}\right)-\nabla f(\pmb{x})\right\|_{2}^{2} \leqslant L\left\|\pmb{x}^{\prime}-\pmb{x}\right\|_{2}^{2} \quad\left(\forall \pmb{x}, \pmb{x}^{\prime}\right)
$$

则在$\pmb x_k$附近可将$f(\pmb{x})$通过二阶泰勒展式近似为

$$
\begin{aligned} \hat{f}(\pmb{x}) & \simeq f\left(\pmb{x}_{k}\right)+\left\langle\nabla f\left(\pmb{x}_{k}\right), \pmb{x}-\pmb{x}_{k}\right\rangle+\frac{L}{2}\left\|\pmb{x}-\pmb{x}_{k}\right\|^{2} \\ &=\frac{L}{2}\left\|\pmb{x}-\left(\pmb{x}_{k}-\frac{1}{L} \nabla f\left(\pmb{x}_{k}\right)\right)\right\|_{2}^{2}+\mathrm{const} \end{aligned}
$$

其中，const表示与$\pmb x$无关的常数，显然上式的最小值在如下处获得

$$
\pmb{x}_{k+1}=\pmb{x}_{k}-\frac{1}{L} \nabla f\left(\pmb{x}_{k}\right)
$$

因此若通过梯度下降法对$f(\pmb{x})$进行最小化，则每一步梯度下降迭代实际上等价于最小化二次函数，将其推广至最初的优化目标，得到每一步的迭代为

$$
\pmb{x}_{k+1}=\underset{\pmb{x}}{\arg \min } \frac{L}{2}\left\|\pmb{x}-\left(\pmb{x}_{k}-\frac{1}{L} \nabla f\left(\pmb{x}_{k}\right)\right)\right\|_{2}^{2}+\lambda\|\pmb{x}\|_{1}
$$

即在每一步对$f(\pmb{x})$进行梯度下降迭代的同时考虑$L_1$范数最小化

对于上式，可先计算$\pmb{z}=\pmb{x}\_{k}-\frac{1}{L} \nabla f\left(\pmb{x}\_{k}\right)$，然后求解

$$
\pmb{x}_{k+1}=\underset{\pmb{x}}{\arg \min } \frac{L}{2}\|\pmb{x}-\pmb{z}\|_{2}^{2}+\lambda\|\pmb{x}\|_{1}
$$

令$x^i$表示$\pmb x$的第i个分量，将上式按分量展开可看出，其中不存在$x^ix^j(i\neq j)$这样的项，即$\pmb x$的各分量互不影响，于是上式有闭式解

$$
x_{k+1}^{i}=\left\{\begin{array}{ll}{z^{i}-\lambda / L,} & {\lambda / L<z^{i}} \\ {0,} & {\left|z^{i}\right| \leqslant \lambda / L} \\ {z^{i}+\lambda / L,} & {z^{i}<-\lambda / L}\end{array}\right.
$$

因此，通过PGD能使LASSO和其他基于$L_1$范数最小化的方法得以快速求解

# 5. 稀疏表示与字典学习

**两种稀疏**：

把数据集D看成是一个矩阵，每行为一个样本，每列为一个特征。特征选择考虑的问题是特征具有稀疏性，即矩阵中的许多列与当前学习任务无关，现在我们考虑另外一种稀疏性，D所对应的矩阵中存在很多零元素，但这些元素并不是以整列、整行形式存在的，如在文档分类任务中，每个文档看成是一个样本，每个字（或词）看成是一个特征，很多字在某个文档中没有出现过，对应的值为0

**稀疏的好处**：

当样本具有这样的稀疏表达形式时，对学习任务来说有很多好处，如线性支持向量机之所以在文本数据上有很好的性能，正是由于文本数据在使用上述的字频表示后具有高度的稀疏性，使大多数问题变得线性可分，且由于对稀疏矩阵已有很多高效的存储方法，不会造成存储上的巨大负担

**字典学习**：

若给定数据集D是稠密的，即普通非稀疏数据，能否将其转化为稀疏表示（sparse representation）形式，从而享有稀疏性带来的好处呢？需注意，应为恰当稀疏而非过度稀疏

首先，我们需学习出一个字典，为普通稠密表达的样本找到合适的样本字典，将样本转化为合适的稀疏表示形式，从而使学习任务得以简化，模型复杂度得以降低，通常称为字典学习（dictionary learning）或稀疏编码（sparse coding），两者稍有差别，前者更侧重于学得字典的过程，后者更侧重于对样本进行稀疏表达的过程，由于两者通常是在一个优化求解过程中完成的，因此下面我们不做进一步区分，笼统地称为字典学习

给定数据集$\left\\{\pmb{x}\_{1}, \pmb{x}\_{2}, \ldots, \pmb{x}\_{m}\right\\}$，字典学习最简单的形式为

$$
\min _{\mathbf{B}, \pmb{\alpha}_{i}} \sum_{i=1}^{m}\left\|\pmb{x}_{i}-\mathbf{B} \pmb{\alpha}_{i}\right\|_{2}^{2}+\lambda \sum_{i=1}^{m}\left\|\pmb{\alpha}_{i}\right\|_{1}
$$

其中$\mathbf{B} \in \mathbb{R}^{d \times k}$为字典矩阵，k称为字典的词汇量，通常由用户指定，$\pmb{\alpha}\_{i} \in \mathbb{R}^{k}$则是样本$\pmb{x}\	_{i} \in \mathbb{R}^{d}$的稀疏表示，显然上式的第一项是希望能很好地重构，第二项则是希望尽量稀疏

与LASSO相比，上式显然麻烦得多，但是我们可采用变量交替优化的策略来求解上式

- 首先，固定住字典$\mathbf{B}$，将上式按照分量展开，可看出其它不涉及$\alpha_{i}^{u} \alpha_{i}^{v}(u \neq v)$这样的交叉项，因此可参照LASSO的解法求解下式，从而为每个样本$\pmb x_i$找到对应的$\pmb \alpha_i$ 

$$
\min _{\pmb{\alpha}_{i}}\left\|\pmb{x}_{i}-\mathbf{B} \pmb{\alpha}_{i}\right\|_{2}^{2}+\lambda\left\|\pmb{\alpha}_{i}\right\|_{1}
$$

- 其次，以$\pmb \alpha_i$为初值来更新字典$\mathbf{B}$ 

$$
\min _{\mathbf{B}}\|\mathbf{X}-\mathbf{B} \mathbf{A}\|_{F}^{2}
$$

其中$\mathbf{X}=\left(\pmb{x}\_{1}, \pmb{x}\_{2}, \ldots, \pmb{x}\_{m}\right) \in \mathbb{R}^{d \times m}, \mathbf{A}=\left(\pmb{\alpha}\_{1}, \pmb{\alpha}\_{2}, \ldots, \pmb{\alpha}\_{m}\right) \in \mathbb{R}^{k \times m},\|\cdot\|_{F}$是矩阵的Frobenius范数，上式有多种求解方法，常用的有基于逐列更新策略的KSVD，令$\pmb b\_i$表示$\mathbf B$的第i列，$\pmb \alpha^i$表示稀疏矩阵$\mathbf A$的第i行，则上式可重写为

$$
\begin{aligned} \min _{\mathbf{B}}\|\mathbf{X}-\mathbf{B} \mathbf{A}\|_{F}^{2} &=\min _{\pmb{b}_{i}}\left\|\mathbf{X}-\sum_{j=1}^{k} \pmb{b}_{j} \pmb{\alpha}^{j}\right\|_{F}^{2} \\ &=\min _{\pmb{b}_{i}}\left\|\left(\mathbf{X}-\sum_{j \neq i} \pmb{b}_{j} \pmb{\alpha}^{j}\right)-\pmb{b}_{i} \pmb{\alpha}^{i}\right\|_{F}^{2} \\ &=\min _{\pmb{b}_{i}}\left\|\mathbf{E}_{i}-\pmb{b}_{i} \pmb{\alpha}^{i}\right\|_{F}^{2} \end{aligned}
$$

在更新字典的第i列时，其它各列都是固定的，因此$\mathbf{E}\_{i}=\sum_{j \neq i} \mathbf{b}\_{j} \pmb{\alpha}^{j}$是固定的，于是最小化上式原则上只需对$\mathbf E\_i$进行奇异值分解以取得最大奇异值所对应的正交向量，然而直接对$\mathbf E\_i$进行奇异值分解会同时修改$\pmb{b}\_{i}$和$\pmb{\alpha}^{i}$，从而可能破坏$\mathbf A$的稀疏性，为避免这种情况，KSVD对$\mathbf E\_i$和$\pmb{\alpha}^{i}$进行专门处理：$\pmb{\alpha}^{i}$仅保留非零元素，$\mathbf E\_i$仅保留$\pmb b\_i$和$\pmb{\alpha}^{i}$的非零元素的乘积项，然后再进行奇异值分解，保持了第一步所得到的稀疏性

反复迭代以上两步，得到最终解。可通过设置词汇量k的大小来控制字典的规模，进而影响到稀疏程度

# 6. 压缩感知

在现实任务中，我们常希望根据部分信息来回复全部信息，例如在数据通讯中要将模拟信号转换为数字信息，根据奈奎斯特（Nyquist）采样定理，令采样频率达到模拟信号最高频率的两倍以上，则采样后的数字信号就保留了模拟信号的全部信息，即由此获得的数字信号能精确重构原模拟信号，然而为了便于传输、存储，实践中常对采样的数字信号进行压缩，这就可能损失一些信息，在信号传输过程中由于丢包等问题，又可能损失部分信息，接收方基于收到的信号能精确地重构出原信号吗？压缩感知（conpressed sensing）为解决此类问题提供了新思路

假定有长度为m的离散信号$\pmb x$，我们以远小于奈奎斯特采样定理要求的采样频率进行采样，得到长度为n的采样后信号$\pmb y$，$n \ll m$，即

$$
\pmb{y}=\mathbf{\Phi} \pmb{x}
$$

其中$\mathbf{\Phi} \in \mathbb{R}^{n \times m}$是对信号$\pmb x$的测量矩阵，它确定了以什么频率采样以及如何将采样样本组成采样后的信号，由于$n \ll m$，因此$\pmb{y},\mathbf{\Phi}, \pmb{x}$组成的是一个欠定方程，无法轻易求出数值解。不妨假定存在某个线性变换$\Psi \in \mathbb{R}^{m \times m}$，使得$\pmb{x}=\pmb{\Psi} \pmb{s}$，因此有

$$
\pmb{y}=\pmb{\Phi} \Psi s=\mathbf{A} s
$$

其中$\mathbf{A}=\mathbf{\Phi} \Psi \in \mathbb{R}^{n \times m}$，若能从$\pmb{y}$中恢复出$\pmb{s}$，则可恢复出$\pmb{x}$，若$\pmb{s}$具有稀疏性，该问题能很好地得以解决，因为稀疏性使得未知因素的影响大为减少，此时$\Psi$称为稀疏基，$\mathbf A$的作用类似于字典，将信号转换为稀疏表示

与特征选择、稀疏表示不同，压缩感知关注的是如何利用信号本身所具有的稀疏性，从部分观测样本中恢复原信号，通常认为压缩感知分为感知测量和重构恢复两个阶段，前者关注如何对原始信号进行处理以获得稀疏样本表示，后者关注如何基于稀疏性从少量观测中恢复原信号，这是压缩感知的精髓，通常压缩感知指该部分

**限定等距性**（Restricted Isometry Property, RIP）：

对大小为$n \times m(n \ll m)$的矩阵$\mathbf A$，若存在常数$\delta_{k} \in(0,1)$使得对任意的$\pmb s$和$\mathbf A$的所有子矩阵$\mathbf{A}_{k} \in \mathbb{R}^{n \times k}$，有

$$
\left(1-\delta_{k}\right)\|\pmb s\|_{2}^{2} \leqslant\left\|\mathbf{A}_{k} \pmb s\right\|_{2}^{2} \leqslant\left(1+\delta_{k}\right)\|\pmb{s}\|_{2}^{2}
$$

则称$\mathbf A$满足k限定等距性（k-RIP），此时可通过下面的优化问题近乎完美地从$\pmb y$中恢复出系数信号$\pmb s$，进而恢复出$\pmb x$：

$$
\begin{array}{c}\underset{\pmb{s}}{\min} & \|\pmb{s}\|_{0} \\ \text { s.t. }  & \pmb{y}=\mathbf{A} s\end{array}
$$

然而上式涉及$L_0$范数最小化，这是个NP难问题，值得庆幸的是$L_1$范数最小化在一定条件下与$L_0$范数最小化问题共解，于是实际上只需关注

$$
\begin{array}{c}\underset{\pmb{s}}{\min} & \|\pmb{s}\|_{1} \\ \text { s.t. }  & \pmb{y}=\mathbf{A} s\end{array}
$$

这样压缩感知问题就可通过$L_1$范数最小化问题求解，例如上式可转化为LASSO的等价形式再通过近端梯度下降法求解，即使用基寻踪去噪（Basis Pursuit De-Noising）

**矩阵补全**（matrix completion）：

基于部分信息来恢复全部信息的技术在很多现实任务中有重要应用，如收集读者对部分书的评价根据读者的读书偏好来进行新书推荐，能否通过读者评价得到的数据当作部分信号，基于压缩感知的思想恢复出完整信号呢？

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191123145237565.png#pic_center)

矩阵补全技术可用于解决这个问题，其形式为

$$
\begin{array}{l}\underset{\mathbf{X}}{\min }  \operatorname{rank}(\mathbf{X}) \\ \text { s.t. } (\mathbf{X})_{i j}=(\mathbf{A})_{i j}, \quad(i, j) \in \Omega\end{array}
$$

其中，$\mathbf{X}$表示需恢复的稀疏信号，$rank(\mathbf{X})$表示其秩，$\mathbf{A}$是如上表的读者评分矩阵这样的已观测信号，$\Omega$是$\mathbf{A}$中非？元素的下标$(i,j)$的集合

上式也是一个NP难问题，注意到$rank(\mathbf{X})$在集合$\left\\{\mathbf{X} \in \mathbb{R}^{m \times n}:\|\mathbf{X}\|_{F}^{2} \leqslant 1\right\\}$的凸包是$\mathbf{X}$的核范数（nuclear norm）：

$$
\|\mathbf{X}\|_{*}=\sum_{j=1}^{\min \{m, n\}} \sigma_{j}(\mathbf{X})
$$

其中，$\sigma_{j}(\mathbf{X})$表示$\mathbf{X}$的奇异值，即矩阵的核范数为矩阵的奇异值之和，遇事可通过最小化矩阵核范数来近似求解上上式，即

$$
\begin{array}{cl}\underset{\mathbf{x}}{\min } & \|\mathbf{X}\|_{*} \\ {\text { s.t. }} & {(\mathbf{X})_{i j}=(\mathbf{A})_{i j}, \quad(i, j) \in \Omega}\end{array}
$$

这是一个凸优化问题，可通过半正定规划（Semi-Definite Programming, SDP）求解，理论研究表明，在满足一定条件时，若$\mathbf A$的秩为$r, n \ll m$，则只需观察到$O\left(m r \log ^{2} m\right)$个元素就能完美恢复出$\mathbf A$



