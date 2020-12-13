---
title: 西瓜书 | 第六章 支持向量机
author: 钟欣然
date: 2020-12-12 00:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

## 6.1 间隔与支持向量

#### 6.1.1 问题的直观描述



给定训练样本集$D=\\left\\{\left(\pmb{x_1}, y_{1}\right),\left(\boldsymbol{x_2}, y_{2}\right), \ldots,\left(\boldsymbol{x_m}, y_{m}\right)\\right\\}, y_{i} \in \\{-1,+1\\}$，分类学习最基本的思想就是找到一个划分超平面，将不同类别的样本分开。直观上看，应该找位于两类样本最中间的划分超平面，其对样本局部扰动的容忍性最好，换言之，这样的分类结果是最鲁棒的，泛化能力最强。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191108095748827.png#pic_center)

#### 6.1.2 问题的数学描述

在样本空间中，划分的超平面可用如下线性方程描述

$$
\pmb{w}^{\mathrm{T}} \pmb{x}+b=0
$$

其中$\pmb{w}=\left(w_{1} ; w_{2} ; \ldots ; w_{d}\right)$为法向量，决定了超平面的方向，b为偏移项，决定了超平面与原点之间的距离，因此我们可以将超平面记为$(\pmb w,b)$

样本空间内一点$\pmb x$到超平面的距离为

$$
r=\frac{\left|\pmb{w}^{\mathrm{T}} \pmb{x}+b\right|}{\|\pmb{w}\|}
$$

假设超平面能将训练样本分类正确，即若$y_i=+1$，则有$\pmb{w}^{\mathrm{T}} \pmb{x}_i+b>0$，若$y_i=-1$，则有$\pmb{w}^{\mathrm{T}} \pmb{x}_i+b<0$。令

$$
\left\{\begin{array}{ll}{\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b \geqslant+1,} & {y_{i}=+1} \\ {\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b \leqslant-1,} & {y_{i}=-1}\end{array}\right.
$$

距离超平面最近的几个点使得等号成立，称其为支持向量（support vector）。两个异类支持向量到超平面的距离之和为$\gamma=\frac{2}{\|\pmb{w}\|}$，称其为间隔（margin）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191108100901862.png#pic_center)

欲找到具有最大间隔（maximum margin）的划分超平面，则问题转化为

$$
\begin{array}{cl} \underset{\pmb{w}, b}{\max} & {\frac{2}{\|\pmb{w}\|}} \\ {\text { s.t. }} & {y_{i}\left(\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m}\end{array}
$$

进一步可写成

$$
\begin{array}{cl} \underset{\pmb{w}, b}{\max} & {\frac1{2}{\|\pmb{w}\|}^2} \\ {\text { s.t. }} & {y_{i}\left(\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m}\end{array}
$$

即为支持向量机（Support Vector Machine）的基本型

## 6.2 对偶问题
#### 6.2.1 对偶问题描述
对上式使用拉格朗日乘子法得到其对偶问题，拉格朗日函数为

$$
L(\pmb{w}, b, \pmb{\alpha})=\frac{1}{2}\|\pmb{w}\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-y_{i}\left(\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b\right)\right),\alpha_i\geq0
$$

令上式对$\pmb{w}, b$的偏导为0，得

$$
\begin{aligned} \pmb{w} &=\sum_{i=1}^{m} \alpha_{i} y_{i} \pmb{x}_{i} \\ 0 &=\sum_{i=1}^{m} \alpha_{i} y_{i} \end{aligned}
$$

带回上式，得到其对偶问题为

$$
\begin{aligned} \max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \pmb{x}_{i}^{\mathrm{T}} \pmb{x}_{j} \\ \text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\ & \alpha_{i} \geqslant 0, \quad i=1,2, \ldots, m \end{aligned}
$$

求解出$\alpha$后，即可得到模型

$$
\begin{aligned} f(\pmb{x}) &=\pmb{w}^{\mathrm{T}} \pmb{x}+b \\ &=\sum_{i=1}^{m} \alpha_{i} y_{i} \pmb{x}_{i}^{\mathrm{T}} \pmb{x}+b \end{aligned}
$$

上述过程需满足KKT条件

$$
\left\{\begin{array}{l}{\alpha_{i} \geqslant 0} \\ {y_{i} f\left(\pmb x_{i}\right)-1 \geqslant 0} \\ {\alpha_{i}\left(y_{i} f\left(\pmb x_{i}\right)-1\right)=0}\end{array}\right.
$$

对训练样本$(\pmb x_i,y_i)$，一定有$\alpha_i=0$或$y_if(\pmb x_i)=1$，即或者该约束不起作用，或者样本点落在最大间隔边界上，是一个支持向量。由此得到SVM的一个重要性质，训练完成后，大部分的训练样本都不需要保留，最终模型仅与支持向量有关。

#### 6.2.2 SMO求解
可使用二次规划算法来求解，但是该问题的规模正比于样本数，在实际任务中有很大开销。因此可以采用更多高效的方法，如	SMO（Sequential Minimal Optimization）。

**基本思路**

SMO的基本思路是先固定$\alpha_i$以外的所有参数，然后求$\alpha_i$上的极值，由于存在约束$\sum_{i=1}^{m} \alpha_{i} y_{i}=0$，若固定除$\alpha_i$外的所有参数，$\alpha_i$可直接导出。因此，可以在参数初始化之后，每次选择两个参数$\alpha_i,\alpha_j$，固定其他参数，求解获得更新后的$\alpha_i,\alpha_j$，直至收敛。

注意到只需选取的$\alpha_i,\alpha_j$有一个不满足KKT条件，目标函数就会在迭代后减小，且直观来看，KKT条件违背的程度越大，变量更新后目标函数的减幅越大。因此，SMO第一个变量取违背KKT条件程度最大的变量，第二个变量选取使目标函数减小最快的变量，但由于比较各变量对应的目标函数的减幅的复杂度过高，因此SMO采用了一个启发式：使选取的两变量对应的样本间的间隔最大。直观解释是这样的两个变量有很大差别，与两个相似的变量进行更新相比能给目标函数带来更大的变化。

**高效性**

SMO算法的高效性在于每次更新两个参数的过程非常高效。固定其他参数后，可以用

$$
\alpha_{i} y_{i}+\alpha_{j} y_{j}=c,\alpha_{i} \geqslant 0,\alpha_{j} \geqslant 0
$$

消去$\alpha_j$，得到一个关于$\alpha_i$的单变量二次规划问题，仅有的约束为$\alpha_i\geq 0$，有闭式解，不必调用数值优化算法即可高效地计算出更新后的变量。

**求解偏移项**

对于支持向量$(\pmb x_s,y_s)$有$y_sf(\pmb x_S)=1$，即

$$
y_{s}\left(\sum_{i \in S} \alpha_{i} y_{i} \pmb{x}_{i}^{\mathrm{T}} \pmb{x}_{s}+b\right)=1
$$

理论上可采用任意支持向量求解b，实际中往往采用更鲁棒的做法，使用所有支持向量的平均值

$$
b=\frac{1}{|S|} \sum_{s \in S}\left(y_{s}-\sum_{i \in S} \alpha_{i} y_{i} \pmb{x}_{i}^{\mathrm{T}} \pmb{x}_{s}\right)
$$

## 6.3 核函数

#### 6.3.1 核函数在SVM中的应用
某些问题在原始样本空间内可能不存在一个能正确划分两类样本的超平面，此时可以将样本从原始空间映射到一个更高位的特征空间，使得样本在这个特征空间内线性可分。如果原始空间是有限维，则一定存在一个高维特征空间式样本可分。

令$\phi(\pmb x)$表示$\pmb x$映射后的特征向量，则在特征空间中划分超平面对应的模型可表示为

$$
f(\pmb{x})=\pmb{w}^{\mathrm{T}} \phi(\pmb{x})+b
$$

原始问题为


$$
\begin{array}{cl}\underset{\pmb{w}, b}{\min} & {\frac{1}{2}\|\pmb{w}\|^{2}} \\ {\text { s.t. }} & {y_{i}\left(\pmb{w}^{\mathrm{T}} \phi\left(\pmb{x}_{i}\right)+b\right) \geqslant 1, \quad i=1,2, \ldots, m}\end{array}
$$

对偶问题为

$$
\begin{aligned} \max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \phi\left(\pmb x_{i}\right)^{\mathrm{T}} \phi\left(\pmb x_{j}\right)\\ \text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\ & \alpha_{i} \geqslant 0, \quad i=1,2, \ldots, m \end{aligned}
$$

$\phi\left(\pmb x_{i}\right)^{\mathrm{T}} \phi\left(\pmb x_{j}\right)$是$\pmb x_{i},\pmb x_{j}$映射到特征空间后的内积，为避免高维的特征空间计算苦难，可定义

$$
\kappa\left(\pmb{x}_{i}, \pmb{x}_{j}\right)=\left\langle\phi\left(\pmb{x}_{i}\right), \phi\left(\pmb{x}_{j}\right)\right\rangle=\phi\left(\pmb{x}_{i}\right)^{\mathrm{T}} \phi\left(\pmb{x}_{j}\right)
$$

对偶问题进一步转化为

$$
\begin{array}{cl}\underset{\alpha}{\max} & {\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \kappa\left(\pmb{x}_{i}, \pmb{x}_{j}\right)} \\ {\text { s.t. }} & {\sum_{i=1}^{m} \alpha_{i} y_{i}=0} \\ {} & {\alpha_{i} \geqslant 0, \quad i=1,2, \ldots, m}\end{array}
$$


求解后即可得到

$$
\begin{aligned} f(\pmb{x}) &=\pmb{w}^{\mathrm{T}} \phi(\pmb{x})+b \\ &=\sum_{i=1}^{m} \alpha_{i} y_{i} \phi\left(\pmb{x}_{i}\right)^{\mathrm{T}} \phi(\pmb{x})+b \\ &=\sum_{i=1}^{m} \alpha_{i} y_{i} \kappa\left(\pmb{x}, \pmb{x}_{i}\right)+b \end{aligned}
$$

其中，$\kappa(\cdot,\cdot)$即为核函数。上式显示出模型最优解可通过训练样本的核函数展开，这一展式成为支持向量展式（support vector expansion）

#### 6.3.2 核函数的性质
**核函数的充要条件**

令$\chi$为输入空间，$\kappa(\cdot,\cdot)$是定义在$\chi\times \chi$上的对称函数，则$\kappa$是核函数当且仅当对于任意数据$D=\{\pmb x_1,\pmb x_2,\dots ,\pmb x_m\}$，核矩阵（kernel matrix）$\mathbf{K}$总是半正定的

$$
\mathbf{K}=\left[\begin{array}{ccccc}{\kappa\left(\pmb{x}_{1}, \pmb{x}_{1}\right)} & {\cdots} & {\kappa\left(\pmb{x}_{1}, \pmb{x}_{j}\right)} & {\cdots} & {\kappa\left(\pmb{x}_{1}, \pmb{x}_{m}\right)} \\ {\vdots} & {\ddots} & {\vdots} & {\ddots} & {\vdots} \\ {\kappa\left(\pmb{x}_{i}, \pmb{x}_{1}\right)} & {\cdots} & {\kappa\left(\pmb{x}_{i}, \pmb{x}_{j}\right)} & {\cdots} & {\kappa\left(\pmb{x}_{i}, \pmb{x}_{m}\right)} \\ {\vdots} & {\ddots} & {\vdots} & {\ddots} & {\vdots} \\ {\kappa\left(\pmb{x}_{m}, \pmb{x}_{1}\right)} & {\cdots} & {\kappa\left(\pmb{x}_{m}, \pmb{x}_{j}\right)} & {\cdots} & {\kappa\left(\pmb{x}_{m}, \pmb{x}_{m}\right)}\end{array}\right]
$$

上述定理表明只要一个对称函数所对应的核矩阵半正定，它就能作为核函数使用。事实上，对每一个半正定核矩阵，总能找到一个与之对应的映射$\phi$，换言之，任何一个核函数都隐式地定义了一个特征空间，称为再生核希尔伯特空间（Reproducing Kernel Hilbert Space, RKHS）

 - [ ] 如何理解再生核希尔伯特空间

**常用的核函数** ：

核函数选择是支持向量机的最大变数，若核函数选择不合适，则意味着将样本映射到了一个不合适的特征空间，很可能导致性能不佳。

几种常用的核函数：

|名称|表达式|参数|
|-|-|-|
|线性核|$\kappa\left(x_{i}, x_{j}\right)=x_{i}^{T} x_{j}$| |
|多项式核|$\kappa\left(x_{i}, x_{j}\right)=(x_{i}^{T} x_{j})^d$|$d\geq1$为多项式的次数|
|高斯核|$\kappa\left(x_{i}, x_{j}\right)=\exp \left(-\frac{\left\|x_{i}-x_{j}\right\|^{2}}{2 \sigma^{2}}\right)$|$\sigma>0$为高斯核的带宽（width）|
|拉普拉斯核|$\kappa\left(x_{i}, x_{j}\right)=\exp \left(-\frac{\left\|x_{i}-x_{j}\right\|^{2}}{ \sigma}\right)$|$\sigma>0$|
|Sigmoid核|$\kappa\left(x_{i}, x_{j}\right)=\tanh \left(\beta x_{i}^{T} x_{j}+\theta\right)$|tanh为双曲正切函数，$\beta>0,\theta<0$

一些基本的经验：文本数据常采用线性核，情况不明时可先尝试高斯核

**核函数的组合**
- 线性组合：若$\kappa_1,\kappa_2$为核函数，$\forall \gamma_1>0,\gamma_2>0$，$\gamma_1\kappa_1+\gamma_2\kappa_2$也是核函数
- 直积：若$\kappa_1,\kappa_2$为核函数，$\kappa_1 \otimes \kappa_2(\pmb x, \pmb z)=\kappa_1(\pmb x, \pmb z)_{\kappa_2}(\pmb x, \pmb z)$也是核函数
- 变换：若$\kappa_1$为核函数，$\forall g(x)$，$\kappa(\pmb{x}, \pmb{z})=g(\pmb x) \kappa_{1}(\pmb x, \pmb z) g(\pmb{z})$也是核函数

## 6.4 软间隔与正则化

#### 6.4.1 软间隔及其求解过程

**软间隔**：

现实任务中往往很难确定合适的核函数使得训练样本在特征空间中线性可分，退一步说，即使恰好找到了某个核函数使训练集在特征空间中线性可分，也很难断定这个貌似线性可分的结果不是过拟合造成的。

解决办法是允许支持向量机在一些样本上出错，称为软间隔（soft margin）；与之对应的硬间隔（hard margin）要求所有样本满足约束（划分正确）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191109143359430.png#pic_center)

此时，优化目标为在最大化间隔的同时，不满足约束的样本尽可能少，目标函数为

$$
\min _{\pmb{w}, b} \frac{1}{2}\|\pmb{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left(\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b\right)-1\right)
$$

其中$C>0$是一个常数，$\ell_{0 / 1}$是0/1损失函数

$$
\ell_{0 / 1}(z)=\left\{\begin{array}{ll}{1,} & {\text { if } z<0} \\ {0,} & {\text { otherwise }}\end{array}\right.
$$

C为无穷大时，所有样本都必须满足约束，C取有限值时，允许一些样本不满足约束

**损失函数**：

$\ell_{0 / 1}$非凸、非连续，数学性质不好，人们常用其它损失函数替代，称为替代损失（surrogate loss）
- hinge损失：$\ell_{hinge}(z)=\max(0,1-z)$
- 指数损失（exponential loss）：$\ell_{exp}(z)=exp(-z)$
- 对率损失（logistic loss）：$\ell_{log}(z)=log(1+exp(-z))$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191109171659685.png#pic_center)

**hinge损失求解过程**：

目标函数为：

$$
\min _{\pmb{w}, \pmb{b}} \frac{1}{2}\|\pmb{w}\|^{2}+C \sum_{i=1}^{m} \max \left(0,1-y_{i}\left(\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b\right)\right)
$$

引入松弛变量（slack variables）$\xi_i\geq0$，上式可重写为

$$
\begin{array}{cl}\underset{\pmb{w}, b}{\min} & \frac{1}{2}\|\pmb{w}\|^{2}+C \sum_{i=1}^{m} \xi_i \\ {\text { s.t. }} & {y_{i}\left(\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b\right) \geqslant 1-\xi_i}\\&\xi_i\geq0,i=1,2, \ldots, m\end{array}
$$

这就是软间隔支持向量机，每个样本对应一个松弛变量，用以表征该样本不满足约束的程度。通过引入拉格朗日函数

$$
\begin{aligned} L(\pmb{w}, b, \pmb{\alpha}, \pmb{\xi}, \pmb{\mu})=& \frac{1}{2}\|\pmb{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i} \\ &+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\pmb{w}^{\mathrm{T}} \pmb{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i} \end{aligned}
$$

对$\pmb w,b,\xi$求偏导为零

$$
\begin{aligned} \pmb{w} &=\sum_{i=1}^{m} \alpha_{i} y_{i} \pmb{x}_{i} \\ 0 &=\sum_{i=1}^{m} \alpha_{i} y_{i} \\ C &=\alpha_{i}+\mu_{i} \end{aligned}
$$

代入可得其对偶问题为

$$
\begin{aligned} \max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \pmb{x}_{i}^{\mathrm{T}} \pmb{x}_{j} \\ \text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y_{i}=0 \\ & 0\leq \alpha_{i} \leq C, \quad i=1,2, \ldots, m \end{aligned}
$$

不难发现，软间隔与硬间隔的对偶问题唯一的差别在于对乘子的约束不同，前者是$0\leq \alpha_i\leq C$，后者是$1\leq \alpha$，类似地，对软间隔支持向量机KKT条件要求

$$
\left\{\begin{array}{l}{\alpha_{i} \geqslant 0, \quad \mu_{i} \geqslant 0} \\ {y_{i} f\left(\pmb{x}_{i}\right)-1+\xi_{i} \geqslant 0} \\ {\alpha_{i}\left(y_{i} f\left(\pmb{x}_{i}\right)-1+\xi_{i}\right)=0} \\ {\xi_{i} \geqslant 0, \mu_{i} \xi_{i}=0}\end{array}\right.
$$

对任意训练样本$(\pmb x_i,y_i)$，有以下两种情况：
- $\alpha_i=0$，此时该约束不起作用
- $\alpha_i>0,y_i \left(\pmb{x}_i\right)=1-\xi_i$，此时该样本为支持向量：
	- $\alpha_i<C$，则$\mu_i>0,\xi_i=0$，该样本在最大间隔边界上
	- $\alpha_i=C$，则$\mu_i=0$：
		- $\xi_i\leq 1$，样本落在最大间隔内部
		- $\xi>1$，样本被错误分类

软间隔支持向量机的最终模型仅与支持向量有关，即通过hinge损失函数仍保持了稀疏性

**对率损失**：

如果使用对率损失函数来替代0/1损失函数，则几乎得到了对率回归模型，实际上支持向量机与对率回归的目标相近，通常情形下性能也相当
- 对率回归的输出具有自然的概率意义
- 对率回归能直接用于多分类任务
- 对率回归不能导出类似支持向量的概念，因此其解依赖于更多的训练样本，其预测开销更大



#### 6.4.2 正则化
无论采用何种损失函数，都有一个共性：优化目标中第一项用来描述划分超平面的间隔大小，第二项用来描述训练集上的误差

$$
\min _{f} \Omega(f)+C \sum_{i=1}^{m} \ell\left(f\left(\pmb{x}_{i}\right), y_{i}\right)
$$

其中，$\Omega(f)$为结构风险（structural risk），用于描述模型f的某种性质；第二项为经验风险（empirical risk），用于描述模型与训练数据的契合程度；C用于对二者进行折中

从经验风险最小化的角度来看
- $\Omega(f)$表述了我们希望获得具有何种性质的模型（如希望获得复杂度小的模型），这为引入领域知识和用户意图提供了途径
- 该信息有助于消减假设空间，从而降低了最小化训练误差的过拟合风险

从这个角度来说，上式称为正则化（regularization）问题，$\Omega(f)$称为正则化项，C则称为正则化常数，$L_p$范数（norm）是常用的正则化项，其中$L_2$范数$\Vert \pmb w\Vert_2$倾向于$\pmb w$的分量取值尽量均衡，即非零分量个数尽量稠密，$L_0$范数$\Vert \pmb w\Vert_0$和$L_1$范数$\Vert \pmb w\Vert_1$则倾向于$\pmb w$的分量尽量稀疏，即非零分量个数尽量少

## 6.5 支持向量回归

#### 6.5.1 思想与求解过程
支持向量回归（support vector regression, SVR）希望学得一个形如$\pmb{w}^{\mathrm{T}} \pmb{x}+b=0$的回归模型，使得$f(\pmb x)$与y尽可能接近，与传统回归不同的是，我们能容忍$f(\pmb x)$与y之间最多有$\epsilon$的偏差，即仅当$f(\pmb x)$与y之间的差别超过$\epsilon$时才计算损失，相当于以$f(\pmb x)$为中心构建了一个宽度为$2\epsilon$的间隔带，训练样本落入此间隔带时认为是被预测正确的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191109171618476.png#pic_center)

因此，SVR可形式化为

$$
\min _{\pmb{w}, b} \frac{1}{2}\|\pmb{w}\|^{2}+C \sum_{i=1}^{m} \ell_{c}\left(f\left(\pmb{x}_{i}\right)-y_{i}\right)
$$

其中，C为正则化常数，$\ell_\epsilon$为$\epsilon$-不敏感损失（$\epsilon$-insensitive loss）函数

$$
\ell_{\epsilon}(z)=\left\{\begin{array}{ll}{0,} & {\text { if }|z| \leqslant \epsilon} \\ {|z|-\epsilon,} & {\text { otherwise }}\end{array}\right.
$$

![在这里插入图片描述](https://img-blog.csdnimg.cn/201911091729540.png#pic_center)

引入松弛变量$\xi_i,\hat \xi_i$，可将上式重写为

$$
\begin{array}{ll}\underset{ {\pmb{w}, b, \pmb{\xi}_{i}, \hat{\pmb{\xi}}_{i}}}{\min} & \frac{1}{2}\|\pmb{w}\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right)\\{\text { s.t. }} & {f\left(\pmb{x}_{i}\right)-y_{i} \leqslant \epsilon+\xi_{i}} \\ {} & {y_{i}-f\left(\pmb{x}_{i}\right) \leqslant \epsilon+\hat{\xi}_{i}} \\ {} & {\xi_{i} \geqslant 0, \hat{\xi}_{i} \geqslant 0, i=1,2, \ldots, m}\end{array}
$$

拉格朗日函数为

$$
\begin{array}{l}{L(\pmb{w}, b, \pmb{\alpha}, \hat{\pmb{\alpha}}, \pmb{\xi}, \hat{\pmb{\xi}}, \pmb{\mu}, \hat{\pmb{\mu}})} \\ {=\frac{1}{2}\|\pmb{w}\|^{2}+C \sum_{i=1}^{m}\left(\xi_{i}+\hat{\xi}_{i}\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}-\sum_{i=1}^{m} \hat{\mu}_{i} \hat{\xi}_{i}} \\ {+\sum_{i=1}^{m} \alpha_{i}\left(f\left(\pmb{x}_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)+\sum_{i=1}^{m} \hat{\alpha}_{i}\left(y_{i}-f\left(\pmb{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)}\end{array}
$$

对$\pmb{w}, b,\xi_i,\hat\xi_i$求偏导为零得

$$
\begin{aligned} \pmb{w} &=\sum_{i=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \pmb{x}_{i} \\ 0 &=\sum_{i=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \\ C &=\alpha_{i}+\mu_{i} \\ C &=\hat{\alpha}_{i}+\hat{\mu}_{i} \end{aligned}
$$

代入得对偶问题为

$$
\begin{aligned} \max _{\pmb{\alpha}, \hat{\pmb{x}}} & \sum_{i=1}^{m} y_{i}\left(\hat{\alpha}_{i}-\alpha_{i}\right)-\epsilon\left(\hat{\alpha}_{i}+\alpha_{i}\right) \\ &-\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right)\left(\hat{\alpha}_{j}-\alpha_{j}\right) \pmb{x}_{i}^{\mathrm{T}} \pmb{x}_{j} \\ \text { s.t. } & \sum_{i=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right)=0 \\ & 0 \leqslant \alpha_{i}, \hat{\alpha}_{i} \leqslant C \end{aligned}
$$

上述过程需满足KKT条件

$$
\left\{\begin{array}{l}{\alpha_{i}\left(f\left(\pmb{x}_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)=0} \\ {\hat{\alpha}_{i}\left(y_{i}-f\left(\pmb{x}_{i}\right)-\epsilon-\hat{\xi}_{i}\right)=0} \\ {\alpha_{i} \hat{\alpha}_{i}=0, \xi_{i} \hat{\xi}_{i}=0} \\ {\left(C-\alpha_{i}\right) \xi_{i}=0,\left(C-\hat{\alpha}_{i}\right) \hat{\xi}_{i}=0}\end{array}\right.
$$



可以看出，当且仅当$f\left(\pmb{x_i}\right)-y_i-\epsilon-\xi_i=0$时$\alpha_i$可以取非零值，当且仅当$y_i-f\left(\pmb{x_i}\right)-\epsilon-\hat{\xi_i}=0$时$\hat{\alpha_i}$可以取非零值，且二者不能同时取非零值



因此，SVR的解形如
$$
f(\pmb{x})=\sum_{i=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \pmb{x}_{i}^{\mathrm{T}} \pmb{x}+b
$$

能使上式中的$\hat{\alpha_i}-\alpha_i\neq0$的样本即为SVR的支持向量，它们必落在$\epsilon$-间隔带之外。因此，SVR的支持向量仅是训练样本的一部分，即其解仍具有稀疏性

对于满足$0<\alpha_i<C$的样本，有$\xi_i=0$，进而有$f\left(\pmb{x_i}\right)=y_i+\epsilon$，故

$$
b=y_{i}+\epsilon-\sum_{i=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \pmb{x}_{i}^{\mathrm{T}} \pmb{x}
$$

更鲁棒性的做法是选取多个满足$0<\alpha_i<C$的样本求解b后取平均值

#### 6.5.2 特征映射形式

若考虑特征映射形式，则有

$$
\pmb{w}=\sum_{i=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \phi\left(\pmb{x}_{i}\right)
$$

$$
f(\pmb{x})=\sum_{i=1}^{m}\left(\hat{\alpha}_{i}-\alpha_{i}\right) \kappa\left(\pmb{x}, \pmb{x}_{i}\right)+b
$$

## 6.6 核方法
若不考虑偏移项b，则无论SVM还是SVR，学得的模型总能表示成核函数$\kappa(\cdot,\cdot)$的线性组合，表示定理给了更一般的结论

#### 6.5.1 表示定理：
令$\mathbb{H}$为核函数$\kappa$对应的再生核希尔伯特空间，$\|h\|_{\mathbb{H}}$表示$\mathbb{H}$空间中关于h的范数，对于任意单调递增函数$\Omega:[0, \infty] \mapsto \mathbb{R}$和任意非负损失函数$\ell: \mathbb{R}^{m} \mapsto[0, \infty]$，优化问题

$$
\min _{h \in \mathbb{H}} F(h)=\Omega\left(\|h\|_{\mathbb{H}}\right)+\ell\left(h\left(\pmb{x}_{1}\right), h\left(\pmb{x}_{2}\right), \ldots, h\left(\pmb{x}_{m}\right)\right)
$$

的解总可写为

$$
h^{*}(\pmb{x})=\sum_{i=1}^{m} \alpha_{i} \kappa\left(\pmb{x}, \pmb{x}_{i}\right)
$$

表示定理对损失函数没有限制，对正则化项$\Omega$仅要求单调递增，甚至不要求是凸函数，意味着对于一般的损失函数和正则化项，优化问题的解都可以表示成核函数的线性组合

人们发展出一系列基于核函数的学习方法，统称为核方法（kernel methods），最常见的是通过核化（即引入核函数）来将线性学习器拓展为非线性学习器，如核线性判别分析（Kernelized Linear Discriminant Analysis, KLDA）

#### 6.5.2 核线性判别分析
我们先假设可通过某种映射$\phi:\chi\rightarrow \mathbb{F}$将样本映射到一个特征空间$\mathbb{F}$，然后在$\mathbb{F}$中执行线性判别分析，以求得

$$
h(\pmb{x})=\pmb{w}^{\mathrm{T}} \pmb{\phi}(\pmb{x})
$$

学习目标是

$$
\max _{\pmb{w}} J(\pmb{w})=\frac{\pmb{w}^{\mathrm{T}} \mathbf{S}_{b}^{\phi} \pmb{w}}{\pmb{w}^{\mathrm{T}} \mathbf{S}_{w}^{\phi} \pmb{w}}
$$

其中，$\mathbf{S_b}^{\phi},\mathbf{S_w}^{\phi}$分别为训练样本在特征空间$\mathbb{F}$中的类间散度矩阵和类内散度矩阵，令$X_i$表示第$i\in\{0,1\}$类样本的集合，其样本数为$m_i$，第$i$类样本在特征空间$\mathbb{F}$中的均值为

$$
\pmb{\mu}_{i}^{\phi}=\frac{1}{m_{i}} \sum_{\pmb{x} \in X_{i}} \phi(\pmb{x})
$$

两个散度矩阵分别为

$$
\begin{aligned} \mathbf{S}_{b}^{\phi} &=\left(\pmb{\mu}_{1}^{\phi}-\pmb{\mu}_{0}^{\phi}\right)\left(\pmb{\mu}_{1}^{\phi}-\pmb{\mu}_{0}^{\phi}\right)^{\mathrm{T}} \\ \mathbf{S}_{w}^{\phi} &=\sum_{i=0}^{1} \sum_{\pmb{x} \in X_{i}}\left(\phi(\pmb{x})-\pmb{\mu}_{i}^{\phi}\right)\left(\phi(\pmb{x})-\pmb{\mu}_{i}^{\phi}\right)^{\mathrm{T}} \end{aligned}
$$

使用核函数$\kappa\left(\pmb{x}, \pmb{x_i}\right)=\phi\left(\pmb{x_i}\right)^{\mathrm{T}} \phi(\pmb{x})$来隐式地表达这个映射和特征空间$\mathbb{F}$，把$J(\pmb{w})$作为损失函数$\ell$，再令$\Omega \equiv 0$，由表示定理

$$
h(\pmb{x})=\sum_{i=1}^{m} \alpha_{i} \kappa\left(\pmb{x}, \pmb{x}_{i}\right)
$$

则有

$$
\pmb{w}=\sum_{i=1}^{m} \alpha_{i} \phi\left(\pmb{x}_{i}\right)
$$

令$\mathbf{K} \in \mathbb{R}^{m \times m}$为核函数$\gamma$所对应的核矩阵，$(\mathbf{K_{i j}})=\kappa\left(\pmb{x_i}, \pmb{x_j}\right)$，令$\mathbf{1}_i \in\\{1,0\\}^{m \times 1}$为令i类样本的指示向量，即$\mathbf{1}_i$的第j个分量为1当且仅当$\pmb x_j\in X_i$，否则为0，再令

$$
\begin{aligned} \hat{\pmb{\mu}}_{0} &=\frac{1}{m_{0}} \mathbf{K} \mathbf{1}_{0} \\ \hat{\pmb{\mu}}_{1} &=\frac{1}{m_{1}} \mathbf{K} \mathbf{1}_{1} \\ \mathbf{M} &=\left(\hat{\pmb{\mu}}_{0}-\hat{\pmb{\mu}}_{1}\right)\left(\hat{\pmb{\mu}}_{0}-\hat{\pmb{\mu}}_{1}\right)^{\mathrm{T}} \\ \mathbf{N} &=\mathbf{K} \mathbf{K}^{\mathrm{T}}-\sum_{i=0}^{1} m_{i} \hat{\pmb{\mu}}_{i} \hat{\pmb{\mu}}_{i}^{\mathrm{T}} \end{aligned}
$$

则优化目标等价于



$$
\max _{\pmb{\alpha}} J(\pmb{\alpha})=\frac{\pmb{\alpha}^{\mathrm{T}} \mathbf{M} \pmb{\alpha}}{\pmb{\alpha}^{\mathrm{T}} \mathbf{N} \pmb{\alpha}}
$$


显然，使用线性判别分析方法即可得到$\pmb \alpha$，进而可得到投影函数$h(\pmb x)$