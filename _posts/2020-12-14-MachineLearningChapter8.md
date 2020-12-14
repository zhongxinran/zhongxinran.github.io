---
title: 西瓜书 | 第八章 集成学习
author: 钟欣然
date: 2020-12-13 02:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200608163946899.png)
<center>集成学习示意图（参照《Python 机器学习：原理与实践》整理）</center><br>


# 1. 个体与集成
集成学习（ensemble learning），也称为多分类器系统（multi-classifier system）、基于委员会的学习（committee-based learning），通过构建并结合多个学习器来完成学习任务

## 集成的过程

集成先产生一组个体学习器（individual learner），再用某种策略将他们结合起来，个体学习器由一个现有的学习算法从训练数据产生

- 同质的（homogeneous）：集成中只包含同种类型的个体学习器，同质集成中的个体学习器称为基学习器（base learner），相应的学习算法称为基学习算法（base learning algorithm）
- 异质的（heterogenous）：集成中包含不同类型的学习器，异质集成中的个体学习器称为组建学习器或直接称为个体学习器

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191113173008183.png#pic_center)

## 集成的效果

**集成学习常可获得比单一学习器显著优越的泛化性能**，这对弱学习器（weak learner）尤为明显，因此集成学习的很多理论研究都是针对弱学习器进行的，而基学习器又是直接称为弱学习器。但在实践中人们往往会使用比较强的学习器

如何获得比最好的单一学习器更好的性能呢？以下图二分类问题为例

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191113173340787.png#pic_center)

集成学习的结果通过投票法（voting）产生，即少数服从多数。上图例子显示，要获得好的学习器，个体学习器应好而不同，即个体学习器要有一定的**准确性和多样性**

**集成的准确率与个体学习器数目的关系**

考虑二分类问题$y \in\\{-1,+1\\}$和真实函数$f$，假设基分类学习器的错误率为$\epsilon$，即对每个基分类学习器$h_i$，有

$$
P\left(h_{i}(\pmb{x}) \neq f(\pmb{x})\right)=\epsilon
$$

假设集成通过简单投票法结合T个基分类器，若有超过半数的基分类器分类正确，则集成分类就正确

$$
H(\pmb{x})=\operatorname{sign}\left(\sum_{i=1}^{T} h_{i}(\pmb{x})\right)
$$

假设基分类器的错误率相互独立，则由Hoeffding不等式可知，集成的错误率为

$$
\begin{aligned} P(H(\pmb{x}) \neq f(\pmb{x})) &=\sum_{k=0}^{\lfloor T / 2\rfloor}\left(\begin{array}{c}{T} \\ {k}\end{array}\right)(1-\epsilon)^{k} \epsilon^{T-k} \\ & \leqslant \exp \left(-\frac{1}{2} T(1-2 \epsilon)^{2}\right) \end{aligned}
$$

随着集成中个体分类器数目T的增大，集成的错误率将指数级下降，最终趋向于零

但必须注意到，基学习器的误差不可能相互独立，因为他们都是为解决同一个问题训练出来的；事实上，**个体学习器的准确性和多样性本身就存在冲突**，一般的，准确性很高后，增加多样性就需牺牲准确性，因此，如何产生好而不同的个体学习器是集成学习的核心

## 集成学习的分类

依据个体学习器的生成方式分成两类：

- 个体学习器之间存在强依赖关系，必须串行生成的序列化方法，代表是Boosting（8.2）
- 个体学习器之间不存在强依赖关系，可同时生成的并行化方法，代表室Bagging和随机森林（Random Forest）（8.3）

# 2. Boosting

Boosting是一族可将弱学习器提升为强学习器的算法，这族算法的工作机制类似：先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续收到更多关注，然后基于调整后的样本分布来训练下一个基学习器，如此重复进行，直至生成T个学习器数目，最终将这T个基学习器进行加权结合。

## AdaBoost
AdaBoost是Boosting的著名代表，适用于二分类问题，令$y_{i} \in\\{-1,+1\\}$，$f$是真实函数，其过程如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115091026386.png#pic_center)

下面我们以加性模型（additive model）为例进行推导，加性模型即以基学习器的线性组合来最小化指数损失函数（exponential loss function）

$$
H(\pmb{x})=\sum_{t=1}^{T} \alpha_{t} h_{t}(\pmb{x})
$$

$$
\ell_{\exp }(H | \mathcal{D})=\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H(\pmb{x})}\right]
$$

**指数损失函数最小化则分类错误率最小化**

若$H(\pmb{x})$能令指数损失函数最小化，则令指数损失函数对$H(\pmb{x})$的偏导为零，可得

$$
\frac{\partial \ell_{\exp }(H | \mathcal{D})}{\partial H(\pmb{x})}=-e^{-H(\pmb{x})} P(f(\pmb{x})=1 | \pmb{x})+e^{H(\pmb{x})} P(f(\pmb{x})=-1 | \pmb{x})=0
$$

$$
H(\pmb{x})=\frac{1}{2} \ln \frac{P(f(x)=1 | \pmb{x})}{P(f(x)=-1 | \pmb{x})}
$$

因此，有

$$
\begin{aligned} \operatorname{sign}(H(\pmb{x})) &=\operatorname{sign}\left(\frac{1}{2} \ln \frac{P(f(x)=1 | \pmb{x})}{P(f(x)=-1 | \pmb{x})}\right) \\ &=\left\{\begin{array}{ll}{1,} & {P(f(x)=1 | \pmb{x})>P(f(x)=-1 | \pmb{x})} \\ {-1,} & {P(f(x)=1 | \pmb{x})<P(f(x)=-1 | \pmb{x})}\end{array}\right. \\&\;{=\underset{y \in\{-1,1\}}{\arg \max } P(f(x)=y | \pmb{x})}\end{aligned}
$$

即$\operatorname{sign}(H(\pmb{x}))$达到了贝叶斯最优错误率，这说明指数损失函数是分类任务原本0/1损失函数的一致的（consistent）替代损失函数，由于这个替代函数有更好的数学性质，如连续可微，因此常用它替代0/1损失函数作为优化目标

**AdaBoost的迭代公式**

- **基分类器权重**$\alpha_t$：第一个基分类器$h_1$是通过直接将基学习算法用于初始数据分布而得，此后迭代地生成$h_t$和$\alpha_t$，当基分类器基于分布$\mathcal{D}_{t}$产生后，该基分类器的权重$\alpha_t$应使得$\alpha_th_t$最小化指数损失函数

$$
\begin{aligned} \ell_{\mathrm{exp}}\left(\alpha_{t} h_{t} | \mathcal{D}_{t}\right) &=\mathbb{E}_{\pmb{x} \sim \mathcal{D}_{t}}\left[e^{-f(\pmb{x}) \alpha_{t} h_{t}(\pmb{x})}\right] \\ &=\mathbb{E}_{\pmb{x} \sim \mathcal{D}_{t}}\left[e^{-\alpha_{t}} \mathbb{I}\left(f(\pmb{x})=h_{t}(\pmb{x})\right)+e^{\alpha_{t}} \mathbb{I}\left(f(\pmb{x}) \neq h_{t}(\pmb{x})\right)\right] \\ &=e^{-\alpha_{t}} P_{x \sim \mathcal{D}_{t}}\left(f(\pmb{x})=h_{t}(\pmb{x})\right)+e^{\alpha_{t}} P_{x \sim \mathcal{D}_{t}}\left(f(\pmb{x}) \neq h_{t}(\pmb{x})\right) \\ &=e^{-\alpha_{t}}\left(1-\epsilon_{t}\right)+e^{\alpha_{t}} \epsilon_{t} \end{aligned}
$$

其中$\epsilon\_{t}=P\_{\pmb{x} \sim \mathcal{D}\_{t}}\left(h\_{t}(\pmb{x}) \neq f(\pmb{x})\right)$，令指数损失函数的导数为零，得

$$
\frac{\partial \ell_{\exp }\left(\alpha_{t} h_{t} | \mathcal{D}_{t}\right)}{\partial \alpha_{t}}=-e^{-\alpha_{t}}\left(1-\epsilon_{t}\right)+e^{\alpha_{t}} \epsilon_{t}=0
$$

$$
\alpha_{t}=\frac{1}{2} \ln \left(\frac{1-\epsilon_{t}}{\epsilon_{t}}\right)
$$

- **基分类器**$h_t$：算法在获得$H_{t-1}$之后将样本分布进行调整，使得下一轮的基学习器$h_t$能纠正$H_{t-1}$的一些错误，理想情形是纠正$H_{t-1}$的全部错误，即最小化

$$
\begin{aligned} \ell_{\exp }\left(H_{t-1}+h_{t} | \mathcal{D}\right) &=\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x})\left(H_{t-1}(\pmb{x})+h_{t}(\pmb{x})\right)}\right] \\ &=\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t-1}(\pmb{x})} e^{-f(\pmb{x}) h_{t}(\pmb{x})}\right] \end{aligned}
$$

问题：此处的$h_t$是否应为$\alpha_th_t？$

注意到$f^{2}(\pmb{x})=h_{t}^{2}(\pmb{x})=1$，上式可使用$e^{-f(\pmb{x}) h_{t}(\pmb{x})}$的泰勒展式近似为

$$
\begin{aligned} \ell_{\exp }\left(H_{t-1}+h_{t} | \mathcal{D}\right) & \simeq \mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}\left(1-f(\pmb{x}) h_{t}(\pmb{x})+\frac{f^{2}(\pmb{x}) h_{t}^{2}(\pmb{x})}{2}\right)\right]\\&=\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}\left(1-f(\pmb{x}) h_{t}(\pmb{x})+\frac{1}{2}\right)\right] \end{aligned}
$$

于是，理想的基学习器为

$$
\begin{aligned}h_{t}(\pmb{x})&=\underset{h}{\arg \min } \ell_{\exp }\left(H_{t-1}+h | \mathcal{D}\right)\\&{=\underset{h}{\arg \min } \mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}\left(1-f(\pmb{x}) h(\pmb{x})+\frac{1}{2}\right)\right]} \\ &{=\underset{h}{\arg \max } \mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t-1}(\pmb{x})} f(\pmb{x}) h(\pmb{x})\right]} \\ &{=\arg \max _{h} \mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[\frac{e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}}{\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}\right]} f(\pmb{x}) h(\pmb{x})\right]}\end{aligned}
$$

注意到$\mathbb{E}_{\pmb{x} \sim \mathcal{D}}[e^{f(\pmb{x})H\_{t-1}(\pmb{x})}]$是一个常数，令$\mathcal{D}\_{t}$表示一个分布



$$
\mathcal{D}_{t}(\pmb{x})=\frac{\mathcal{D}(\pmb{x}) e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}}{\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}\right]}
$$

这等价于令

$$
\begin{aligned} h_{t}(\pmb{x}) &=\underset{h}{\arg \max } \mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[\frac{e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}}{\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t-1}(\pmb{x})}\right]} f(\pmb{x}) h(\pmb{x})\right] \\ &=\underset{h}{\arg \max } \mathbb{E}_{\pmb{x} \sim \mathcal{D}_{t}}[f(\pmb{x}) h(\pmb{x})] \end{aligned}
$$

由$f(\pmb{x}), h(\pmb{x}) \in\\{-1,+1\\}$，有

$$
f(\pmb{x}) h(\pmb{x})=1-2 \mathbb{I}(f(\pmb{x}) \neq h(\pmb{x}))
$$

则理想的基学习器为

$$
h_{t}(\pmb{x})=\underset{h}{\arg \min } \mathbb{E}_{\pmb{x} \sim \mathcal{D}_{t}}[\mathbb{I}(f(\pmb{x}) \neq h(\pmb{x}))]
$$

由此可见，理想的$h\_t$在$\mathcal{D}\_{t}$下最小化分类误差，因此，弱分类器将基于分布$\mathcal{D}\_{t}$来训练，且针对$\mathcal{D}\_{t}$的训练误差应小于0.5，这在一定程度上类似残差逼近的思想

- **分布**$\mathcal{D}\_{t}$：考虑到$\mathcal{D}\_{t}$和$\mathcal{D}\_{t+1}$的关系，有

$$
\begin{aligned} \mathcal{D}_{t+1}(\pmb{x}) &=\frac{\mathcal{D}(\pmb{x}) e^{-f(\pmb{x}) H_{t}(\pmb{x})}}{\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t}(\pmb{x})}\right]} \\ &=\frac{\mathcal{D}(\pmb{x}) e^{-f(\pmb{x}) H_{t-1}(\pmb{x})} e^{-f(\pmb{x}) \alpha_{t} h_{t}(\pmb{x})}}{\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[e^{-f(\pmb{x}) H_{t}(\pmb{x})}\right]} \\ &=\mathcal{D}_{t}(\pmb{x}) \cdot e^{-f(\pmb{x}) \alpha_{t} h_{t}(\pmb{x}) {\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[\exp\left(-f(\pmb{x}) H_{t-1}(\pmb{x})\right)\right]}/{\mathbb{E}_{\pmb{x} \sim \mathcal{D}}\left[\exp\left(-f(\pmb{x}) H_{t}(\pmb{x})\right)\right]}} \end{aligned}
$$

## Boosting总结

Boosting算法要求基学习器能对特定的数据分布进行学习

- 重赋权法（re-weighting）：在训练过程的每一轮中，根据样本分布为每个训练样本重新赋予一个权重
- 重采样法（re-sampling）：对无法接受带权样本的基学习算法，在每一轮学习中，根据样本分布对训练集重新进行采样，再用重采样而得的样本集对基学习器进行训练

二者没有明显的优劣区别，但在训练的每一轮都要检查当前生成的基学习器是否满足基本条件（比随机猜测好），一旦条件不满足就抛弃当前基学习器，停止学习过程，此时可能还未达到T，导致最终集成只包含很少的基学习器而性能不佳，若采用重采样法，可获得重启动机会，即在抛弃当前基学习器后，根据当前分布重新对训练样本进行采样，再基于新的采样结果重新训练出基学习器，最终完成T轮

从偏差-方差分解的角度看，Boosting主要关注降低偏差，因此可以基于泛化性能相当弱的学习器构建出很强的集成。以决策树桩为基学习器，在西瓜数据集上运行AdaBoost算法，结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115101002732.png#pic_center)

# 3. Bagging与随机森林
由8.1节，欲得到泛化性能强的集成，集成中的个体学习器应尽可能相互独立，虽然独立无法做到，但可以设法使基学习器尽可能有较大的差异，一种可能的做法是对训练样本进行采样得到不同的子集，从每个子集中训练出一个基学习器，但如果每个子集数据完全不同，则每个基学习器只用到了一小部分训练数据，无法保证产生比较好的基学习器，因此可以考虑使用相互有交叠的采样子集。

## Bagging

**过程**：

Bagging是并行式集成学习方法的著名代表，基于自助采样法（bootstrap sampling），即对一个包含m个样本的数据集，每次放回抽样，得到一个包含m个样本的采样集，部分样本可能多次出现，部分样本从未出现。由第二章知，初始训练集中约有63.2%的样本出现在采样集中

训练时，我们采样出T个包含m个训练样本的采样集，基于每个采样集训练出一个基学习器，再将这些基学习器结合。预测时对分类任务使用简单投票法，对回归任务使用简单平均法，同票时随机选一个或进一步考察学习器投票的置信度

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115102308930.png#pic_center)

**优点**：

- 与直接使用基学习算法的复杂度同阶：假定基学习器的计算复杂度为$O(m)$，则Bagging的复杂度大致为$T(O(m)+O(s))$，考虑到采样与投票/平均的复杂度$O(s)$很小，T通常是一个不太大的常数，因此，Bagging集成与直接使用基学习算法的复杂度同阶，很高效
- 可直接用于多分类、回归等任务，而标准AdaBoost只适用于二分类任务
- 可对泛化性能进行包外估计（out-of-bag estimate）：由于采用自助采样，剩下约36.8%的样本可用作验证集来对泛化性能进行包外估计，为此需记录每个基学习器所使用的训练样本，令$H^{o b b}$表示对样本$\pmb x$的包外预测，即仅考虑那些未使用$\pmb x$训练的基学习器上$\pmb x$的预测，有

$$
H^{o b b}(\pmb{x})=\underset{y \in \mathcal{Y}}{\arg \max } \sum_{t=1}^{T} \mathbb{I}\left(h_{t}(\pmb{x})=y\right) \cdot \mathbb{I}\left(\pmb{x} \notin D_{t}\right)
$$

则Bagging泛化误差的包外估计为

$$
\epsilon^{o o b}=\frac{1}{|D|} \sum_{(\pmb{x}, y) \in D} \mathbb{I}\left(H^{o o b}(\pmb{x}) \neq y\right)
$$

- 包外样本还有其他用途：基学习器是决策树时，可用来辅助剪枝，或用于估计决策树中各结点的后验概率以辅助对零训练样本结点的处理；基学习器是神经网络时，可用来辅助早停以减小过拟合风险

**总结**：

从偏差-方差分解的角度看，Bagging主要关注降低方差，因此在不剪枝决策树、神经网络等易受样本扰动的学习器上效用更明显。在西瓜数据集上结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115103712884.png#pic_center)

## 随机森林

**过程**：

随机森林（Random Forest, RF）是Bagging的一个拓展变体，在以决策树为基学习器构建Bagging集成的基础上进一步在决策树的训练过程中引入了随机属性选择

具体来说，传统决策树在选择划分属性时是在当前结点的属性集合（假设有d个属性）中选择一个最优属性，而RF对基决策树的每个结点都从属性集合中随机选择一个包含k个属性的子集，然后从这个子集中选择一个最优属性用于划分

参数k控制了随机性的引入程度

- $k=d$，则基决策树与传统决策树相同
- $k=1$，则随机选择一个属性用于划分
- 推荐值$k=\log _2d$

**优点**：

- 简单、易实现、计算开销小，性能强大，被誉为“代表集成学习技术水平的方法”：与Bagging中基学习器的多样性仅通过样本扰动（通过对初始训练集进行采样）而来不同，随机森林加入了属性扰动，使得最终集成的泛化性能可通过个体学习器之间差异度的增加而进一步提升。但基学习器数量较小时性能往往较差，随着基学习器数目的增加，通常会收敛到更低的泛化误差

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019111510490974.png)

- 训练效率优于Bagging：因为在个体决策树的构建过程中，Bagging使用确定型决策树，在划分属性时要对结点的所有属性进行考察，而随机森林使用随机型决策树只需考察一个属性子集

# 4. 结合策略

学习器结合从三个角度带来好处：

- 从统计的方面来看，由于学习任务的假设空间往往很大，可能有多个假设在训练集上达到同等性能，此时若使用单学习器可能误选导致泛化性能不佳，多个学习器减小这一风险
- 从计算的方面来看，学习算法可能会陷入局部最小，有时会使泛化性能很差，通过对多次运行进行结合，降低陷入糟糕局部最小点的风险
- 从表示的方面看，某些学习任务的真实假设可能不在学习算法所考虑的假设空间内，此时单学习器无效，通过结合多个学习器，由于相应的假设空间有所扩大，有可能学得更好的近似

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115211328256.png#pic_center)

本节假定集成包含T个基学习器$\left\\{h_{1}, h_{2}, \ldots, h_{T}\right\\}$，其中$h_i$在示例$\pmb x$上的输出为$h_i(\pmb x)$

## 平均法

平均法（averaging）适用于数值型输出

- 简单平均法（simple averaging）

$$
H(x)=\frac{1}{T} \sum_{i=1}^{T} h_{i}(x)
$$

- 加权平均法（weighted averaging）：

$$
H(\pmb{x})=\sum_{i=1}^{T} w_{i} h_{i}(\pmb{x})
$$

其中，$w_i$为权重，通常$w_{i} \geqslant 0, \sum_{i=1}^{T} w_{i}=1$

集成学习中的各种结合都可以视为加权平均法的特例或变体，权重一般从训练数据中学习而得。现实任务中的训练样本通常存在不充分或噪声，这将使得学出的权重不完全可靠，尤其对规模比较大的集成来说，由于要学习的权重比较多，容易导致过拟合

加权平均未必一定优于简单平均，在个体学习器性能差别较大时宜用加权平均，在性能相近时宜用简单平均

## 投票法
投票法（voting）适用于分类任务，学习器$h_i$从类别标记集合$\\{c_{1}, c_{2}, \ldots, c_{N}\\}$中预测出一个标记

**符号表示及说明**

为便于讨论，将$h_i$在样本$\pmb x$上的预测输出表示为一个N维向量$\left(h_{i}^{1}(\pmb{x}) ; h_{i}^{2}(\pmb{x}) ; \ldots ; h_{i}^{N}(\pmb{x})\right)$，其中$h_{i}^{j}(\pmb{x})$是$h_i$在类别$c_j$上的输出，常见的类型有：

- 类标记：$h_{i}^{j}(\pmb{x}) \in\{0,1\}$，使用类标记的投票称为硬投票
- 类概率：$h_{i}^{j}(\pmb{x}) \in[0,1]$，使用类概率的投票称为软投票

不同类型的$h_{i}^{j}(\pmb{x})$不能混用，有以下说明：

- 对一些能在预测出类别标记的同时产生分类置信度的学习器，其分类置信度可转化为类概率使用，若此类值未进行规范化，例如支持向量机的分类间隔值，则必须使用一些技术如Platt缩放（Platt scaling）、等分回归（isotonic regression）等进行校准（calibration）后才能作为类概率使用
- 虽然分类器估计的类概率值一般都不太准确，但基于类概率进行结合却往往比直接基于类标记更好
- 若基学习器的类型不同，则其概率值不能直接进行比较，而需转化为类标记再投票

**几种投票法**

- 绝对多数投票法（majority voting）：若某标记得票过半数，则预测为该标记，否则拒绝预测

$$
H(\pmb{x})=\left\{\begin{array}{ll}{c_{j},} & {\text { if } \sum_{i=1}^{T} h_{i}^{j}(\pmb{x})>0.5 \sum_{k=1}^{N} \sum_{i=1}^{T} h_{i}^{k}(\pmb{x})} \\ {\text { reject, }} & {\text { otherwise. }}\end{array}\right.
$$

- 相对多数投票法（plurality voting）：预测为得票最多的标记，若有多个标记获得相等最高票则从中随机选一个

$$
H(\pmb{x})=c_{\arg \max \sum_{i=1}^{T} h_{i}^{j}(\pmb{x})}
$$

- 加权投票法（weighted voting）：$w_i$为权重，通常$w_{i} \geqslant 0, \sum_{i=1}^{T} w_{i}=1$

$$
H(\pmb{x})=c_{\arg \max \sum_{i=1}^{T} w_ih_{i}^{j}(\pmb{x})}
$$

标准的绝对多数投票法提供了拒绝预测选项，这在可靠性要求较高的学习任务中是一个很好的极值，但若学习任务要求必须提供预测结果，则退化为相对多数投票法，此时二者统称为多数投票法

## 学习法
当训练数据很多时，可以采用更强大的结合方式学习法，即通过另一个学习器来进行结合，这里把个体学习器称为初级学习器，用于结合的学习器称为次级学习器或元学习器。

**Stacking**：

Stacking是学习法的典型代表，初级学习器的输出被当做样例输入特征，而初始样本的标记仍被当作阳历标记，假定初级集成是异质的，算法如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115215408994.png#pic_center)

次级学习器是利用初级学习器产生的，若直接用初级学习器的训练集产生次级学习器，过拟合风险较大，一般通过交叉验证或留一法，以k折交叉验证为例，初始训练集被随机划分为k个子集，每个初级学习器每次在每种k-1个子集上训练，在剩下的一个子集上预测，重复上述步骤k次，则这个学习器对每个样本都有一个预测，对每个学习器重复上述步骤，得到每个学习器对每个样本的预测，记样本$\pmb x_i$产生的次级训练样例的示例部分为$\pmb{z}\_{i}=\left(z\_{i 1} ; z\_{i 2} ; \ldots ; z\_{i T}\right)$，标记部分为$y\_i$，因此次级训练集为$D^{\prime}=\\{\left(\pmb{z}\_{i}, y\_{i}\right)\\}\_{i=1}^{m}$

次级学习器的输入属性表示和次级学习算法对Stacking集成的泛化性能有很大影响，研究表明将初级学习器的输出类概率作为次级学习期的输入属性，用多响应线性回归（Multi-response Linear Regression, MLR）作为次级学习算法效果较好，在MLR中使用不同的属性集效果更佳

**贝叶斯模型平均**：

贝叶斯模型平均（Bayes Model Averaging, BMA）基于后验概率来为不同模型赋予权重，可视为加权平均法的一种特殊实现

和Stacking的比较：

- 理论上来说，若数据生成模型恰在当前考虑的模型中，且数据噪声很少，BMA不差于Stacking
- 现实应用中，无法确保数据生成模型一定在当前考虑的模型中，甚至可能难以用当前考虑的模型来近似，因此Stacking通常优于BMA，因为其鲁棒性更好，而且BMA对模型近似误差非常敏感

# 5. 多样性

## 误差-分歧分解

假定我们用个体学习器$h_{1}, h_{2}, \ldots, h_{T}$通过加权平均法结合产生的集成来完成回归学习任务$f: \mathbb{R}^{d} \mapsto \mathbb{R}$

**对单个样本**：

对示例$\pmb x$，定义学习器$h_i$的分歧（ambiguity）为

$$
A\left(h_{i} | \pmb{x}\right)=\left(h_{i}(\pmb{x})-H(\pmb{x})\right)^{2}
$$

则集成的分歧是

$$
\begin{aligned} \bar{A}(h | \pmb{x}) &=\sum_{i=1}^{T} w_{i} A\left(h_{i} | \pmb{x}\right) \\ &=\sum_{i=1}^{T} w_{i}\left(h_{i}(\pmb{x})-H(\pmb{x})\right)^{2} \end{aligned}
$$

这里的分歧表征了个体学习器在样本$\pmb x$上的不一致性，即在一定程度上反映了个体学习器的多样性

个体学习器$h_i$和集成H的平方误差分别为

$$
E\left(h_{i} | \pmb{x}\right)=\left(f(\pmb{x})-h_{i}(\pmb{x})\right)^{2}
$$

$$
E(H | \pmb{x})=(f(\pmb{x})-H(\pmb{x}))^{2}
$$

个体学习器误差的加权均值为

$$
\bar{E}(h | \pmb{x})=\sum_{i=1}^{T} w_{i} \cdot E\left(h_{i} | \pmb{x}\right)
$$

因此有

$$
\begin{aligned} \bar{A}(h | \pmb{x}) &=\sum_{i=1}^{T} w_{i} E\left(h_{i} | \pmb{x}\right)-E(H | \pmb{x}) \\ &=\bar{E}(h | \pmb{x})-E(H | \pmb{x}) \end{aligned}
$$

**对所有样本**：

令$p(\pmb x)$表示样本的概率密度，则在全样本上有

$$
\sum_{i=1}^{T} w_{i} \int A\left(h_{i} | \pmb{x}\right) p(\pmb{x}) d \pmb{x}=\sum_{i=1}^{T} w_{i} \int E\left(h_{i} | \pmb{x}\right) p(\pmb{x}) d \pmb{x}-\int E(H | \pmb{x}) p(\pmb{x}) d \pmb{x}
$$

个体学习器$h_i$在全样本上的繁华误差和分歧项分别为

$$
E_{i}=\int E\left(h_{i} | \pmb{x}\right) p(\pmb{x}) d \pmb{x}
$$

$$
A_{i}=\int A\left(h_{i} | \pmb{x}\right) p(\pmb{x}) d \pmb{x}
$$

集成的泛化误差为

$$
E=\int E(H | \pmb{x}) p(\pmb{x}) d \pmb{x}
$$

个体学习器泛化误差的加权均值为

$$
\bar{E}=\sum_{i=1}^{T} w_{i} E_{i}
$$

个体学习器的加权分歧值为

$$
\bar{A}=\sum_{i=1}^{T} w_{i} A_{i}
$$

因此有

$$
E=\bar{E}-\bar{A}
$$

上式即为误差-分歧分解（error-ambiguity decomposition），个体学习器准确性越高，多样性越大，则集成越好，即个体学习器应好而不同

需注意：

- 我们不能把$\bar{E}-\bar{A}$作为优化目标来求解从而得到最优集成，现实任务中很难直接对$\bar{E}-\bar{A}$进行优化，不仅由于它们是定义在整个样本空间上，还由于$\bar A$不是一个可直接操作的多样性度量，它仅在集成构造好后才能进行估计
- 上述推导过程只适用于回归分析，难以直接推广到分类任务上

## 多样性度量
多样性度量（diversity measure）是用于度量集成中个体分类器的多样性，典型做法是考虑个体分类器的两两相似性或不相似性

给定数据集$D=\left\\{\left(\pmb{x}\_{1}, y\_{1}\right),\left(\pmb{x}\_{2}, y\_{2}\right), \ldots,\left(\pmb{x}\_{m}, y\_{m}\right)\right\\}$，对二分类任务，$y_i\in \\{-1,+1\\}$，分类器$h\_i,h\_j$的预测结果列联表（contingency table）为

| | $h_i=+1$ | $h_i=-1$
-|-|-
$h_j=+1$|a|c
$h_j=-1$|b|d

且有$a+b+c+d=m$，下面给出多样性的几种度量

- 不合度量（disagreement measure）：值域为$[0,1]$，值越大多样性越大

$$
d i s_{i j}=\frac{b+c}{m}
$$

- 相关系数（correlation coefficient）：值域为$[-1,1]$，无关则值为0，正相关为正，负相关为负

$$
\rho_{i j}=\frac{a d-b c}{\sqrt{(a+b)(a+c)(c+d)(b+d)}}
$$

- Q-统计量（Q-statistic）：与相关系数$\rho_{ij}$符号相同，且有$\vert Q_{i j}\vert \leqslant \vert \rho_{ij}\vert$ 

$$
Q_{i j}=\frac{a d-b c}{a d+b c}
$$

- $\kappa$-统计量（$\kappa$-statistic）：若分类器在$h_i,h_j$在D上完全一致则值为1，若他们仅为偶然达成一致则值为0，$\kappa$通常为非负值，仅在二者达成一致的概率甚至低于偶然性的情况下为负值

$$
\kappa=\frac{p_{1}-p_{2}}{1-p_{2}}
$$

其中，$p_1$为两个分类器取得一致的概率，$p_2$为二者偶然打成一致的概率

$$
\begin{aligned} p_{1} &=\frac{a+d}{m} \\ p_{2} &=\frac{(a+b)(a+c)+(c+d)(b+d)}{m^{2}} \end{aligned}
$$

以上均为成对型（pairwise）多样性度量，他们可以容易地用二维图绘制出来，例如$\kappa$-误差图，将每对分类器作为一个点，横坐标为他们的$\kappa$值，纵坐标是他们的平均误差，下图为一个例子。显然，数据点云的位置越高，则个体学习器准确性越低，点云的位置越靠右，则多样性越小

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115233519906.png#pic_center)

## 多样性增强

在学习过程中引入随机性，通常有如下策略，不同的多样性增强机制可同时使用

- 数据样本扰动：给定初始数据集，从中产生不同子集，再利用不同子集训练出不同的学习器，常基于采样法，例如Bagging采用自助采样，Adaboost使用序列采样，对不稳定学习器（如决策树、神经网络等训练样本稍加变化就会导致学习器有显著变化）很有效，对稳定基学习器（stable base learner）（如线性学习器、支持向量机、朴素贝叶斯、K近邻学习器对数据样本的扰动不敏感）需采用其它策略
- 输入属性扰动：训练样本通常由一组属性描述，不同的子空间（subspace），即属性子集提供了观察数据的不同视角。随机子空间（random subspace）算法从初始属性集中抽取出若干个子集，再基于每个属性子集训练一个基学习器，算法如下

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20191115234241558.png#pic_center)
	
	- 对包含大量冗余属性的数据，该方法不仅能产生多样性大的个体，还会因属性数的减少而大幅节省开销，同时由于冗余属性多，减少一些属性后训练出的学习器也不会太差
	- 对只包含少量属性或冗余属性很少的数据不宜采用输入属性扰动法
- 输出表示扰动：对输出表示进行操纵以增强多样性
	- 对训练样本的类标记稍作改动，如翻转法（Flipping Output）随机改变一些训练样本的标记
	- 对输出表示进行转化，输出调制法（Output Smearing）将分类输出转化为回归输出后构建个体学习器
	- 将原任务拆解为多个可同时求解的子任务，如ECOC法，利用纠错输出码将多分类任务拆解为一系列二分类任务来训练基学习器
- 算法参数扰动：随机设置不同的参数
	- 例如负相关法，显式地通过正则化来强制个体神经网络使用不同的参数
	- 对参数少的算法可将其学习过程中某些环节用其它类似方法代替，例如决策树中更改属性选择机制
	- 值得注意的是，单一学习器通常需要使用交叉验证等方法确定参数，事实上已经使用不同参数训练出多个学习器，只是最终仅选择一个，而集成学习相当于把这些学习器都利用起来，计算开销不比单一学习器大很多




