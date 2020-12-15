---
title: 主题模型 | Supervised Topic Models（包含推导证明、理解、注释等）
author: 钟欣然
date: 2020-12-12 00:44:00 +0800
categories: [杂谈, 主题模型]
math: true
mermaid: true
---

**摘要：** sLDA（supervised latent Dirichlet allocation）是针对标记文档的主题模型，该模型采用变分EM算法，适用于多种响应变量（response variable），优于lasso和先使用无监督主题模型再进行回归。

# 1. 背景

现在有很多大型语料库，需要建立合适的统计模型对其进行分析，如基于主题的分层统计模型LDA等，但现在的主题模型大都是无监督的，只对文档中的单词进行建模，最大化其后验概率。

本文关注有响应变量的文档，如电影评论有数字作为评级、论文有下载次数，对其建立监督主题模型，和无监督主题模型主要用于分类（降维）不同，sLDA主要用于预测。

# 2. 模型

主题模型中每个文档被表示为单词$w_{1:n}$的集合，我们将文档看成是一组潜在主题产生的单词，即词汇上的一组未知分布。语料库中的文档共享相同的K个主题，但是不同的文档有不同的主题比例。在LDA中，我们从狄利克雷分布中抽取主题比例，然后从这些比例中抽取一个主题，再从相应的主题中抽取一个单词，重复$n$次构成一个文档。

在sLDA中，我们在LDA的基础上，对每个文档添加一个响应变量，对文档和响应变量共同建模，以便找到潜在的主题，从而最好地预测未标记文档的响应变量。

sLDA使用与广义线性模型相同的概率机制来适应各种类型的响应变量：无约束实值、被约束为正的实值（例如故障时间）、有序或无序的类标签、非负整数（例如计数数据）等。

模型参数：
- $K$个主题$\beta_{1:K}$，$\beta_k$是第k个主题中每个词出现概率的向量
- 狄利克雷参数$\alpha$
- 响应参数$\eta,\delta$

在sLDA中，每个文档和响应变量都来自以下生成过程：

1. 抽取主题比例$\theta \vert \alpha \sim \operatorname{Dir}(\alpha)$
2. 对于每一个词：
	
	（a）抽取主题$z_{n} \vert \theta \sim \operatorname{Mult}(\theta)$
	
	（b）抽取单词$w_{n} \vert z_{n}, \beta_{1: K} \sim \operatorname{Mult}\left(\beta_{z_{n}}\right)$
	
3. 抽取响应变量$y \vert z_{1: N}, \eta, \delta \sim \operatorname{GLM}(\bar{z}, \eta, \delta)$，其中$\bar{z}=\frac1N\sum_{n=1}^{N} z_{n}$

图示如下：

![Alt](https://img-blog.csdnimg.cn/20191222201838726.png#pic_center =500x252)
<center>图1 sLDA的图形表示 </center><br>

响应变量的分布为广义线性模型

$$
p\left(y \vert z_{1: N}, \eta, \delta\right)=h(y, \delta) \exp \left\{\frac{\left(\eta^{\top} \bar{z}\right) y-A\left(\eta^{\top} \bar{z}\right)}{\delta}\right\} \tag 1
$$

> $h(y,\delta)$为潜在测度，$\eta^{\top}\bar z$为自然参数，$\delta$为分散参数（为对$y$的方差建模提供了灵活性），$A(\eta^{\top}\bar z)$对数规范化因子

GLM框架为我们提供了对不同类型的响应变量建模的灵活性，只要响应变量的分布可以被写成上式指数分散族（exponential dispersion family）的形式。包括很多常用分布，如正态分布（适用于实值响应变量）、二项分布（适用于二项响应变量）、多项分布（适用于分类响应变量）、泊松分布和负二项分布（适用于计数响应变量）、伽马分布和威布尔分布和逆高斯分布（适用于故障时间数据）等。每个分布都对应于特定的$h(y, \delta)$和$A\left(\eta^{\top} \bar{z}\right)$。

sLDA与通常的GLM的区别在于协变量是文档中主题的经验频率，这些经验频率是不能直接观测到的。在生成过程中，这些潜在的变量负责生成文档的单词，因此响应变量和单词得以联系在一起。回归系数记为$\eta$。注意到GLM中通常包含截距项，相当于添加一个恒等于1的协变量，而在sLDA中，这一项是多余的，因为$\bar z$的各分量和恒为1。

对主题的经验频率而不是主题比例$\theta$进行回归，前者将响应变量和单词视为是不可交换的（exchangeable），首先在全部单词可交换的条件下生成文档（单词及其主题分配），然后基于该文档生成响应变量，后者将响应变量和单词视为是可交换的。前者更加合理，因为响应变量取决于文档中实际出现的主题频率，而不是产生主题的分布，如果主题数足够多，在后者中允许一些主题被完全用来解释响应变量，另一些主题被完全用来解释单词的出现，这降低了预测的性能，而在前者中，决定响应变量的潜在变量和决定单词出现的潜在变量是相同的。~~这个模型不能推断用于解释响应变量的主题集合，也不能用它来解释一些观测到的单词。

# 3. 推导
推导包括三个部分：

1. 后验推断：给定单词$w_{1:N}$和语料库范围内的模型参数，计算文档级别的潜变量的条件分布，即主题比例$\theta$和主题分配$z_{1:N}$的条件分布。这个分布不能直接计算，我们采用变分推断对其进行近似。
2. 参数估计：给定文档和响应变量对$\left\\{w_{d, 1: N}, y_{d}\right\\}_{d=1}^{D}$，

	估计狄利克雷参数$\alpha$，GLM参数$\eta$和$\delta$，主题多项式$\beta_{1:K}$。我们采用变分EM算法。
	
3. 预测：给定新文档$w_{1:N}$和模型参数，预测响应变量$y$。这相当于近似得到后验期望$\mathrm{E}\left[y \vert w_{1: N}, \alpha, \beta_{1: K}, \eta, \delta\right]$

我们针对sLDA的一般GLM设置依次处理这些问题，并指出需要计算或近似GLM特定量的位置，然后针对高斯响应变量和泊松响应变量的特殊情况计算精确的式子，最后对其它响应变量分布采用通用的近似方法。

## 后验推断

参数估计和预测都依赖于后验推断，给定文档和响应变量，潜变量的后验分布是

$$
\begin{aligned}p(\theta, z_{1: N} &\vert w_{1: N}, y, \alpha, \beta_{1: K}, \eta, \delta )\\ &= \frac{p(\theta \vert \alpha)\left(\prod_{n=1}^{N} p\left(z_{n} \vert \theta\right) p\left(w_{n} \vert z_{n}, \beta_{1: K}\right)\right) p\left(y \vert z_{1: N}, \eta, \delta\right)}{\int d \theta p(\theta \vert \alpha) \sum_{z_{1: N}}\left(\prod_{n=1}^{N} p\left(z_{n} \vert \theta\right) p\left(w_{n} \vert z_{n}, \beta_{1: K}\right)\right) p\left(y \vert z_{1: N}, \eta, \delta\right)} \end{aligned}\tag{2}
$$

归一化值为观察到的值，即文档$w_{1:N}$和响应变量$y$的边际概率，我们采用变分方法来近似后验概率。

变分方法包括许多类型的后验归一化值的近似，这里我们使用平均场变分推断（$q(\boldsymbol z)$对$\boldsymbol z$的所有分量是独立的，$q(\boldsymbol z)=q(z_1)q(z_2)\dots q(z_n)$），其中詹森不等式用于归一化值的下限，令$\pi$表示模型参数值，$\pi=\left\\{\alpha, \beta_{1: K}, \eta, \delta\right\\}$，令$q\left(\theta, z_{1: N}\right)$表示潜在变量的变分分布。变分分布与真实后验分布的的KL散度

$$
\begin{aligned}D(q(\theta,z_{1:N})&\Vert p(\theta,z_{1:N}\vert w_{1:N},\pi))\\&=\mathrm{E}_q[\log q(\theta,z_{1:N})]-\mathrm{E}_q[\log p(\theta,z_{1:N}\vert w_{1:N},\pi)]\\&=\mathrm{E}_q[\log q(\theta,z_{1:N})]-\mathrm{E}_q[\log p(\theta,z_{1:N},w_{1:N}\vert\pi)]+\log p(w_{1:N}\vert\pi)\\&\geq 0\end{aligned}\tag 3
$$

因此证据下界（ELBO）为

$$
\log p(w_{1:N}\vert\pi)\geq \mathrm{E}_q[\log p(\theta,z_{1:N},w_{1:N}\vert\pi)]-\mathrm{E}_q[\log q(\theta,z_{1:N})] \tag 4
$$

我们将其记为$\mathcal{L}(\cdot)$，第一项为对隐藏变量和观察变量联合概率的对数的期望，第二项为变分分布的熵，记$\mathrm{H}(q)=-\mathrm{E}\left[\log q\left(\theta, z_{1: N}\right)\right]$，在其拓展形式中，sLDA ELBO为

$$
\begin{aligned}\mathcal{L}\left(w_{1: N}, y \vert \pi\right)&=\mathrm{E}_q[\log p(\theta \vert \alpha)]+\sum_{n=1}^{N} \mathrm{E}_q\left[\log p\left(z_{n} \vert \theta\right)\right]  \\&+\sum_{n=1}^{N} \mathrm{E}_q\left[\log p\left(w_{n} \vert z_{n}, \beta_{1: K}\right)\right]+\mathrm{E}_q\left[\log p\left(y \vert z_{1: N}, \eta, \delta\right)\right]+\mathrm{H}(q)\end{aligned}\tag 5
$$

在变分推断中，我们首先为变分分布构造一个参数化族，然后对给定的观测值拟合它的参数以最大化（5）式。变分分布的参数化决定了优化的速度。当$q\left(\theta, z_{1: N}\right)$就是后验分布时，（5）式恰好等于$\log p(w_{1:N}\vert\pi)$，但是包含后验分布的分布族会导致难以解决的优化问题，因此我们选择了一个更简单的可以完全分解的族

$$
q\left(\theta, z_{1: N} \vert \gamma, \phi_{1: N}\right)=q(\theta \vert \gamma) \prod_{n=1}^{N} q\left(z_{n} \vert \phi_{n}\right)\tag6
$$

这里$\gamma$是$K$维狄利克雷参数，$\phi_n$是$K$维多项分布，$z_n$为$K$维指示向量，有$\mathrm{E}\left[z_{n}\right]=q\left(z_{n}\right)=\phi_{n}$。最大化（5）式相当于找到在KL散度意义下最接近后验分布的变分分布。因此，给定文档和响应变量对，我们寻找使（5）式最大化的$\phi_{1:N}$和$\gamma$，从而估计后验分布。

在解决优化问题之间，我们进一步展开（5）式。前三项和变分分布的熵与无监督LDA相同：

- 第一项：

$$
\mathrm{E}[\log p(\theta \vert \alpha)]=\log \Gamma\left(\sum_{i=1}^{K} \alpha_{i}\right)-\sum_{i=1}^{N} \log \Gamma\left(\alpha_{i}\right)+\sum_{i=1}^{K}\left(\alpha_{i}-1\right) \mathrm{E}\left[\log \theta_{i}\right]\tag 7
$$

> 推导：根据狄利克雷分布
> 
> $$
> p(\theta \vert \alpha)=\frac{\Gamma\left(\sum_{i=1}^{K} \alpha_{i}\right)}{\prod_{i=1}^K\Gamma(\alpha_i)}\prod_{i=1}^K\theta_i^{\alpha_i-1}
> $$
> 

- 第二项：

$$
\operatorname{E}\left[\log p\left(z_{n} \vert \theta\right)\right]=\sum_{i=1}^{K} \phi_{n, i} \mathrm{E}\left[\log \theta_{i}\right]\tag 8
$$

> 推导：
> 
> $$
> \begin{aligned} E\left[\log p\left(z_{n} \vert \theta\right)\right] &=E_{q\left(\theta, z_{n} \vert \gamma, \phi_{1:N}\right)}\left[\log p\left(z_{n} \vert \theta\right)\right] \\ &=\sum_{i=1}^{K} q\left(z_{n,i} \vert \phi_{n}\right) E_{q(\theta \vert \gamma)}\left[\log \theta_{i}\right] \\&=\sum_{i=1}^{K} \phi_{n, i} \mathrm{E}\left[\log \theta_{i}\right]\end{aligned}
> $$
> 

- 第三项：

$$
\mathrm{E}\left[\log p\left(w_{n} \vert z_{n}, \beta_{1: K}\right)\right]=\sum_{i=1}^{K} \phi_{n, i} \log \beta_{i, w_{n}}\tag {9}
$$

> 推导：
> 
> $$
> \begin{aligned} E\left[\log p\left(w_{n} \vert z_{n}, \beta_{1:K}\right)\right]  &=E_{q\left(z_{n} \vert \phi_n\right)}\left[\log p\left(w_{n} \vert z_{n}, \beta_{1:K}\right)\right] \\ &=\sum_{i=1}^{K} q\left(z_{n,i} \vert \phi_n\right) \log p\left(w_{n} \vert z_{n,i},  \beta_{1:K}\right) \\ &=\sum_{i=1}^{K} \phi_{n, i} \log \beta_{i, w_{n}} \end{aligned}
> $$
> 

- 变分分布的熵$H(q)$：

$$
\begin{aligned}\mathrm{H}(q)=&-\sum_{n=1}^{N} \sum_{i=1}^{K} \phi_{n, i} \log \phi_{n, i}-\log \Gamma\left(\sum_{i=1}^{K} \gamma_{i}\right)\\&+\sum_{i=1}^{K} \log \Gamma\left(\gamma_{i}\right)-\sum_{i=1}^{K}\left(\gamma_{i}-1\right) \mathrm{E}\left[\log \theta_{i}\right]\end{aligned}\tag{10}
$$

> 推导：
> 
> $$
> H(q)=-E_{q}[\log q(\mathbf{z} \vert \phi_{1:N})]-E_{q}[\log q(\theta \vert \gamma)]
> $$
> 
> $$
> \begin{aligned} E_{q}[\log q(\mathbf{z} \vert \phi_{1:N})] &=\sum_{n=1}^{N} E_{q}\left[\log q\left(z_{n} \vert \phi_{1:N}\right)\right] \\ &=\sum_{n=1}^{N} E_{q\left(z_{n} \vert \phi_{1:N}\right)}\left[\log q\left(z_{n} \vert \phi_{1:N}\right)\right] \\ &=\sum_{n=1}^{N} \sum_{i=1}^{K} q\left(z_{n,i} \vert \phi_{1:N}\right) \log q\left(z_{n,i} \vert \phi_{1:N}\right) \\ &=\sum_{n=1}^{N} \sum_{i=1}^{K} \phi_{n,i} \log \phi_{n,i} \end{aligned}
> $$
> 
> 又根据狄利克雷分布
> 
> $$
> p(\theta \vert \gamma)=\frac{\Gamma\left(\sum_{i=1}^{K} \gamma_{i}\right)}{\prod_{i=1}^K\Gamma(\gamma_i)}\prod_{i=1}^K\theta_i^{\gamma_i-1}
> $$
> 
> 有
> 
> $$
> E_{q}[\log q(\theta \vert \gamma)]=\log \Gamma\left(\sum_{i=1}^{K} \gamma_{i}\right)-\sum_{i=1}^{K} \log \Gamma\left(\gamma_{i}\right)+\sum_{i=1}^{K}\left(\gamma_{i}-1\right) \mathrm{E}\left[\log \theta_{i}\right]
> $$
> 

注意：上述诸式中服从狄利克雷分布的随机变量的对数期望为

$$
\mathrm{E}\left[\log \theta_{i}\right]=\Psi\left(\gamma_{i}\right)-\Psi\left(\sum_{j=1}^{K} \gamma_{j}\right)
$$

其中，$\Psi(x)$表示对数伽马函数的一阶导数

$$
\Psi\left(x\right)=\frac{\mathrm{d}}{\mathrm{d}x} \log \Gamma\left(x\right)
$$

> 证明：设随机变量$\theta$服从狄利克雷分布$\theta \sim \operatorname{Dir}(\theta \vert \alpha)$，利用指数分布族性质，求函数$log(\theta)$关于狄利克雷分布的数学期望$E[\log \theta]$<br>
> 指数分布族为$p(x \vert \eta)=h(x) \exp \left\\{\eta^{\mathrm{T}} T(x)-A(\eta)\right\\}$，其中$\eta$为自然参数，$T(x)$为充分统计量，$h(x)$为潜在测度，$A(\eta)$是对数规范化因子$A(\eta)=\log \int h(x) \exp \left\\{\eta^{\mathrm{T}} T(x)\right\\} \mathrm{d} x$<br>
> 指数分布族具有性质：对数规范化因子$A(\eta)$ 对自然参数$\eta$的导数等于充分统计量$T(x)$的数学期望
> 
> $$
> \begin{aligned} \frac{\mathrm{d}}{\mathrm{d} \eta} A(\eta) &=\frac{\mathrm{d}}{\mathrm{d} \eta} \log \int h(x) \exp \left\{\eta^{\mathrm{T}} T(x)\right\} \mathrm{d} x \\ &=\frac{\int T(x) \exp \left\{\eta^{\mathrm{T}} T(x)\right\} h(x) \mathrm{d} x}{\int h(x) \exp \left\{\eta^{\mathrm{T}} T(x)\right\} \mathrm{d} x} \\ &=\int T(x) p(x \vert \eta) \mathrm{d} x \\ &=E[T(X)] \end{aligned}
> $$
> 
> 狄利克雷分布属于指数分布族，因为其密度函数可以写成指数分布族的密度函数
形式
> 
> $$
> \begin{aligned} p(\theta \vert \alpha) &=\frac{\Gamma\left(\sum_{l=1}^{K} \alpha_{l}\right)}{K} \prod_{k=1}^{K} \theta_{k}^{\alpha_{k}-1} \\ &=\exp \left\{\left(\sum_{k=1}^{K}\left(\alpha_{k}-1\right) \log \theta_{k}\right)+\log \Gamma\left(\sum_{l=1}^{K} \alpha_{l}\right)-\sum_{k=1}^{K} \log \Gamma\left(\alpha_{k}\right)\right\} \end{aligned}
> $$
> 
> 其中，自然参数为$\eta_k=\alpha_k-1$，充分统计量为$T(\theta_k)=\log \theta_k$，对数规范化因子是$A(\alpha)=\sum_{k=1}^{K} \log \Gamma\left(\alpha_{k}\right)-\log \Gamma\left(\sum_{l=1}^{K} \alpha_{l}\right)$<br>
>  
> 因此
> 
> $$
> \begin{aligned} E_{p(\theta \vert \alpha)}\left[\log \theta_{k}\right] &=\frac{\mathrm{d}}{\mathrm{d} \alpha_{k}} A(\alpha)=\frac{\mathrm{d}}{\mathrm{d} \alpha_{k}}\left[\sum_{k=1}^{K} \log \Gamma\left(\alpha_{k}\right)-\log \Gamma\left(\sum_{l=1}^{K} \alpha_{l}\right)\right] \\ &=\Psi\left(\alpha_{k}\right)-\Psi\left(\sum_{l=1}^{K} \alpha_{l}\right), \quad k=1,2, \cdots, K \end{aligned}
> $$
> 

sLDA的证据下界与LDA不同的地方是第四项：

$$
\mathrm{E}\left[\log p\left(y \vert z_{1: N}, \eta, \delta\right)\right]=\log h(y, \delta)+\frac{1}{\delta}\left[\eta^{\top}(\mathrm{E}[\bar{z}] y)-\mathrm{E}\left[A\left(\eta^{\top} \bar z\right)\right]\right]\tag{11}
$$

其中

$$
\mathrm{E}[\bar{z}]=\bar{\phi}=\frac{1}{N} \sum_{n=1}^{N} \phi_{n}\tag{12}
$$

$\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]$ 在某些模型中是可以计算的，如当响应变量为高斯分布或泊松分布时。

但一般的情况下只能进行估计，这部分将在3.4节进行讨论。此处我们假定这个问题是可以解决的，继续进行推导。

我们依次针对每个变分参数使用坐标上升法最大化（5）式，其$\gamma$的更新规则与无监督LDA相同，因为其不涉及响应变量$y$：

$$
\gamma^{\mathrm{new}} \leftarrow \alpha+\sum_{n=1}^{N} \phi_{n}\tag{13}
$$

> 推导：<br>
设$\gamma_k$是第$k$个话题的狄利克雷分布参数，考虑（5）式关于$\gamma_k$的最大化
> 
> $$
> \begin{aligned} L_{\left[\gamma_{k}\right]}=& \sum_{k=1}^{K}\left(\alpha_{k}-1\right)\left[\Psi\left(\gamma_{k}\right)-\Psi\left(\sum_{l=1}^{K} \gamma_{l}\right)\right]+\sum_{n=1}^{N} \sum_{k=1}^{K} \phi_{n k}\left[\Psi\left(\gamma_{k}\right)-\Psi\left(\sum_{l=1}^{K} \gamma_{l}\right)\right]-\\ & \log \Gamma\left(\sum_{l=1}^{K} \gamma_{l}\right)+\log \Gamma\left(\gamma_{k}\right)-\sum_{k=1}^{K}\left(\gamma_{k}-1\right)\left[\Psi\left(\gamma_{k}\right)-\Psi\left(\sum_{l=1}^{K} \gamma_{l}\right)\right] \end{aligned}
> $$
> 
> 简化为
> 
> $$
> L_{\left[\gamma_{k}\right]}=\sum_{k=1}^{K}\left[\Psi\left(\gamma_{k}\right)-\Psi\left(\sum_{l=1}^{K} \gamma_{l}\right)\right]\left(\alpha_{k}+\sum_{n=1}^{N} \phi_{n k}-\gamma_{k}\right)-\log \Gamma\left(\sum_{l=1}^{K} \gamma_{l}\right)+\log \Gamma\left(\gamma_{k}\right)
> $$
> 
> 对$\gamma_k$求偏导数得
> 
> $$
> \frac{\partial L}{\partial \gamma_{k}}=\left[\Psi^{\prime}\left(\gamma_{k}\right)-\Psi^{\prime}\left(\sum_{l=1}^{K} \gamma_{l}\right)\right]\left(\alpha_{k}+\sum_{n=1}^{N} \phi_{n k}-\gamma_{k}\right)
> $$
> 
> 令其为零，得
> 
> $$
> \gamma_{k}=\alpha_{k}+\sum_{n=1}^{N} \phi_{n k}
> $$
> 

sLDA和LDA的核心区别是变分多项式参数$\phi_n$的更新，求证据下界对$\phi_n$的偏导数

$$
\begin{aligned}\frac{\partial \mathcal{L}}{\partial \phi_{n}}=&\mathrm{E}[\log \theta]+\mathrm{E}\left[\log p\left(w_{n} \vert \beta_{1: K}\right)\right]-\log \phi_{n}+1\\&+\left(\frac{y}{N \delta}\right) \eta-\left(\frac{1}{\delta}\right) \frac{\partial}{\partial \phi_{n}}\left\{\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]\right\}\end{aligned}\tag{14}
$$

对$\phi_n$的更新取决于$\frac{\partial}{\partial \phi_{n}}\left\\{\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]\right\\}$

在某些模型中，如当响应变量为高斯分布或泊松分布时，是可以计算的，但一般的情况下需要基于梯度的优化方法，这部分将在3.4节进行讨论。

根据（13）和（14）式，通过迭代更新变分参数$\left\\{\gamma, \phi_{1: N}\right\\}$来进行变分推断，找到（5）中ELBO的局部最优值，所得的变分分布被用作后验分布。

## 参数估计

sLDA的参数为$K$个主题$\beta_{1:K}$，狄利克雷超参数$\alpha$，GLM参数$\eta,\delta$，我们采用变分EM算法，变分EM算法是EM算法（期望最大化）的一种近似形式，其中期望是对变分分布来取的。变分EM算法对整个语料库的证据下界，即各个文档的证据下界之和，进行优化。

由于考虑的是整个语料库，需要将文档索引添加到上述诸式中，因此响应变量从$y$变为$y_d$，主题分配的经验频率$\bar z$变为$\bar z_d$，以此类推。注意到不同文档的变分分布不同，此处的期望是对文档特定的变分分布$q_{d}\left(z_{1: N}, \theta\right)$来取的

$$
\mathcal{L}\left(\alpha, \beta_{1: K}, \eta, \delta ; \mathcal{D}\right)=\sum_{d=1}^{D} \mathrm{E}_{d}\left[\log p\left(\theta_{d}, z_{d, 1: N}, w_{d, 1: N}, y_{d}\right)\right]+\mathrm{H}\left(q_{d}\right)\tag{15}
$$

变分EM算法：

- E步：根据3.1节的变分推断算法来估计每个文档响应变量对的近似后验分布
- M步：针对模型参数最大化整个语料库的ELBO
	- 估计主题参数$\beta_{1:K}$：与无监督LDA相同，第k个主题生成词w的概率与这个词被分配给该主题的预期次数成比例，见（16）式，其中$\phi_{d, n}^{k}$表示根据变分分布第$d$个文本的第$n$个单词属于第$k$个话题的概率，成比例意味着每个新的$\beta_k$被重新归一化为1
	- 估计GLM参数$\eta,\delta$：语料库级别的ELBO对GLM系数$\eta$的偏导数见（17）式，其中$\mu(\cdot)=\mathrm{E}\_{\mathrm{GLM}}[Y \vert \cdot]$（指数分布族具有性质：对数规范化因子对自然参数的导数等于充分统计量的数学期望），响应变量在GLM分布下的期望是$\eta^{\top} \bar{z}_{d}$的函数，在某些模型中，如当响应变量为高斯分布或泊松分布时，有精确的解，但一般的情况下只能获得近似的期望，这部分将在3.4节进行讨论。<br>
	语料库级别的ELBO对$\delta$的偏导数见（18）式，（18）式可以在优化系数$\eta$的同时精确或近似地计算出右边的总和，根据$h(y,\delta)$及其相对于$\delta$的偏导数，我们可以用封闭形式或通过一维数值优化得到新的$\delta$
	- 估计狄利克雷参数$\alpha$：估计狄利克雷参数和估计狄利克雷分布的步骤相同。在完全观测的环境中，狄利克雷分布的充分统计量是观测到的向量的对数。这里，它们是对数主题比例的期望，见（19）式

$$
\hat{\beta}_{k, w}^{\mathrm{new}} \propto \sum_{d=1}^{D} \sum_{n=1}^{N} I\left(w_{d, n}=w\right) \phi_{d, n}^{k}$其中，$\phi_{d, n}^{k}\tag{16}
$$

$$
\begin{aligned} \frac{\partial \mathcal{L}}{\partial \eta} &=\frac{\partial}{\partial \eta}\left(\frac{1}{\delta}\right) \sum_{d=1}^{D}\left\{\eta^{\top} \mathrm{E}\left[\bar{z}_{d}\right] y_{d}-\mathrm{E}\left[A\left(\eta^{\top} \bar{z}_{d}\right)\right]\right\} \\ &=\left(\frac{1}{\delta}\right)\left\{\sum_{d=1}^{D} \bar{\phi}_{d} y_{d}-\sum_{d=1}^{D} \mathrm{E}_{d}\left[\mu\left(\eta^{\top} \bar{z}_{d}\right) \bar{z}_{d}\right]\right\} \end{aligned}\tag{17}
$$

$$
\left\{\sum_{d=1}^{D} \frac{\partial h\left(y_{d}, \delta\right) / \partial \delta}{h\left(y_{d}, \delta\right)}\right\}-\left(\frac{1}{\delta^2}\right)\left\{\sum_{d=1}^{D}\left[\hat{\eta}_{\text {new }}^{\top}\left(\mathrm{E}\left[\bar{z}_{d}\right] y_{d}\right)-\mathrm{E}\left[A\left(\hat{\eta}_{\text {new }}^{\top} \bar{z}_{d}\right)\right]\right\}\right.\tag{18}
$$

$$
\alpha_i=\mathrm{E}\left[\log \theta_{i}\right]=\Psi\left(\gamma_{i}\right)-\Psi\left(\sum_{j=1}^{K} \gamma_{j}\right)\tag{19}
$$

## 预测

应用sLDA的重点是预测。给定新文件$w_{1:N}$和拟合模型$\left\\{\alpha, \beta_{1: K}, \eta, \delta\right\\}$，我们要计算预期响应值

$$
\mathrm{E}\left[Y \vert w_{1: N}, \alpha, \beta_{1: K}, \eta, \delta\right]=\mathrm{E}\left[\mu\left(\eta^{\top} \bar{z}\right) \vert w_{1: N}, \alpha, \beta_{1: K}\right]\tag{20}
$$

为了进行预测，我们使用变分推断近似$\bar z$的后验均值。步骤与3.1节相同，但是此处有关响应变量$y$的项应从（5）式中的ELBO中移除。对变分参数实施以下坐标上升更新

$$
\gamma^{\text {new }}=\alpha+\sum_{n=1}^{N} \phi_{n}\tag{21}
$$

$$
\phi_{n}^{\mathrm{new}} \quad \propto \quad \exp \left\{\mathrm{E}_{q}[\log \theta]+\log \beta_{w_{n}}\right\}\tag{22}
$$

其中向量的对数是对每个分量取对数的向量，$\beta_{w_n}$是由$p\left(w_{n} \vert \beta_{k}\right)$组成的$K$维向量，比例意味着向量被重新归一化为1。这种坐标上升算法与无监督LDA的变分推断相同。

因此，给定一个新的文献，我们首先计算潜在变量$\theta$和$z_n$的变分后验分布$q(\theta,z_{1:N})$，然后用以下方法估计响应变量

$$
\mathrm{E}\left[Y \vert w_{1: N}, \alpha, \beta_{1: K}, \eta, \sigma^{2}\right] \approx \mathrm{E}_{q}\left[\mu\left(\eta^{\top} \bar{z}\right)\right]\tag{23}
$$

和参数估计一样，这取决于能够计算或近似得到$\mathrm{E}_{q}\left[\mu\left(\eta^{\top} \bar{z}\right)\right]$，将在3.4节进行讨论。

## 高斯、泊松及一般的响应变量分布

3.1至3.3节概述了sLDA的一般计算策略：

- 变分推断：在每个文档和响应变量的基础上计算主题比例$\theta$和主题分配$z_{1:N}$的近似后验分布
- 参数估计：使用变分EM算法拟合主题$\beta_{1:K}$和GLM参数$\eta,\delta$
- 预测：基于变分后验分布的近似期望从观察到的文档预测新响应变量

响应变量的分布在三个点上阻碍了我们的推导：

- 变分推断（5）式中每个文档的ELBO中$\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]$的计算
- 变分推断（14）式中变分参数$\phi_n$的更新中$\frac{\partial}{\partial \phi_{n}}\left\\{\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]\right\\}$的计算
- 参数估计（17）式中变分EM算法中拟合GLM参数时$\mathrm{E}\_{q}\left[\mu\left(\eta^{\top} \bar{z}\_{d}\right) \bar{z}\_{d}\right]$的计算
- 预测（23）式中响应变量的预测值$\mathrm{E}\_{q}\left[\mu\left(\eta^{\top} \bar{z}\_{d}\right)\right]$的计算

我们重点推导响应变量服从高斯分布和泊松分布的情况，然后给出在GLM框架下处理其他响应变量分布类型的通用近似方法。

### 高斯分布

响应变量服从高斯分布时，可以将其写成如下指数族形式：

$$
\begin{aligned} p(y \vert \bar{z}, \eta, \delta) &=\frac{1}{\sqrt{2 \pi \delta}} \exp \left\{-\frac{\left(y-\eta^{\top} \bar{z}\right)^{2}}{2 \delta}\right\} \\ &=\frac{1}{\sqrt{2 \pi \delta}} \exp \left\{\frac{-y^{2} / 2+y \eta^{\top} \bar{z}-\left(\eta^{\top} \bar{z} \bar{z}^{\top} \eta\right) / 2}{\delta}\right\} \end{aligned}\tag{24}
$$

这里均值即为自然参数$\eta^{\top} \bar{z}$，因此$\mu\left(\eta^{\top} \bar{z}\right)=\eta^{\top} \bar{z}$。方差等于$\delta$，且有

$$
h(y, \delta)=\frac{1}{\sqrt{2 \pi \delta}} \exp \left\{-y^{2} / 2\right\}\tag {25}
$$

$$
A\left(\eta^{\top} \bar{z}\right)=\left(\eta^{\top} \bar{z} \bar{z}^{\top} \eta\right) / 2\tag{26}
$$

**变分推断（5）式中每个文档的ELBO中$\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]$的计算：**

上式依赖于$\mathrm{E}\left[\bar{z} \bar{z}^{\top}\right]$ 

$$
\begin{aligned} \mathrm{E}\left[\bar{z} \bar{z}^{\top}\right] &=\frac{1}{N^{2}} \sum_{n=1}^{N} \sum_{m=1}^{N} \mathrm{E}\left[z_{n} z_{m}^{\top}\right] \\ &=\frac{1}{N^{2}}\left(\sum_{n=1}^{N} \sum_{m \neq n} \phi_{n} \phi_{m}^{\top}+\sum_{n=1}^{N} \operatorname{diag}(\phi_{n})\right) \end{aligned}\tag{27}
$$

注意到$m \neq n$时，由于变分分布是可以完全分解的，$\mathrm{E}\left[z_{n} z_{m}^{\top}\right]=\mathrm{E}\left[z_{n}\right] \mathrm{E}\left[z_{m}\right]^{\top}=\phi_{n} \phi_{m}^{\top}$，$m=n$时，由于$z_n$是指示向量，$\mathrm{E}\left[z_{n} z_{n}^{\top}\right]=\operatorname{diag}\left(\mathrm{E}\left[z_{n}\right]\right)=\operatorname{diag}\left(\phi_{n}\right)$。

**变分推断（14）式中变分参数$\phi_n$的更新中$\frac{\partial}{\partial \phi_{n}}\left\\{\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]\right\\}$的计算**

将$\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]$看作单个变分参数$\phi_n$的函数，令$\phi_{-j}:=\sum_{n \neq j} \phi_{n}$，则有

$$
\begin{aligned} f\left(\phi_{j}\right) &=\left(\frac{1}{2 N^{2}}\right) \eta^{\top}\left[\phi_{j} \phi_{-j}^{\top}+\phi_{-j} \phi_{j}^{\top}+\operatorname{diag}\left\{\phi_{j}\right\}\right] \eta+\text { const } \\ &=\left(\frac{1}{2 N^{2}}\right)\left[2\left(\eta^{\top} \phi_{-j}\right) \eta^{\top} \phi_{j}+(\eta \circ \eta)^{\top} \phi_{j}\right]+\text { const } \end{aligned}\tag{28}
$$

> $\phi,\eta$均为$K\times1$维向量，$\eta\circ \eta=(\eta_1^2,\eta_2^2,\dots ,\eta_K^2)$

因此导数为

$$
\frac{\partial}{\partial \phi_{j}} \mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]=\frac{1}{2 N^{2}}\left[2\left(\eta^{\top} \phi_{-j}\right) \eta+(\eta \circ \eta)\right]\tag{29}
$$

将这个梯度代入（14）式得到$\phi_n$的更新

$$
\begin{aligned}\phi_{j}^{\mathrm{new}} &\propto \\&\exp \left\{\mathrm{E}[\log \theta]+\mathrm{E}\left[\log p\left(w_{j} \vert \beta_{1: K}\right)\right]+\left(\frac{y}{N \delta}\right) \eta-\frac{1}{2 N^{2} \delta}\left[2\left(\eta^{\top} \phi_{-j}\right) \eta+(\eta \circ \eta)\right]\right\}\end{aligned}\tag{30}
$$

向量的指数等于这个向量的每个分量的指数组成的向量，成比例意味着每个新的$\phi_j$被重新归一化为1

> 上式进一步揭示了sLDA和LDA的区别和联系。sLDA和LDA一样，第$j$个词在不同主题上的变分分布$\phi_j$取决于这个词来自各个主题的实际概率（由$\beta_{1:K}$决定），不同的是，在sLDA中第$j$个词和其他词的变分分布会影响响应变量，考虑高斯响应变量的$\mathrm{E}[\log p(y \vert \bar{z}, \eta, \delta)]$，它是这个文档的ELBO的一部分，变分分布$q(z_n)$在期望的残差平方和$\mathrm{E}\left[\left(y-\eta^{\top} \bar{z}\right)^{2}\right]$中起作用，而（30）式的更新也帮助减小期望的残差平方和

此外，上式的更新取决于除第$j$个词外其他所有词的变分参数$\phi_{-j}$，因此sLDA和LDA不同，$\phi_j$不能并行更新

**参数估计（17）式中变分EM算法中拟合GLM参数时$\mathrm{E}\_{q}\left[\mu\left(\eta^{\top} \bar{z}\_{d}\right) \bar{z}\_{d}\right]$的计算**

令$y=y_{1: D}$表示语料库中各文档的响应变量组成的向量，$X$是$D \times K$维矩阵，每一行为向量$\bar z_d$，令语料库的ELBO对$\eta$的偏导数为0，则有

$$
\mathrm{E}\left[X^{\top} X\right] \eta=\mathrm{E}[X]^{\top} y \quad \Rightarrow \quad \hat{\eta}_{\text {new }} \leftarrow\left(\mathrm{E}\left[X^{\top} X\right]\right)^{-1} \mathrm{E}[X]^{\top} y\tag{31}
$$

注意到$\mathrm E[X]$的第d行就是（12）式中的$\mathrm E[\bar z_d]$，因此$$\mathrm{E}\left[X^{\top} X\right]=\sum_{d} \mathrm{E}\left[\bar{z}_{d} \bar{z}_{d}^{\top}\right]\tag{32}$$

其中每一项在E步都已经确定，见（27）式。

接下来计算$\delta$的更新

$$
\frac{\partial h\left(y_{d}, \delta\right) / \partial \delta}{h\left(y_{d}, \delta\right)}=-\frac{1}{2 \delta}\tag{33}
$$

由（18）式，

$$
\hat{\delta}_{\mathrm{new}} \leftarrow \frac{1}{D}\left\{y^{\top} y-y^{\top} \mathrm{E}[X]\left(\mathrm{E}\left[X^{\top} X\right]\right)^{-1} \mathrm{E}[X]^{\top} y\right\}\tag{34}
$$

上式类似对$y$和$\mathrm E[X]$建立OLS回归的残差平方和的$\frac1D$倍，但是注意到上式中为$\mathrm{E}\left[X^{\top} X\right]$而不是$\mathrm{E}[X]^{\top} \mathrm{E}[X]$，因此$\delta$的更新方程不是$y$的OLS回归。

> 对$y$和$\mathrm E[X]$建立OLS回归，回归系数为$\left(\mathrm{E}[X]^{\top} \mathrm{E}[X]\right)^{-1} \mathrm{E}[X]^{\top} y$，残差平方和为
> 
> $$
> \begin{aligned}(y-\mathrm E[X]\left(\mathrm{E}[X]^{\top} \mathrm{E}[X]\right)^{-1} \mathrm{E}[X]^{\top} y)^\top(y-\mathrm E[X]\left(\mathrm{E}[X]^{\top} \mathrm{E}[X]\right)^{-1} \mathrm{E}[X]^{\top} y)\\=y^{\top} y-y^{\top} \mathrm{E}[X]\left(\mathrm{E}\left[X]^{\top}\mathrm E[ X\right]\right)^{-1} \mathrm{E}[X]^{\top} y\end{aligned}
> $$
> 

**预测（23）式中响应变量的预测值$\mathrm{E}\_{q}\left[\mu\left(\eta^{\top} \bar{z}\_{d}\right)\right]$的计算**：

$$
\mathrm{E}\left[Y \vert w_{1: N}, \alpha, \beta_{1: K}, \eta, \delta\right] \approx \eta^{\top} \mathrm{E}[\bar{z}]\tag{35}
$$

### 泊松分布

响应变量服从过度离散的泊松分布（overdispersed Poisson）时，密度函数为

$$
p(y \vert \lambda, \delta)=\frac{1}{y !} \lambda^{y / \delta} \exp \{-\lambda / \delta\}\tag{36}
$$

可以将其写成如下过度离散的指数族（overdispersed exponential family）形式：

$$
p(y \vert \lambda)=\frac{1}{y !} \exp \left\{\frac{y \log \lambda-\lambda}{\delta}\right\}\tag{37}
$$

这里自然参数$\log \lambda=\eta^{\top} \bar{z}$，$h(y, \delta)=1 / y !$，且有

$$
A\left(\eta^{\top} \bar{z}\right)=\mu\left(\eta^{\top} \bar{z}\right)=\exp \left\{\eta^{\top} \bar{z}\right\}\tag{38}
$$

**变分推断（5）式中每个文档的ELBO中$\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]$的计算：**

$$
\begin{aligned} \mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right] &=\mathrm{E}\left[\exp \left\{(1 / N) \sum_{n=1}^{N} \eta^{\top} z_{n}\right\}\right] \\ &=\prod_{n=1}^{N} \mathrm{E}\left[\exp \left\{(1 / N) \eta^{\top} z_{n}\right\}\right] \end{aligned}\tag{39}
$$

其中，

$$
\begin{aligned} \mathrm{E}\left[\exp \left\{(1 / N) \eta^{\top} z_{n}\right\}\right] &=\sum_{i=1}^{K} \phi_{n, i} \exp \left\{\eta_{i} / N\right\}+\left(1-\phi_{n, i}\right) \\ &=K-1+\sum_{i=1}^{K} \phi_{n, i} \exp \left\{\eta_{i} / N\right\} \end{aligned}\tag{40}
$$

> 上式第一行为$z_{n,i}$为1，对应的值为$\exp \left\\{\eta_{i} / N\right\\}$，概率为$\phi_{n,i}$及$j\neq i$时$z_{n,j}=0$，对应的值为1，总概率为$1-\phi_{n,i}$的和

**变分推断（14）式中变分参数$\phi_n$的更新中$\frac{\partial}{\partial \phi_{n}}\left\\{\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]\right\\}$的计算**

令$C=\mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]$，$C_{-n}$表示C中除去第$n$项的结果，则有

$$
\frac{\partial}{\partial \phi_{n}} \mathrm{E}\left[A\left(\eta^{\top} \bar{z}\right)\right]=C_{-n} \exp \{\eta / N\}\tag{41}
$$

将这个梯度代入（14）式得到$\phi_n$的更新

$$
\phi_{j}^{\mathrm{new}} \propto \exp \left\{\mathrm{E}[\log \theta]+\mathrm{E}\left[\log p\left(w_{j} \vert \beta_{1: K}\right)\right]+\left(\frac{y}{N \delta}\right) \eta-\frac{1}{\delta}C_{-n} \exp \{\eta / N\}\right\}\tag{42}
$$

向量的指数等于这个向量的每个分量的指数组成的向量，成比例意味着每个新的$\phi_j$被重新归一化为1

**参数估计（17）式中变分EM算法中拟合GLM参数时$\mathrm{E}\_{q}\left[\mu\left(\eta^{\top} \bar{z}\_{d}\right) \bar{z}\	_{d}\right]$的计算**

$$
\begin{aligned} \mathrm{E}\left[\mu\left(\eta^{\top} \bar{z}\right) \bar{z}\right] &=\frac{1}{N} \sum_{n=1}^{N} \mathrm{E}\left[\mu\left(\eta^{\top} \bar{z}\right) z_{n}\right] \\ &=\exp \{\eta / N\}\circ \frac{1}{N} \sum_{n=1}^{N} C_{-n} \phi_{n} \end{aligned}\tag{43}
$$

> 
> $$
> \begin{aligned} \mathrm{E}\left[\mu\left(\eta^{\top} \bar{z}\right) \bar{z}\right] &=\frac{1}{N} \sum_{n=1}^{N} \mathrm{E}\left[\mu\left(\eta^{\top} \bar{z}\right) z_{n}\right] \\&=\frac1N\sum_{n=1}^N\mathrm E\left[\left\{\prod_{i=1}^N\exp \left\{\frac1N  \eta^{\top} z_{i}\right\}\right\}z_n\right]\\&=\frac1N\sum_{n=1}^NC_{-n}\exp \{\eta / N\}\circ \phi_n\\ &=\exp \{\eta / N\}\circ \frac{1}{N} \sum_{n=1}^{N} C_{-n} \phi_{n}\end{aligned}
> $$
> 

我们不能找到$\eta$更新的解析解，但是可以用凸优化算法，梯度为

$$
\frac{\partial \mathcal{L}}{\partial \eta}=\frac{1}{\delta}\left(\sum_{d} \mathrm{E}_{d}\left[\bar{z}_{d}\right] y_{d}-\sum_{d} \exp \left\{\eta / N_{d}\right\}\circ \frac{1}{N_{d}} \sum_{n} C_{d,-n} \phi_{d, n}\right)\tag{44}
$$

接下来计算$\delta$的更新，$h(y, \delta)=1 / y !$对$\delta$的导数为0， 因此M步中$\delta$是确定的

$$
\hat{\delta}_{\mathrm{new}} \rightarrow \frac{\sum_{d} \hat{\eta}_{\mathrm{new}}^{\top} \mathrm{E}\left[\bar{z}_{d}\right] y_{d}}{\sum_{d} \mathrm{E}\left[A\left(\hat{\eta}_{\text {new }}^{\top} \bar{z}_{d}\right)\right]}\tag{45}
$$

**预测（23）式中响应变量的预测值$\mathrm{E}\_{q}\left[\mu\left(\eta^{\top} \bar{z}\_{d}\right)\right]$的计算**：

由于$\mu(\cdot)$和$A(\cdot)$相等，可以参见（39）式

### 一般的响应变量分布
对于一般的指数族分布的响应变量，可以采用多元delta方法近似获得难以求解的期望。

> **delta方法**<br>
假定统计量$X_n$是参数$\theta$的一个估计，但是我们感兴趣的是$\phi(\theta)$，一个很自然的想法是用$\phi(X_n)$来估计，考虑$\phi(X_n)$的渐进性质<br>
> **一元delta方法**：如果一列随机变量$X_n$满足$\sqrt{n}\left[X_{n}-\theta\right] \rightarrow N\left(0, \sigma^{2}\right), n \rightarrow \infty$，其中$\theta,\sigma^2$为有限的常数，那么
> 
> $$
> \sqrt{n}\left[g\left(X_{n}\right)-g(\theta)\right] \rightarrow N\left(0, \sigma^{2}\left[g^{\prime}(\theta)\right]^{2}\right), n \rightarrow \infty
> $$
> 
> 其中$g(\theta)$满足$g'(\theta)$存在且取值不为零<br>
**多元delta方法**：设$\mathrm{g}\_{j}(1 \leq j \leq m)$都是$k$变元函数，有一阶全微分，$g=\left(g\_{1}, \ldots, g\_{m}\right)^{\prime}$，又$\xi\_{n}=\left(\xi\_{1 n}, \ldots, \xi\_{k n}\right)^{\prime}(n \geq 1)$为一串随机变量满足条件$\sqrt{n}\left(\xi\_{n}-a\right) \rightarrow N(0, B), n \rightarrow \infty$，这里$a=\left(a\_{1}, \dots, a\_{k}\right)^{\prime}$，$B \geq 0$为K阶常方阵，
> 
> $$
>  \sqrt{n}\left(g\left(\xi_{n}\right)-g(a)\right) \rightarrow N\left(0, C B C^{\prime}\right), n \rightarrow \infty
> $$
> 
> 其中$C$为$m\times k$矩阵，其$(i,j)$元为$\frac{\partial g_{i}}{\partial u_{j}}\vert_{u=a}$



## 结果
利用五折交叉验证评估预测质量，采用两种方法测量误差：
- 折外预测和真实值之间的相关性
- 折外响应值的变异分数$predictive\;R^2$ 

$$
\mathrm{pR}^{2}=1-\frac{\left.\sum(y-\hat{y})^{2}\right)}{\left.\sum(y-\bar{y})^{2}\right)}
$$

我们将sLDA与先使用LDA再对响应变量和$\bar{\phi}_{d}$进行回归进行了比较，并查看了不同主题数下的情形，结果说明sLDA对数据的预测做出了改进。结果如下：

![Alt](https://img-blog.csdnimg.cn/20191226151819859.png#pic_center)
<center> 图5 电影评论的预测结果对比 </center><br>

![Alt](https://img-blog.csdnimg.cn/2019122615234411.png#pic_center)
<center> 图6 第109届参议会修正案的预测结果对比 </center><br>

![Alt](https://img-blog.csdnimg.cn/20191226152553295.png#pic_center)
<center> 图7 第110届参议会修正案的预测结果对比 </center><br>

最后，我们将sLDA与LASSO进行比较。利用每个文档在词汇表上的经验分布作为LASSO的协变量，比较LASSO在不同复杂系数下达到的最高的$pR^2$和sLDA在不同主题数下达到的最高的$pR^2$。电影评论数据、109届参议院修正案和110届参议院修正案中，sLDA和LASSO达到的最高的$pR^2$分别为0.432和0.426，0.27和0.15，0.23和0.16，sLDA效果更好。

> LASSO是L1正则化最小二乘回归，是针对高维数据常用的预测方法

# 5. 讨论
未来可以改进的方向有以下四个：

- sLDA的半监督版本：对于常见的部分标记的语料库，只需要省略（14）式中的最后两项，并且只对有标记的文档计算（17）式和（18）式即可
- 如果每个文档都可以观测到一个额外的固定维的协向量，可以将其纳入线性预测中
- 本文中提到的处理响应变量的方法也可以纳入其它LDA变体中，如作者主题模型（Rosen-Zvi et al., 2004[3]）、群体遗传学模型（Pritchard et al., 2000[4]）和调查数据模型（Erosheva, 2002[5]）

# 6. 参考文献
1. Pang, B. and Lee, L. (2005). Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the Association of Computational Linguistics.
2. Clinton, J., Jackman, S., and Rivers, D. (2004). The statistical analysis of roll call data. American Political Science Review, 98(2):355-370.
3. Rosen-Zvi, M., Griffiths, T., Steyvers, M., and Smith, P. (2004). The author-topic model for authors and documents. In Proceedings of the 20th Conference on Uncertainty in Articial Intelligence, pages 487-494. AUAI Press.
4. Pritchard, J., Stephens, M., and Donnelly, P. (2000). Inference of population structure using multilocus genotype data. Genetics, 155:945-959.
5. Erosheva, E. (2002). Grade of membership and latent structure models with application to disability survey data. PhD thesis, Carnegie Mellon University, Department of Statistics.