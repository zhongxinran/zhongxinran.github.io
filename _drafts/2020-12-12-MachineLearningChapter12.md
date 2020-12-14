@[TOC]
## 12.1 基础知识
计算学习理论（computational learning theory）研究的是关于通过计算进行学习的理论，即关于机器学习的理论基础，其目的是分析学习任务的困难本质，为学习算法提供理论保证，并根据分析结果指导算法设计

**研究问题及符号表示**：

本章讨论二分类问题，给定样例集$D=\left\{\left(\boldsymbol{x}_{1}, y_{1}\right),\left(\boldsymbol{x}_{2}, y_{2}\right), \ldots,\left(\boldsymbol{x}_{m}, y_{m}\right)\right\}, \boldsymbol{x}_{i} \in \mathcal{X},y_{i} \in \mathcal{Y}=\{-1,+1\}$，假设$\mathcal X$中所有样本独立同分布，服从一个隐含未知的分布$\mathcal D$

令$h$为从$\mathcal X$到$\mathcal Y$的一个映射，其泛化误差为$$E(h ; \mathcal{D})=P_{\boldsymbol{x} \sim \mathcal{D}}(h(\boldsymbol{x}) \neq y)$$ 在$D$上的经验误差为$$\widehat{E}(h ; D)=\frac{1}{m} \sum_{i=1}^{m} \mathbb{I}\left(h\left(\boldsymbol{x}_{i}\right) \neq y_{i}\right)$$将上两式分别简记为$E(h),\widehat E(h)$，由于$D$是$\mathcal D$的独立同分布采样，$h$的经验误差的期望等于其泛化误差，令$\epsilon$为$E(h)$的上限，即$E(h) \leqslant \epsilon$，用$\epsilon$表示预先设定的学得模型所应满足的误差要求，也称为误差参数

本章后面部分将研究经验误差与泛化误差之间的逼近成都，若$h$在数据集D上的经验误差为0，则称$h$与$D$一致，否则称其与$D$不一致，对任意两个映射$h_{1}, h_{2} \in \mathcal{X} \rightarrow \mathcal{Y}$，通过其不合（disagreement）度量他们之间的差别$$d\left(h_{1}, h_{2}\right)=P_{\boldsymbol{x} \sim \mathcal{D}}\left(h_{1}(\boldsymbol{x}) \neq h_{2}(\boldsymbol{x})\right)$$

**几个常用的不等式**（此处的$\mathbb E$指期望）：
- Jensen不等式：对任意凸函数$f(x)$，有$$f(\mathbb{E}(x)) \leqslant \mathbb{E}(f(x))$$
- Hoeffding不等式：若$x_{1}, x_{2}, \ldots, x_{m}$为m个独立随机变量，且满足$0 \leqslant x_{i} \leqslant 1$，则对任意$\epsilon>0$，有$$P\left(\frac{1}{m} \sum_{i=1}^{m} x_{i}-\frac{1}{m} \sum_{i=1}^{m} \mathbb{E}\left(x_{i}\right) \geqslant \epsilon\right) \leqslant \exp \left(-2 m \epsilon^{2}\right)$$ $$P\left(\left|\frac{1}{m} \sum_{i=1}^{m} x_{i}-\frac{1}{m} \sum_{i=1}^{m} \mathbb{E}\left(x_{i}\right)\right| \geqslant \epsilon\right) \leqslant 2 \exp \left(-2 m \epsilon^{2}\right)$$
- Mcdiarmid不等式：若$x_{1}, x_{2}, \ldots, x_{m}$为m个独立随机变量，且对任意$1\leq i \leq m$，函数$f$满足$$\sup _{x_{1}, \ldots, x_{m}, x_{i}^{\prime}}\left|f\left(x_{1}, \ldots, x_{m}\right)-f\left(x_{1}, \ldots, x_{i-1}, x_{i}^{\prime}, x_{i+1}, \ldots, x_{m}\right)\right| \leqslant c_{i}$$则对任意$\epsilon>0$，有$$P\left(f\left(x_{1}, \ldots, x_{m}\right)-\mathbb{E}\left(f\left(x_{1}, \ldots, x_{m}\right)\right) \geqslant \epsilon\right) \leqslant \exp \left(\frac{-2 \epsilon^{2}}{\sum_{i} c_{i}^{2}}\right)$$ $$P\left(\left|f\left(x_{1}, \ldots, x_{m}\right)-\mathbb{E}\left(f\left(x_{1}, \ldots, x_{m}\right)\right)\right| \geqslant \epsilon\right) \leqslant 2 \exp \left(\frac{-2 \epsilon^{2}}{\sum_{i} c_{i}^{2}}\right)$$
## 12.2 PAC学习
计算学习理论中最基本的是概率近似正确（Probably Approximately Correct, PAC）学习理论
- 概念（concept）：从样本空间$\mathcal X$到标记空间$\mathcal Y$的映射，用c表示
- 目标概念：若对任何样例有$c(\boldsymbol{x})=y$成立，则称c为目标概念
- 概念类：所有我们希望学得的目标概念所构成的集合称为概念类（concept class），用$\mathcal C$表示
- 假设空间（hypothesis space）：给定学习算法$\mathcal{L}$，它考虑的所有可能概念的集合称为假设空间，用$\mathcal H$表示，$\mathcal H$和$\mathcal C$通常是不同的
- 假设（hypothesis）：对$h\in \mathcal H$，称之为假设
- 可分的（separable）/一致的（consistent）：若目标概念$c\in \mathcal H$，则$\mathcal H$中存在假设能将所有示例按与真实标记一致的方式完全分开，则称该问题对学习算法$\mathcal L$是可分的/一致的
- 可分的（non-separable）/一致的（non-consistent）：若目标概念$c \notin \mathcal{H}$，则$\mathcal H$中不存在任何假设能将所有示例完全正确分开，则称该问题对学习算法$\mathcal L$是不可分的/不一致的

我们希望基于学习算法$\mathcal L$学得的模型所对应的的假设$h$应尽可能接近目标概念$c$，之所以不是精确学到$c$，是因为机器学习过程会受到很多因素的制约，如在数据集D上存在等效的假设无法进一步区分，或采样得到D时会有一定的偶然性等，因此我们希望以较大的概率学得误差满足预设上限的模型

> **定义：PAC辨识**（PAC Identity）对$0<\epsilon, \delta<1$，所有$c\in \mathcal C$和分布$\mathcal D$，若存在学习算法$\mathcal L$，其输出假设$h\in \mathcal H$满足$$P(E(h) \leqslant \epsilon) \geqslant 1-\delta$$则称学习算法$\mathcal L$能从假设空间$\mathcal H$中PAC辨识概念类$\mathcal C$

这样的学习算法$\mathcal L$能以较大的概率（至少$1-\delta$）学得目标概念c的近似（误差最多为$\epsilon$），在此基础上可定义：

> **定义：PAC可学习**（PAC Learnable）令m表示从分布$\mathcal D$中独立同分布采样得到的样例数目，$0<\epsilon, \delta<1$，对所有分布$\mathcal D$，若存在学习算法$\mathcal L$和多项式函数$poly(·,·,·,·)$，使得对于任何$m \geqslant \operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))$，$\mathcal L$能从假设空间$\mathcal H$中PAC辨识概念类$\mathcal C$，则称概念类$\mathcal C$对假设空间$\mathcal H$而言是PAC可学习的

对计算机算法来说，必然要考虑时间复杂度，因此定义：

> **定义：PAC学习算法（PAC Learning Algorithm）** 若学习算法$\mathcal L$使概念类$\mathcal C$为PAC可学习的，且$\mathcal L$的运行时间也是多项式函数$\operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))$，则称概念类$\mathcal C$是高效PAC可学习（efficiently PAC learnable），称$\mathcal L$为概念类$\mathcal C$的PAC学习算法

假定学习算法$\mathcal L$处理每个样本的时间为常数，则$\mathcal L$的时间复杂度等价于样本复杂度，因此定义：

> **定义：样本复杂度**（Sample Complexity）满足PAC学习算法$\mathcal L$所需的$m \geqslant \operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))$中最小的m，称为学习算法$\mathcal L$的样本复杂度

PAC学习给出了一个抽象地刻画机器学习能力的框架，基于这个框架能对很多重要问题进行理论探讨，例如研究某任务在什么样的条件下可学得好的模型？某算法在什么样的条件下可进行有效的学习？需多少训练样例才能获得较好的模型？

PAC学习中一个关键因素是假设空间$\mathcal H$的复杂度，若在PAC学习中假设空间与概念类完全相同，即$\mathcal H=\mathcal C$，称为恰PAC学习（properly PAC learnable），直观地看这意味着学习算法的能力与学习任务恰好匹配，但在实际任务中不太可能。一般而言，$\mathcal H$越大，其包含任意目标概念的可能性越大，但从中找到某个具体目标概念的难度也越大，$|\mathcal H|$有限时，我们称$\mathcal H$为有限假设空间，否则称为无限假设空间
## 12.3 有限假设空间
#### 12.3.1 可分情形
D中样例标记都是由目标概念c赋予的，可分情形意味着c存在于假设空间中，那么任何与训练集D上出现标记错误的假设肯定不是目标概念从，于是我们只需保留与D一致的假设，剔除与D不一致的假设即可，若训练集D足够大，可不断借助D中的样例剔除不一致的假设，直至$\mathcal H$中仅剩下一个假设为止，即为目标概念c。通常情形下由于训练集规模有限，假设空间中可能存在不止一个与D一致的等效假设，无法进一步区分其优劣。对PAC学习来说，只要训练集D的规模能使学习算法$\mathcal L$以概率$1-\delta$找到目标假设的$\epsilon$近似即可

我们先估计泛化误差大于$\epsilon$但在训练集上仍表现完美的假设出现的概率，假设h的泛化误差大于$\epsilon$，对在分布$\mathcal D$上随机采样而得的任何样例$(\boldsymbol x,y)$，有$$\begin{aligned} P(h(\boldsymbol{x})=y) &=1-P(h(\boldsymbol{x}) \neq y) \\ &=1-E(h) \\ &<1-\epsilon \end{aligned}$$因此，h与D表现一致的概率为$$\begin{aligned} P\left(\left(h\left(\boldsymbol{x}_{1}\right)=y_{1}\right) \wedge \ldots \wedge\left(h\left(\boldsymbol{x}_{m}\right)=y_{m}\right)\right) &=(1-P(h(\boldsymbol{x}) \neq y))^{m} \\ &<(1-\epsilon)^{m} \end{aligned}$$我们实现并不知道$\mathcal L$会输出$\mathcal H$中的哪个假设，但仅需保证泛化误差大于$\epsilon$，且在训练集上表现完美的所有假设出现概率之和不大于$\delta$即可$$\begin{aligned} P(h \in \mathcal{H}: E(h)>\epsilon \wedge \widehat{E}(h)=0) &<|\mathcal{H}|(1-\epsilon)^{m} \\ &<|\mathcal{H}| e^{-m \epsilon} \end{aligned}$$令$|\mathcal{H}| e^{-m \epsilon} \leqslant \delta$，得$$m \geqslant \frac{1}{\epsilon}\left(\ln |\mathcal{H}|+\ln \frac{1}{\delta}\right)$$

由此可知，有限假设空间$\mathcal H$都是PAC可学习的，所需样例数如上，输出假设h的泛化误差随样例数的增多而收敛到0，收敛速度为$O\left(\frac{1}{m}\right)$
#### 12.3.2 不可分情形
不可分情形中，$\mathcal H$中的任意一个假设都会在训练集上出现或多或少的错误，由Hoeffding不等式：

> **引理12.1** 若训练集D包含m个从分布$\mathcal D$上独立同分布采样而得的样例，$0<\epsilon<1$，则对任意$h \in \mathcal{H}$，有$$\begin{array}{l}{P(\widehat{E}(h)-E(h) \geqslant \epsilon) \leqslant \exp \left(-2 m \epsilon^{2}\right)} \\ {P(E(h)-\widehat{E}(h) \geqslant \epsilon) \leqslant \exp \left(-2 m \epsilon^{2}\right)} \\ {P(|E(h)-\widehat{E}(h)| \geqslant \epsilon) \leqslant 2 \exp \left(-2 m \epsilon^{2}\right)}\end{array}$$

> **推论12.1** 若训练集D包含m个从分布$\mathcal D$上独立同分布采样而得的样例，$0<\epsilon<1$，则对任意$h \in \mathcal{H}$，下式以至少$1-\delta$的概率成立$$\widehat{E}(h)-\sqrt{\frac{\ln (2 / \delta)}{2 m}} \leqslant E(h) \leqslant \widehat{E}(h)+\sqrt{\frac{\ln (2 / \delta)}{2 m}}$$

上述推论表明，样例数目较大时，h的经验误差是其泛化误差很好的近似

> **定理12.1** 若$\mathcal H$为有限假设空间，$0<\delta<1$，则对任意$h\in \mathcal H$，有$$P(|E(h)-\widehat{E}(h)| \leqslant \sqrt{\frac{\ln |\mathcal{H}|+\ln (2 / \delta)}{2 m}}) \geqslant 1-\delta$$

证明：令$h_{1}, h_{2}, \ldots, h_{|\mathcal{H}|}$表示假设空间$\mathcal H$中的假设，有$$\begin{aligned} & P(\exists h \in \mathcal{H}:|E(h)-\widehat{E}(h)|>\epsilon) \\=& P\left(\left(\left|E_{h_{1}}-\widehat{E}_{h_{1}}\right|>\epsilon\right) \vee \ldots \vee\left(\left|E_{h_{|\gamma|}}-\widehat{E}_{h_{|| \gamma |}}\right|>\epsilon\right)\right) \\ \leqslant & \sum_{h \in \mathcal{H}} P(|E(h)-\widehat{E}(h)|>\epsilon) \end{aligned}$$由引理12.1可得$$\sum_{h \in \mathcal{H}} P(|E(h)-\widehat{E}(h)|>\epsilon) \leqslant 2|\mathcal{H}| \exp \left(-2 m \epsilon^{2}\right)$$因此，令$\delta=2|\mathcal{H}| \exp \left(-2 m \epsilon^{2}\right)$即得证

显然，当$c\notin \mathcal H$时，学习算法$\mathcal L$无法学得目标概念c的$\epsilon$近似，但是假设空间给定时必存在一个泛化误差最小的假设，以$\arg \min _{h \in \mathcal{H}} E(h)$为目标可将PAC学习推广到不可分情形，称为不可知学习（agnostic learning）

> **定义：不可知PAC学习**（agnostic PAC learnable）令m表示从分布$\mathcal D$中独立同分布采样而得的样例数目，$0<\epsilon,\delta<0$，对所有分布$\mathcal D$，若存在学习算法$\mathcal L$和多项式函数$poly(·,·,·,·)$，使得对于任何$m \geqslant \operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))$，$\mathcal L$能从假设空间$\mathcal H$中输出满足下式的假设h：$$P\left(E(h)-\min _{h^{\prime} \in \mathcal{H}} E\left(h^{\prime}\right) \leqslant \epsilon\right) \geqslant 1-\delta$$则称假设空间$\mathcal H$是不可知PAC可学习的
> 类似地，若学习算法$\mathcal L$的运行时间也是多项式函数$\operatorname{poly}(1 / \epsilon, 1 / \delta, \operatorname{size}(\boldsymbol{x}), \operatorname{size}(c))$，$\mathcal L$，则称假设空间$\mathcal H$是高效不可知PAC可学习的，学习算法$\mathcal L$则称为假设空间$\mathcal H$的不可知PAC学习算法，满足上述要求的最小m称为学习算法$\mathcal L$的样本复杂度
## 12.4 VC维
现实学习任务所面临的通常是无限假设空间，例如实数域中的所有区间、$\mathbb R^d$空间中的所有线性超平面，欲对此种情形的可学习性进行研究，需度量假设空间的复杂度，最常见的办法是考虑假设空间的VC维（Vapnik-Chervonenkis dimension）

给定假设空间$\mathcal H$和示例集$D=\left\{\boldsymbol{x}_{1}, \boldsymbol{x}_{2}, \ldots, \boldsymbol{x}_{m}\right\}$，$\mathcal H$中每个假设h都能对D中示例赋予标记，标记结果可表示为$$\left.h\right|_{D}=\left\{\left(h\left(\boldsymbol{x}_{1}\right), h\left(\boldsymbol{x}_{2}\right), \ldots, h\left(\boldsymbol{x}_{m}\right)\right)\right\}$$随着m增大，$\mathcal H$中所有假设对D中的示例所能赋予的可能结果数也会增大

> **定义：增长函数（growth function）** 对所有$m \in \mathbb{N}$，假设空间$\mathcal{H}$的增长函数为$$\Pi_{\mathcal{H}}(m)=\max _{\left\{\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{m}\right\} \subseteq \mathcal{X}}\left|\left\{\left(h\left(\boldsymbol{x}_{1}\right), \ldots, h\left(\boldsymbol{x}_{m}\right)\right) | h \in \mathcal{H}\right\}\right|$$

增长函数表示假设空间$\mathcal H$对m个示例所能赋予标记的最大可能结果数，这个数越大，$\mathcal H$的表示能力越强，对学习任务的适应能力也越强，因此增长函数描述了假设空间$\mathcal H$的表示能力，由此反映出假设空间的复杂度，可用其估计经验误差与泛化误差之间的关系：

> **定理12.2** 对假设空间$\mathcal H$，$m \in \mathbb{N}, 0<\epsilon<1$和任意$h\in \mathcal H$，有$$P(|E(h)-\widehat{E}(h)|>\epsilon) \leqslant 4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right)$$

> **定义：对分（dichotomy）与打散（shattering）** 对二分类问题来说，$\mathcal H$中的假设对D中示例赋予标记的每种可能结果称为对D的一种对分；若假设空间$\mathcal H$能实现示例集D上的所有对分，即$\Pi_{\mathcal{H}}(m)=2^{m}$，则称示例集D能被假设空间$\mathcal H$打散

> **定义：VC维** 假设空间$\mathcal H$的VC维是能被$\mathcal H$打散的最大示例集的大小，即$$\mathrm{VC}(\mathcal{H})=\max \left\{m: \Pi_{\mathcal{H}}(m)=2^{m}\right\}$$

$\mathrm{VC}(\mathcal{H})=d$表明存在大小为d的示例集能被假设空间$\mathcal H$打散，不存在任何大小为d+1的示例集能被其打散，但不是所有大小为d的示例集都能被假设空间$\mathcal H$打散，VC维的定义与数据分布$\mathcal D$无关，在数据分布未知时仍能计算VC维。下面为VC维的两个例子：
- 实数域中的区间：令$\mathcal H$表示实数域中所有闭区间构成的集合$\left\{h_{[a, b]}: a, b \in \mathbb{R}, a \leqslant b\right\}, \mathcal{X}=\mathbb{R}$，对$x\in \mathcal X$，若$x\in [a,b]$，则$h_{[a, b]}(x)=+1$，否则$h_{[a, b]}(x)=-1$，则该假设空间的VC维为2，因为大小为2的示例集能被打散，而大小为3的示例集$\{x_3,x_4,x_5\}$不能被打散，因为对$x_3<x_4<x_5$无法实现对分结果$\left\{\left(x_{3},+\right),\left(x_{4},-\right),\left(x_{5},+\right)\right\}$
- 二维实平面上的线性化分：令$\mathcal H$表示二维实平面上所有线性划分构成的集合，$\mathcal X=\mathbb R ^2$，则该假设空间的VC维为3。如下图所示，大小为3的示例集可被打散，但不存在大小为4的示例集可被打散![在这里插入图片描述](https://img-blog.csdnimg.cn/20191125163052745.png#pic_center)

VC维与增长函数有密切联系：
> **引理12.2** 若假设空间$\mathcal H$的VC维为d，则对任意$m\in \mathbb N$有$$\Pi_{\mathcal{H}}(m) \leqslant \sum_{i=0}^{d}\left(\begin{array}{c}{m} \\ {i}\end{array}\right)$$

证明：由数学归纳法证明，当$m=1,d=0$或$d=1$时，定理成立；假设定理对$(m-1,d-1)$和$(m-1,d)$成立，令$D=\left\{x_{1}, x_{2}, \ldots, x_{m}\right\},D^{\prime}=\left\{x_{1}, x_{2}, \ldots, x_{m-1}\right\}$，有$$\mathcal{H}_{| D}=\left\{\left(h\left(\boldsymbol{x}_{1}\right), h\left(\boldsymbol{x}_{2}\right), \ldots, h\left(\boldsymbol{x}_{m}\right)\right) | h \in \mathcal{H}\right\}$$ $$\mathcal{H}_{| D^{\prime}}=\left\{\left(h\left(\boldsymbol{x}_{1}\right), h\left(\boldsymbol{x}_{2}\right), \ldots, h\left(\boldsymbol{x}_{m-1}\right)\right) | h \in \mathcal{H}\right\}$$任何假设$h\in \mathcal H$对$\boldsymbol x_m$的分类结果或为+1或为-1，因此任何出现在$\mathcal{H}_{| D^{\prime}}$中的串都会在$\mathcal{H}_{| D}$中出现一次或两次，令$\mathcal{H}_{D^{\prime} | D}$表示在$\mathcal{H}_{| D}$中出现两次的$\mathcal{H}_{| D^{\prime}}$中串组成的集合，即$$\begin{aligned} \mathcal{H}_{D^{\prime} | D}=&\left\{\left(y_{1}, y_{2}, \ldots, y_{m-1}\right) \in \mathcal{H}_{| D^{\prime}} | \exists h, h^{\prime} \in \mathcal{H}\right. \\ & \left.\left(h\left(\boldsymbol{x}_{i}\right)=h^{\prime}\left(\boldsymbol{x}_{i}\right)=y_{i}\right) \wedge\left(h\left(\boldsymbol{x}_{m}\right) \neq h^{\prime}\left(\boldsymbol{x}_{m}\right)\right), 1 \leqslant i \leqslant m-1\right\} \end{aligned}$$

考虑到$\mathcal{H}_{D^{\prime} | D}$中的串在$\mathcal{H}_{| D}$中出现了两次，但在$\mathcal{H}_{| D^{\prime}}$中仅出现了一次，有$$\left|\mathcal{H}_{| D}\right|=\left|\mathcal{H}_{| D^{\prime}}\right|+\left|\mathcal{H}_{D^{\prime} | D}\right|$$ $D'$的大小为$m-1$，由假设可得$$\left|\mathcal{H}_{| D^{\prime}}\right| \leqslant \Pi_{\mathcal{H}}(m-1) \leqslant \sum_{i=0}^{d}\left(\begin{array}{c}{m-1} \\ {i}\end{array}\right)$$令$\mathcal Q$表示能被$\mathcal{H}_{D^{\prime} | D}$打散的集合，由$\mathcal{H}_{D^{\prime} | D}$定义可知$Q \cup\left\{\boldsymbol{x}_{m}\right\}$必能被$\mathcal{H}_{| D}$打散，由于$\mathcal H$的VC维为d，因此$\mathcal{H}_{D^{\prime} | D}$的VC维最大为d-1，于是有$$\left|\mathcal{H}_{D^{\prime} | D}\right| \leqslant \Pi_{\mathcal{H}}(m-1) \leqslant \sum_{i=0}^{d-1}\left(\begin{array}{c}{m-1} \\ {i}\end{array}\right)$$由以上三式，可得$$\begin{aligned}\left|\mathcal{H}_{| D}\right| & \leqslant \sum_{i=0}^{d}\left(\begin{array}{c}{m-1} \\ {i}\end{array}\right)+\sum_{i=0}^{d-1}\left(\begin{array}{c}{m-1} \\ {i}\end{array}\right) \\ &=\sum_{i=0}^{d}\left(\left(\begin{array}{c}{m-1} \\ {i}\end{array}\right)+\left(\begin{array}{c}{m-1} \\ {i-1}\end{array}\right)\right) \\ &=\sum_{i=0}^{d}\left(\begin{array}{c}{m} \\ {i}\end{array}\right) \end{aligned}$$由集合D的任意性，引理得证

从引理12.2可计算出增长函数的上界：
> **推论12.2** 若假设空间$\mathcal H$的VC维为d，则对任意整数$m\geq d$有$$\Pi_{\mathcal{H}}(m) \leqslant\left(\frac{e \cdot m}{d}\right)^{d}$$

证明：$$\begin{aligned} \Pi_{\mathcal{H}}(m) & \leqslant \sum_{i=0}^{d}\left(\begin{array}{c}{m} \\ {i}\end{array}\right) \\ & \leqslant \sum_{i=0}^{d}\left(\begin{array}{c}{m} \\ {i}\end{array}\right)\left(\frac{m}{d}\right)^{d-i} \\ &=\left(\frac{m}{d}\right)^{d} \sum_{i=0}^{d}\left(\begin{array}{c}{m} \\ {i}\end{array}\right)\left(\frac{d}{m}\right)^{i} \\ & \leqslant\left(\frac{m}{d}\right)^{d} \sum_{i=0}^{m}\left(\begin{array}{c}{m} \\ {i}\end{array}\right)\left(\frac{d}{m}\right)^{i} \\ &=\left(\frac{e \cdot m}{d}\right)^{d} \end{aligned}$$

根据推论12.2和定理12.2可得基于VC维的泛化误差界：
> **定理12.3** 若假设空间$\mathcal H$的VC维为d，则对任意$m>d, 0<\delta<1$和$h\in \mathcal H$有$$P\left(E(h)-\widehat{E}(h) \leqslant\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}\right)\geqslant 1-\delta$$

证明：令$4 \Pi_{\mathcal{H}}(2 m) \exp \left(-\frac{m \epsilon^{2}}{8}\right) \leqslant 4\left(\frac{2 e m}{d}\right)^{d} \exp \left(-\frac{m \epsilon^{2}}{8}\right)=\delta$，解得$$\epsilon=\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta}}{m}}$$代入定理12.2，于是定理12.3得证

由上述定理可知，泛化误差界只与样例数目m有关，收敛速率为$O\left(\frac{1}{\sqrt{m}}\right)$，与数据分布$\mathcal D$和样例集D无关，因此基于VC维的泛化误差界是分布无关（distribution-free）、数据独立（data-independent）的

令h表示学习算法$\mathcal L$输出的假设，若h满足$$\widehat{E}(h)=\min _{h^{\prime} \in \mathcal{H}} \widehat{E}\left(h^{\prime}\right)$$则称$\mathcal L$为满足经验风险最小化（Empirical Risk Minimization, ERM）原则的算法，因此有下面的定理：
> **定理12.4** 任何VC维有限的假设空间$\mathcal H$都是（不可知）PAC可学习的

证明：假设$\mathcal H$为满足经验风险最小化原则的算法，h为学习算法$\mathcal L$输出的假设，令g表示$\mathcal H$中具有最小泛化误差的假设，即$$E(g)=\min _{h \in \mathcal{H}} E(h)$$令$$\delta^{\prime}=\frac{\delta}{2}$$ $$\sqrt{\frac{\left(\ln 2 / \delta^{\prime}\right)}{2 m}}=\frac{\epsilon}{2}$$由推论12.1可知$$\widehat{E}(g)-\frac{\epsilon}{2} \leqslant E(g) \leqslant \widehat{E}(g)+\frac{\epsilon}{2}$$至少以$1-\frac\delta2$的概率成立，令$$\sqrt{\frac{8 d \ln \frac{2 e m}{d}+8 \ln \frac{4}{\delta^{\prime}}}{m}}=\frac{\epsilon}{2}$$则由定理12.3可知$$P\left(E(h)-\widehat{E}(h) \leqslant \frac{\epsilon}{2}\right) \geqslant 1-\frac{\delta}{2}$$从而可知$$\begin{aligned} E(h)-E(g) & \leqslant \widehat{E}(h)+\frac{\epsilon}{2}-\left(\widehat{E}(g)-\frac{\epsilon}{2}\right) \\ &=\widehat{E}(h)-\widehat{E}(g)+\epsilon \\ & \leqslant \epsilon \end{aligned}$$以至少$1-\delta$的概率成立，又上述诸式可解出m，再由$\mathcal H$的任意性可知定理12.4得证
## 12.5 Rademacher复杂度
啦啦啦
