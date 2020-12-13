---
title: 西瓜书 | 第二章 模型评估与选择
author: 钟欣然
date: 2020-12-8 23:01:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

## 2.1 经验误差与过拟合

#### 2.1.1 一些概念
 - 错误率 error rate：分类错误的样本数/样本总数
 - 精度 accuracy：1-错误率
 - 误差 error：学习器的实际预测输出与样本的真实输出之间的差异
 	- 训练误差 training error / 经验误差 empirical error
 	- 泛化误差 generalization error

#### 2.1.2 过拟合与欠拟合
 引入：分类错误率为0，分类精度为100%的学习器在多数情况下都不好
 - 过拟合 overfitting
 	- 有多种因素可能导致过拟合，最重要的是学习能力过于强大，把训练样本所包含的不太一般的特性都学到了
 	- 无法彻底避免
 - 欠拟合 underfitting
 	- 通常是由学习能力低下造成的
 	- 易克服：如在决策树学习中拓展分支、在神经网络学习中增加训练轮数等

## 2.2 评估方法
对学习器的泛化性能进行评估，需要有效可行的实验估计方法（2.2 评估方法）、衡量模型泛化能力的评价标准（2.3 性能度量）、采用适当的方法对度量结果进行比较（2.4 比较检验）

#### 2.2.1 模型选择 model selection
在现实任务中，我们往往有多种学习算法可供选择，对同一个学习算法使用不同的参数配置时，也会产生不同的模型，因此，产生了模型选择问题。**理想方案**是对候选模型的泛化误差进行评估，选择泛化误差最小的模型

为此，我们需要**测试集**（testing set），并以测试集（testing error）作为泛化误差的近似。通常我们假设测试样本也是从样本真是分布中独立同分布采样而得，但需注意的是，**测试集应该尽可能与训练集互斥**。

#### 2.2.2 留出法 hold-out
将数据集D划分为两个互斥的集合，训练集S和测试集T，使得$D=S\cup T,S\cap T=\varnothing$。

训练集与测试集的划分要**尽可能保持数据分布的一致性**，避免因数据划分过程引入额外的偏差而对最终结果产生影响，可以采用**分层采样**（stratified sampling）（如500个正例500个负例抽取70%个作为训练集，则应包含350个正例350个负例）。

一般要采用**若干次随机划分，重复进行实验评估**后取平均值作为评估结果。

一般将2/3-4/5的样本用于训练，剩下的样本用于测试，测试集至少应含30个样例。
- 训练集包含样本过多
	- 训练出的模型与用D训练出的模型差别较小
	- 评估结果不够稳定准确
- 训练集包含样本过少
	- 被评估的模型与用D训练出的模型差别较大，从而降低了评估结果的保真性（fidelity）
	- 评估结果更加稳定准确
#### 2.2.3 交叉验证法 cross validation
将数据集D划分为k个大小相似的互斥子集，使得$D=D_1\cup D_2\cup \dots \cup D_k,D_i\cap D_j=\varnothing$，k-1个子集的并集作为训练集，余下的子集作为测试集。

每个子集$D_i$都要尽可能保持数据分布的一致性，即从D中分层采样得到。

**p次k折交叉验证**（k-fold cross validation）：交叉验证评估结果的稳定性和保真性很大程度上取决于k，k的常用取值为10，也可以取5或20。需要随机使用不同的划分重复p次，常用10次10折交叉验证。

**留一法 Leave-One-Out，即LOO**
优点：
- 不受随机样本划分方式的影响
- 绝大多数情况下，被实际评估的模型与期望评估的用D训练出的模型很相似，因此，结果比较准确

缺点：
- 在数据集比较大时，训练m个模型的计算开销过大，如需调参则更多
- 估计结果未必永远比其它评估方法更准确

#### 2.2.4 自助法 bootstrapping
**自助采样 bootstrap sampling**
数据集D包含m个样本，进行不放回抽样m次得到D'，样本在m次采样中始终不被采到的概率是$(1-\frac1m)^m$，$\lim_{m\rightarrow\infty}(1-\frac1m)^m={\textstyle\frac1e}\approx0.368$，将D'用作训练集，D\D'用作测试集

实际评估的模型与期望评估的模型都使用m个训练样本，仍有数据总量约1/3的、没在训练集中出现的样本用于测试，这样的测试结果称为包外估计（out-of-bag estimate）

优点：
- 在数据集较小、难以有效划分训练/测试集时很有用
- 能从原始数据集中产生多个不同的训练集，这对集成学习等方法有很大的好处

缺点
- 自助法产生的数据集改变了原始数据集的分布，这会引入估计偏差

**在初始数据量足够时，留出法和交叉验证法更常用。**

#### 2.2.5 调参与最终模型
学习算法的很多参数是在实数范围内取值，对每种参数都配指出模型是不可行的，可行方案是对每个参数选定一个范围和变化步长，如在[0,0.2]范围内以0.05为步长，评估5个参数

#### 2.2.6 算法与参数选定后
学习算法和参数配置选定后，应用数据集D重新训练模型

在实际使用中遇到的数据称为测试数据，在模型评估与选择中用于测试的数据集称为验证集（validation set）

## 2.3 性能度量 performance measure
性能度量反映了任务需求
- 预测任务
	- 最常用的为均方误差
	- 对样例集，$$E(f;D)=\frac1m \sum_{i=1}^m(f(x_i)-y_i)^2$$
	- 更一般地，对于数据分布$\mathcal{D}$，$$E(f;\mathcal{D})=\int_{x\sim \mathcal{D}}(f(x)-y)^2p(x)\operatorname dx$$
- 聚类任务
	- 参见第九章
- 分类任务
	- 如下

#### 2.3.1 错误率 error rate & 精度 accuracy
- **错误率**
	- 对样例集，$$E(f;D)=\frac1m \sum_{i=1}^m\mathbb{I}(f(x_i)\neq y_i)$$
	- 更一般地，对于数据分布$\mathcal{D}$，$E(f;\mathcal{D})=\int_{x\sim \mathcal{D}}\mathbb{I}(f(x)\neq y)p(x)\operatorname dx$
- **精度**
	- 对样例集，$$acc(f;D)=\frac1m \sum_{i=1}^m\mathbb{I}(f(x_i)=y_i)=1-E(f;D)$$
	- 更一般地，对于数据分布$\mathcal{D}$，$$E(f;\mathcal{D})=\int_{x\sim \mathcal{D}}\mathbb{I}(f(x)=y)p(x)\operatorname dx=1-E(f;\mathcal{D})$$

#### 2.3.2 查准率 precision & 查全率 recall
对于二分类问题，可根据真实情况和预测结果将样例分为真正例（true positive）、假正例（false positive）、真反例（true negative）、假反例（false negative）

分类结果混淆矩阵 | 预测为正例 | 预测为负例
  - | - | -
  真实为正例 | TP | FN
  真实为负例 | FP | TN

- **查准率：**$$P=\frac{TP}{TP+FP}$$
- **查全率：**$$R=\frac{TP}{TP+FN}$$
- **查准率-查全率曲线**（P-R曲线）：查准率高时，查全率往往偏低；查全率高时，查准率往往偏低
![P-R曲线](https://img-blog.csdnimg.cn/20191021170506266.png#pic_center)
	- 若一个学习器的P-R曲线完全包住另一个学习器的曲线，则前者的性能更好，如图中A好于C
	- 若两个学习器的P-R曲线有交叉时，如A与B，则较难判断，解决方案有四个：
		- **面积**：比较P-R曲线下面积的大小，在一定程度上衡量了二者相对“双高”的比例
		- **平衡点**（Break-Even Point，即BEP）：查准率=查全率时的取值，越大越好
		- **F1度量**：基于查准率和查全率的调和平均$$\frac1{F1}=\frac12\times (\frac1P+\frac1R)$$ $$F1=\frac{2\times P\times R}{P+R}=\frac{2*TP}{样例总数+TP-TN}$$
		- **$F_\beta$度量**：基于查准率和查全率的加权调和平均$$\frac1{F_\beta}=\frac1{1+\beta^2}\times (\frac1P+\frac{\beta^2}R)$$ $$F_\beta=\frac{(1+\beta^2)\times P\times R}{\beta^2P+R}$$
			- $\beta>0$度量了查全率对查准率的相对重要性
			- $\beta=1$时退化为F1度量
			- $\beta>1$时查全率有更大影响，$\beta<1$时查准率有更大影响
- **有多个二分类混淆矩阵时**，如多次训练/测试，或在多个数据集上训练/测试，或多分类任务每两类别的组合都回应一个混淆矩阵
	- 宏查准率（macro-P），宏查全率（macro-R），宏F1（macro-F1）$$macro-P=\frac1n\sum_{i=1}^nP_i$$ $$macro-R=\frac1n\sum_{i=1}^nR_i$$ $$macro-F1=\frac{2\times macro-P\times macro-R}{macro-P+macro-R}$$
	- 微查准率（micro-P），微查全率（micro-R），微F1（micro-F1）$$micro-P=\frac{\overline {TP}}{\overline {TP}+\overline {FP}}$$ $$micro-R=\frac{\overline {TP}}{\overline {TP}+\overline {FN}}$$ $$micro-F1=\frac{2\times micro-P\times micro-R}{micro-P+micro-R}$$

#### 2.3.3 ROC & AUC
很多分类器是为测试样本产生一个实值或概率预测，然后将这个预测值与一个**分类阈值**（threshold）进行比较，大于阈值为正类，小于阈值为负类，这个实值或概率预测结果的好坏，直接决定了学习器的泛化能力。

我们可以根据这个实值或概率将测试样本进行排序，最可能是正例的在最前面，最不可能的在最后面，分类过程就相当于在这个排序中以某个**截断点**（cut point）将样本分为两部分，前一部分判为正例，后一部分判为负例。排序本身的质量好坏，体现了综合考虑学习器在不同任务下的**期望优化性能**的好坏，或者说一般情况下优化性能的好坏。

- ROC（受试者工作特征 Receiver Operating Characteristic）
	- 根据学习器的预测结果对样例进行排序，按此顺序逐个把样本作为正例进行预测，每次计算TPR和FPR
	- 纵轴为真正例率（True Positive Rate，TPR）$$TPR=\frac{TP}{TP+FN}$$
	- 横轴为假正例率（False Positive Rate，FPR）$$FPR=\frac{FP}{FP+TN}$$
![ROC曲线](https://img-blog.csdnimg.cn/20191021174935105.png#pic_center)
	- 若一个学习器的ROC曲线完全包住另一个学习器的曲线，则前者的性能更好
	- 若两个学习器的ROC曲线有交叉时，则较难判断，解决方案为计算AUC值
- **AUC**（Area Under ROC Curve）：ROC曲线下的面积，若ROC曲线是有坐标为$\{(x_1,y_1),(x_2,y_2),\dots ,(x_n,y_n)\}$的点按序连接而成$(x_1=0,x_m=1)$则$$AUC=\frac12\sum_{i=1}^{m-1}(x_{i+1}-x_i)·(y_i+y_{i+1})$$
- **排序损失**（loss）:考虑每一对正、反例，若正例的预测值小于反例，则记一个罚分，若相等，则记0.5个罚分，给定$m^+$个正例和$m^-$个反例，令$D^+$和$D^-$分别表示正、反例集合，则$$\mathcal l_{rank}=\frac1{m^+m^-}\sum_{x^+\in D^+}\sum_{x^-\in D^-}(\mathbb{I}(f(x^+)<\\f(x^-))+\frac12\mathbb{I}(f(x^+)=f(x^-)))$$
	- $\mathcal l_{rank}$是ROC曲线之上的面积
	- $AUC=1-\mathcal l_{rank}$

#### 2.3.4 代价敏感错误率与代价曲线
前面介绍的性能度量大都架设了均等代价，为权衡不同类型错误造成的不同损失，可为错误赋予**非均等代价**（unequal cost），希望最小化总体代价
- **代价矩阵** cost matrix

二分类代价矩阵 | 预测为正例 | 预测为负例
 - | - | -
真实为正例 | 0 | $cost_{01}$
真实为负例 | $cost_{10}$ | 0

 - **代价敏感（cost-sensitive）错误率**$$E(f;D;cost)=\frac1m(\sum_{x_i\in D^+}\mathbb{I}(f(x_i)\neq y_i)\times cost_{01}+\sum_{x_i\in D^-}\mathbb{I}(f(x_i)\neq y_i)\times cost_{10})$$
类似地，可以得到代价敏感精度等
 - **代价曲线**（cost curve）：
	- 引入：非均等代价下，ROC曲线不能直接反映出学习器的期望总体代价
	- 横轴：取值为[0,1]的正例概率代价$$P(+)cost=\frac{p\times cost_{01}}{p\times cost_{01}+(1-p)\times cost_{10}}$$其中p为样例是正例的概率
	- 纵轴：取值为[0,1]的归一化代价$$cost_{norm}=\frac{FNR\times p\times cost_{01}+FPR\times (1-p)\times cost_{10}}{p\times cost_{01}+(1-p)\times cost_{10}}$$其中，FPR是假正例率，FNR=1-TPN是假反例率
	- ==与ROC曲线的关系==：
	这部分只看教材理解的不够好，这里引用[模型评估与选择（后篇）-代价曲线](https://blog.csdn.net/qq_37059483/article/details/78614189)进行解释，并总结如下：
		- ROC曲线上的每一点对应着代价平面上的一条线段，设ROC曲线上点的坐标为(FPR,TPR)，则可相应计算出FNR，然后在代价平面上绘制一条从(0,FPR)到(1,FNR）的线段，线段下的面积即表示了该条件下的期望总体代价，如此将ROC曲线上的每个点转化为代价平面上的一条线段
		**理解**：ROC曲线中有FRP和NPR即可绘制，没有考虑非均等代价（$cost_{01}$和$cost_{10}$）及先验概率（样例为正例的概率），而代价曲线中则考虑了两者，对于ROC曲线中的一点，根据不同的p值，可以绘制出代价平面上不同的点，这些点连接起来即为一条从(0,FPR)到(1,FNR）的线段，
		- 取所有线段的下界，围成的面积即为在所有条件下学习器的期望总体代价
		**理解**：刚刚我们说过，ROC曲线上的每一点对应着代价平面上的一条线段，这些线段上的每一点都对应着ROC曲线上这一点在某个p值下的代价，我们可以将横轴看作p的某种变换，纵轴为对应的代价。取所有线段的下界构成代价曲线，代价曲线的纵轴可以理解为在每一p下最小的代价，故围成的面积即为所有条件下，这里的所有条件指的是所有的p，学习器的期望总体代价
	![代价曲线](https://img-blog.csdnimg.cn/20191022090045395.png#pic_center)
说明：引文中认为只有在不考虑归一化的情况下，ROC曲线上的一点才对应着代价平面上一条直线，在考虑归一化的情况下则对应着一条或凸或凹的曲线，但经过多次数值模拟，认为即使在归一化的情况下，仍然对应着一条直线。（R语言进行模拟）
			```{r}
			fpr <- 0.2
			tpr <- 0.3
			fnr <- 1-tpr
			
			cost01 <- 2
			cost10 <- 0.5
			
			step <- 0.001
			
			x <- rep(0,1/step)
			y <- rep(0,1/step)
			for (p in seq(0,1,by=step)){
			  x[p/step] <- (p*cost01)/(p*cost01+(1-p)*cost10)
			  y[p/step] <- (fnr*p*cost01+fpr*(1-p)*cost10)/(p*cost01+(1-p)*cost10)
			}
			
			plot(x,y,type="l",ylim=c(0,1))
			```
		![代价曲线数值模拟](https://img-blog.csdnimg.cn/20191022092644977.png#pic_center)


## 2.4 比较检验
机器学习性能比较较为复杂：
 - 我们希望比较的是泛化性能，但实验评估方法中我们只能获得在测试集上的性能
 - 测试集上的性能与测试集本身的选择有很大关系
 - 很多算法有一定的随机性，即便使用相同的参数设置在同一个测试集上运行多次，结果也会有不同
  
 本节默认以错误率为性能度量，用$\epsilon$表示

#### 2.4.1 假设检验
- 泛化错误率$\epsilon$
- 测试错误率$\hat\epsilon$
- 假设检验中的错误率临界值$\bar\epsilon$

**二项检验**（binomial test）：用于判断一个学习器性能的临界值，适用于在一个测试集上做一次测试的情况

原假设为$\epsilon\leq\epsilon_0$，
$$\bar\epsilon=max\epsilon\;\;s.t.\;\;\sum_{i=\epsilon_0\times m+1}^m\begin{pmatrix}m\\i\end{pmatrix}\epsilon^i(1-\epsilon)^{m-i}<\alpha$$
若$\hat\epsilon\leq\bar\epsilon$，在$\alpha$的显著性水平下，不能拒绝原假设，即能以$1-\alpha$的置信度认为学习器的泛化性能不大于$\epsilon_0$

**t检验**（t-test）：用于判断一个学习器性能的临界值，适用于多次重复留出法或交叉验证等进行多次训练、测试的情况

原假设为$\epsilon=\epsilon_0$，假定我们得到了k个测试错误率$\hat\epsilon_1,\hat\epsilon_2,\dots ,\hat\epsilon_k$，则平均测试错误率$\mu$和方差$\sigma^2$为$$\mu=\frac1k\sum_{i=1}^k\hat\epsilon_i$$$$\sigma^2=\frac1{k-1}\sum_{i=1}^k(\hat\epsilon_i^2-\mu)^2$$
考虑到这k个测试错误率可以看作泛化错误率的独立采样，则变量$$\tau_t=\frac{\sqrt k(\mu-\epsilon_0)}\sigma$$服从自由度为k-1的t分布

若$\tau_t\in[t_{\frac\alpha2},t_{1-\frac\alpha2}]$，在$\alpha$的显著性水平下，不能拒绝原假设，即能以$1-\alpha$的置信度认为学习器的泛化性能为$\epsilon_0$

#### 2.4.2 交叉验证t检验
**成对t检验**（paired t-tests）：用于在一个数据集上比较两个学习器性能好坏，要求在相同的训练\测试集上进行训练\测试

原假设为$\epsilon^A=\epsilon^B$，假定我们使用k折交叉验证法得到的测试错误率分别为$\hat\epsilon_1^A,\hat\epsilon_2^A,\dots ,\hat\epsilon_k^A$和$\hat\epsilon_1^B,\hat\epsilon_2^B,\dots ,\hat\epsilon_k^B$，令$\triangle_i=\epsilon_i^A-\epsilon_i^B$，根据差值$\triangle_1,\triangle_2,\dots ,\triangle_k$做t检验

若$\tau_t=\frac{\sqrt k\mu}\sigma\in[t_{\frac\alpha2},t_{1-\frac\alpha2}]$，在$\alpha$的显著性水平下，不能拒绝原假设，即能以$1-\alpha$的置信度认为两个学习器的泛化性能相同

**$5\times 2$交叉验证**：用于比较两个学习器性能好坏，要求在相同的训练\测试集上进行训练\测试

引入原因：有效的假设检验要求测试错误率是泛化错误率的独立采样，但是在通常情况下由于样本有限，在使用交叉验证方法时，不同轮次的训练集会有一定重叠，使得测试错误率并不独立，会过高估计假设成绩的概率

方法：做5次2折交叉验证，每次2折交叉验证之前随即将数据打乱，使得5次交叉验证中的数据划分不重复，记$\triangle_i^1$和$\triangle_i^2$分别为第i次2折交叉验证产生的两对错误率的差，为缓解测试错误率的非独立性，我们仅计算第一次2折交叉验证的两个结果的平均值$$\mu=\frac{\triangle_i^1+\triangle_i^2}2$$但对每次2折实验的结果都计算出其方差$$\sigma_i^2=(\triangle_i^1-\frac{\triangle_i^1+\triangle_i^2}2)^2+(\triangle_i^2-\frac{\triangle_i^1+\triangle_i^2}2)^2$$则变量$$\tau_t=\frac\mu{\sqrt {\frac15\sum_{i=1}^5\sigma_i^2}}$$

#### 2.4.3 McNemar检验
**McNemar检验**：用于在一个数据集上比较两个学习器性能好坏
分类差别列联表 | 算法A正确 | 算法A错误
 - | - | -
算法B正确 | $e_{00}$ | $e_{01}$
算法B错误 |$e_{10}$ | $e_{11}$

原假设为$\epsilon^A=\epsilon^B$，则应有$e_{01}=e_{10}$，考虑变量$$\tau_{\chi^2}=\frac{(\left|e_{01}-e_{10}\right|-1)^2}{e_{01}+e_{10}}$$服从自由度为1的$\chi^2$分布

若$\tau_{\chi^2}$小于临界值$\chi_\alpha^2$，在$\alpha$的显著性水平下，不能拒绝原假设，即能以$1-\alpha$的置信度认为两个学习器的泛化性能相同

#### 2.4.4 Friedman检验 & Nemenyi后续检验
**Friedman检验**：用于在多个数据集上比较多个学习器性能好坏

假定我们在数据集$D_1,D_2,D_3,D_4$上对算法A、B、C进行比较，首先使用留出法或交叉验证法得到每个算法在每个数据集上的测试结果，然后在每个数据集上根据测试性能由好到坏排序，并赋予序值1，2，……，若性能相同，则评分赋值，举例如下表：

多个算法比较序值表 | 算法A | 算法B | 算法C
 - | - | - | -
数据集$D_1$ | 1 | 2 | 3
数据集$D_2$ | 1 | 2.5 | 2.5
数据集$D_3$ | 1 | 2 | 3
数据集$D_4$ | 1 | 2 | 3
平均序值 | 1 | 2.125 | 2.875

原假设为三个算法的性能都相同，假定我们在N个数据集上比较k个算法，令$r_i$表示第i个算法的平均序值（此处暂不考虑评分序值的情况），则$r_i$的均值和方差分别为$\frac{k+1}2$和$\frac{k^2-1}{12N}$，变量$$\tau_{\chi^2}=\frac{k-1}k\times \frac{12N}{k^2-1}\sum_{i=1}^k(r_i-\frac{k+1}2)^2\\=\frac{12N}{k(k+1)}(\sum_{i=1}^kr_i^2-\frac{k(k+1)^2}4)$$服从自由度为k-1的$\chi^2$分布，但上述的“原始Friedman检验”过于保守，现在通常使用变量$$\tau_F=\frac{(N-1)\tau_{\chi^2}}{N(k-1)-\tau_{\chi^2}}$$服从自由度为k-1和(k-1)(N-1)的F分布

**Nemenyi后续检验**：若原假设（所有算法的性能都相同）被拒绝，需进行后续检验（post-hoc test）来进一步区别算法

原假设为两个算法的性能相同，如果两个算法的平均序值差别大于$$CD=q_\alpha\sqrt {\frac{k(k+1}{6N}}$$（$q_\alpha$可根据$\alpha$查表获得）在$\alpha$的显著性水平下，可以拒绝原假设，即能以$1-\alpha$的置信度认为两个学习器的泛化性能不同

可视化展示：如果两个算法的横线段有交叠区域，则他们无显著区别
![Friedman检验示意图](https://img-blog.csdnimg.cn/20191024203351414.png#pic_center)

## 2.5 偏差与方差

#### 2.5.1 偏差-方差分解 bias-variance decomposition
对学习算法除了通过实验估计其泛化性能，我们还希望了解其为什么具有这样的性能，偏差-方差分解将学习算法的期望泛化错误率分解为偏差、方差与噪声之和。$$\mathbb{E}(f;D)=bias^2(x)+var(x)+\epsilon^2$$
证明如下：
![偏差-方差分解证明](https://img-blog.csdnimg.cn/20191024205904548.png#pic_center)

对测试样本$x$，令$y_D$为$x$在数据集中的标记，$y$为$x$的真实标记，$f(x;D)$为训练集D上学得模型f在$x$上的预测输出

以回归任务为例，学习算法的期望预测为$$\bar f(x)=\mathbb{E}_D[f(x;D)]$$
期望输出与真实标记的差别称为bias，即$$bias^2(x)=(\bar f(x)-y)^2$$
使用样本数相同的不同训练集产生的方差为$$var(x)=\mathbb{E}_D[(f(x;D)-\bar f(x))^2]$$
噪声为$$\epsilon^2=\mathbb{E}_D[(y_D-y)^2]$$

- 偏差刻画了学习算法本身的拟合能力
- 方差度量了同样大小的训练集的变动导致的学习性能的变化，即刻画了数据扰动所造成的影响
- 噪声表达了在当前任务上任何学习算法所能达到的期望泛化性能的下界，即刻画了学习问题本身的难度

泛化性能是由学习算法的能力、数据的扰动性及学习任务本身的难度所共同决定的

#### 2.5.2 偏差方差窘境 bias-variance dilemma
给定学习任务，假设我们能控制学习算法的训练程度
- 训练不足时，学习器的拟合能力不够强，训练数据的扰动不足以使学习器发生显著变化，偏差主导了泛化错误率
- 训练充足时，学习器的拟合能力已非常强，训练数据发生的轻微扰动都会使学习器发生显著变化，若训练数据自身的、非全局的特性被学到了，将发生过拟合
![偏差-方差窘境](https://img-blog.csdnimg.cn/2019102421073837.png#pic_center)