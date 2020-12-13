---
title: 西瓜书 | 第三章 线性模型
author: 钟欣然
date: 2020-12-9 00:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

## 3.1 基本形式

#### 3.1.1 线性模型 linear model
线性模型试图学得一个通过属性的线性组合来进行预测的函数$$f(x)=w_1x_1+w_2x_2+\dots +w_dx_d+b$$一般用向量形式写作$$f(x)=\boldsymbol w^T\boldsymbol x+b$$其中$\boldsymbol w=(w_1;w_2;\dots ,w_d)$

- 线性模型形式简单，易于建模，却蕴含着机器学习中一些重要的基本思想，许多非线性模型（nonlinear model）可在线性模型的基础上通过引入层级结构或高维映射而得
- 线性模型有很好的的可解释性，$\boldsymbol w$直观表达了各属性在预测中的重要性

## 3.2 线性回归 linear regression
线性回归试图学得一个线性模型以尽可能准确地预测实值输出标记

#### 3.2.1 一元线性回归
**离散属性处理方式**
- 若属性值之间存在序（order）关系，可通过连续化将其转化为连续值
- 若属性值之间不存在序关系，假定有k个属性值，则转化为k维向量

**如何确定w和b**
均方误差是回归任务中最常用的性能度量，且有非常好的几何意义，它对应了欧氏距离（Euclidean distance），因此我们可以采用基于均方误差最小化的最小二乘法（least square method）来进行模型求解$$(w^*,b^*)=\underset{(w,b)}{argmin}\sum_{i=1}^{m}(f(x_i)-y_i)^2=\underset{(w,b)}{argmin}\sum_{i=1}^{m}(y_i-wx_i-b)^2=\underset{(w,b)}{argmin}E_{(w,b)}$$我们可将$E_{(w,b)}$对w和b求导，得到w和b的最优解的闭式（closed-form）解$$w=\frac{\sum_{i=1}^my_i(x_i-\bar x)}{\sum_{i=1}^mx_i^2-m\bar x^2}$$$$b=\frac1m\sum_{i=1}^m(y_i-wx_i)$$

#### 3.2.2 多元线性回归 multivariate linear regression
将$\boldsymbol w$和b吸收入向量形式$\hat \boldsymbol w=(\boldsymbol w;b)$，相应地，把数据集D表示为一个$m\times (d+1)$大小的矩阵$\boldsymbol X$$$\boldsymbol X=\begin{pmatrix}x_{11}&\cdots&x_{1d}&1\\\vdots&\ddots&\vdots&\vdots\\x_{d1}&\cdots&x_{dd}&1\end{pmatrix}$$标记也写成向量形式$\boldsymbol y=(y_1;y_2;\dots ;y_m)$，则有$$\hat\boldsymbol w^*=\underset{\hat\boldsymbol w}{argmin}(\boldsymbol y-\boldsymbol  X\hat\boldsymbol w)^T(\boldsymbol y-\boldsymbol  X\hat\boldsymbol w)=\underset{\hat\boldsymbol w}{argmin}E_{\hat\boldsymbol w}$$$$\frac{\partial E_{\hat\boldsymbol w}}{\partial \hat\boldsymbol w}=2\boldsymbol X^T(\boldsymbol X\hat \boldsymbol w-\boldsymbol y)$$
- 当$\boldsymbol X^T\boldsymbol X$是满秩矩阵（full-rank matrix）或正定矩阵（positive definite matrix）时，$$\hat\boldsymbol w^*=(\boldsymbol X^T\boldsymbol X)^{-1}\boldsymbol X^T\boldsymbol y$$
- 当$\boldsymbol X^T\boldsymbol X$不是满秩矩阵时，可解出多个$\hat\boldsymbol w$，它们都能使均方误差最小化，选择哪个解作为输出将由学习算法的归纳偏好决定，常见的做法是引入正则化（regularization）
#### 3.2.3 广义线性模型 generalized linear model
考虑单调可微函数$g(·)$，令$y=g^{-1}(\boldsymbol w^T\boldsymbol x+b)$，得到的模型即为广义线性模型，其中$g(·)$为联系函数（link function）

一个特例：对数线性回归$\ln y=\boldsymbol w^T\boldsymbol x+b$

## 3.3 对数几率回归 logistic regression
对于分类任务，只需要通过广义线性模型，找一个单调可微函数将分类任务的真实标记y与线性回归模型的预测值联系起来

#### 3.3.1 联系函数
考虑二分类任务，我们需要将实值转化为0/1值

**单位阶跃函数**（unit-step function）：若预测值z大于零就判为正例，小于零判为负例，临界值零任意判别$$y=\left\{\begin{array}{l}0,z<0\\0.5,z=0\\1,z>0\end{array}\right.$$
缺点：单位阶跃函数不连续，因此不能直接用于广义线性模型的$g^-(·)$，因此我们希望找到能在一定程度上近似单位阶跃函数的替代函数（surrogate function），并单调可微

**对数几率函数**（logistic function）：一种Sigmoid函数，将z值转化为一个接近0或1的y值，并且其输出值在z=0附近变化很陡$$y=\frac1{1+e^{-z}}=\frac1{1+e^{-(\boldsymbol w^T\boldsymbol x+b)}}$$可变化为$$\ln \frac y{1-y}=\boldsymbol w^T\boldsymbol x+b$$
若将y视为样本$\boldsymbol x$作为正例的可能性，则1-y是其反例可能性，二者的比值$\frac y{1-y}$称为几率（odds），反映了$\boldsymbol x$作为正例的相对可能性，对几率取对数则得到对数几率（log odds, logit）$\ln \frac y{1-y}$

#### 3.3.2 对数几率回归
用线性回归模型的预测结果去逼近真实标记的对数几率，对应的模型称为对数几率回归，虽然名字是回归，但实际山是一种分类学习方法，优点如下：
- 直接对分类可能性进行建模，无需事先假设数据分布，避免了假设分布不准确带来的问题
- 不仅预测除类别，而是可得到近似概率预测，这对很多需利用概率辅助决策的任务很有用
- 对率回归求解的目标函数是任意阶可导的凸函数，有很好的数学性质，现有的许多数值优化算法都可直接用于求取最优解

**求解过程：**

为便于讨论，令$\beta=(\boldsymbol w,b)$，$\hat \boldsymbol x=(\boldsymbol x,1)$，则将$\boldsymbol w^T\boldsymbol x+b$可简写为$\boldsymbol \beta^T\hat \boldsymbol x$
$$\ln \frac{p(y=1|\boldsymbol x)}{p(y=0|\boldsymbol x)}=\boldsymbol \beta^T\hat \boldsymbol x$$
$$p(y=1|\boldsymbol x)=\frac{e^{\boldsymbol \beta^T\hat \boldsymbol x}}{1+e^{\boldsymbol \beta^T\hat \boldsymbol x}}$$
$$p(y=0|\boldsymbol x)=\frac1{1+e^{\boldsymbol \beta^T\hat \boldsymbol x}}$$
我们可利用**极大似然法**（maximum likelihood method）估计参数，给定数据集，对率回归模型最大化对数似然$$\mathcal{l}(\boldsymbol \beta)=\sum_{i=1}^m\ln p(y_i|\boldsymbol x_i;\boldsymbol \beta)$$
似然项可重写为$$p(y_i|\boldsymbol x_i;\boldsymbol \beta)=y_ip(y=1|\hat\boldsymbol x;\boldsymbol \beta)+(1-y_i)p(y=0|\hat\boldsymbol x;\boldsymbol \beta)$$
故最大化上上式等价于最小化下式$$\mathcal{l}(\boldsymbol \beta)=\sum_{i=1}^m(-y_i\boldsymbol \beta^T\hat\boldsymbol x_i+ln(1+e^{\boldsymbol \beta^T\hat\boldsymbol x_i}))$$
上式是关于$\boldsymbol \beta$的高阶可导连续凸函数，根据凸优化理论，经典的数值优化算法如梯度下降法（gradient descent method）、牛顿法（Newton method）都可求最优解$$\boldsymbol \beta^*=\underset{\boldsymbol \beta}{argmin}\mathcal{l}(\boldsymbol \beta)$$

## 3.4 线性判别分析 Linear Discriminant Analysis
线性判别分析在二分类问题上最早由Fisher提出，亦称“Fisher判别分析”

#### 3.4.1 二分类问题
**思想**

给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离

在对新样本进行分类时，同样将其投影到这条直线上，再根据投影点的位置来确定新样本的类别
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102814534830.png#pic_center)
**目标函数及其计算**

给定数据集$D={(\boldsymbol x_i,y_i)}_{i=1}^m,y_i\in {0,1}$，令$X_i,\boldsymbol \mu_i,\boldsymbol \Sigma_i$分别表示第i类实例的集合、均值向量和协方差矩阵，若将数据投影到直线$\boldsymbol w$上，则两类样本的中心的投影分别为$\boldsymbol w^T \boldsymbol \mu_0、\boldsymbol w^T \boldsymbol \mu_1$，协方差分别为$\boldsymbol w^T \boldsymbol \Sigma_0\boldsymbol w、\boldsymbol w^T \boldsymbol \Sigma_1\boldsymbol w$，由于直线是一位空间，$\boldsymbol w^T \boldsymbol \mu_0、\boldsymbol w^T \boldsymbol \mu_1、\boldsymbol w^T \boldsymbol \Sigma_0\boldsymbol w、\boldsymbol w^T \boldsymbol \Sigma_1\boldsymbol w$都是实数

想要使同类样例的投影点尽可能近，可以让同类样例投影点的协方差尽可能小，即$\boldsymbol w^T \boldsymbol \Sigma_0\boldsymbol w+\boldsymbol w^T \boldsymbol \Sigma_1\boldsymbol w$尽可能小；想要使异类样例的投影点尽可能远，可以让类中心之间的距离尽可能大，即$\Vert \boldsymbol w^T \boldsymbol \mu_0-\boldsymbol w^T \boldsymbol \mu_1\Vert_2^2$，因此目标函数如下$$J=\frac{\Vert \boldsymbol w^T \boldsymbol \mu_0-\boldsymbol w^T \boldsymbol \mu_1\Vert_2^2}{\boldsymbol w^T \boldsymbol \Sigma_0\boldsymbol w+\boldsymbol w^T \boldsymbol \Sigma_1\boldsymbol w}=\frac{\boldsymbol w^T(\boldsymbol \mu_0-\boldsymbol \mu_1)(\boldsymbol \mu_0-\boldsymbol \mu_1)^T\boldsymbol w}{\boldsymbol w^T (\boldsymbol \Sigma_0+\boldsymbol \Sigma_1)\boldsymbol w}$$
定义类内散度矩阵（within-class scatter matrix）$$\boldsymbol S_w=\boldsymbol \Sigma_0+\boldsymbol \Sigma_1=\sum_{\boldsymbol x\in X_0}(\boldsymbol x-\boldsymbol \mu_0)(\boldsymbol x-\boldsymbol \mu_0)^T+\sum_{\boldsymbol x\in X_1}(\boldsymbol x-\boldsymbol \mu_1)(\boldsymbol x-\boldsymbol \mu_1)^T$$类间散度矩阵（between-class scatter matrix）$$\boldsymbol S_b=(\boldsymbol \mu_0-\boldsymbol \mu_1)(\boldsymbol \mu_0-\boldsymbol \mu_1)^T$$
则目标函数转为为$\boldsymbol S_b$和$\bold Ssymbol_w$的广义瑞利商（generalized Rayleigh quotient）$$J=\frac{\boldsymbol w^T\boldsymbol S_b\boldsymbol w}{\boldsymbol w^T \boldsymbol S_w\boldsymbol w}$$注意到分子分母均为$\boldsymbol w$的二次项，因此上式的解与$\boldsymbol w$的长度无关，只与其方向有关，令$\boldsymbol w^T \boldsymbol S_w\boldsymbol w=1$，则上式等价于$$\underset{\boldsymbol w}{min}\boldsymbol w^T \boldsymbol S_b\boldsymbol w\\s.t. \boldsymbol w^T \boldsymbol S_w\boldsymbol w=1$$由拉格朗日乘子法（补充[矩阵求导](https://blog.csdn.net/daaikuaichuan/article/details/80620518)），上式等价于$$\boldsymbol S_b\boldsymbol w=\lambda \boldsymbol S_w\boldsymbol w$$注意到$\boldsymbol S_b\boldsymbol w$的方向恒为$\boldsymbol \mu_0-\boldsymbol \mu_1$，不妨令$$\boldsymbol S_b\boldsymbol w=\lambda(\boldsymbol \mu_0-\boldsymbol \mu_1)$$则有$$\boldsymbol w=\boldsymbol S_w^{-1}(\boldsymbol \mu_0-\boldsymbol \mu_1)$$考虑到数值解的稳定性，通常对$\boldsymbol S_w$进行[奇异值分解](https://www.cnblogs.com/endlesscoding/p/10033527.html)，即$\boldsymbol S_w=\boldsymbol U\boldsymbol \Sigma\boldsymbol V^T$，这里$\boldsymbol \Sigma$为实对角矩阵，其对角线上的元素是$\boldsymbol S_w$的奇异值，得$\boldsymbol S_w^{-1}=\boldsymbol V\boldsymbol \Sigma^{-1}\boldsymbol U^T$

值得一提的是，LDA可从贝叶斯决策理论的角度来阐释，并可证明当两类数据同先验、满足高斯分布且协方差相等时，LDA可达到最优分类。

#### 3.4.2 多分类问题
全局散度矩阵$$\boldsymbol S_t=\boldsymbol S_b+\boldsymbol S_w=\sum_{i=1}^m(\boldsymbol x_i-\boldsymbol \mu)(\boldsymbol x_i-\boldsymbol \mu)^T$$类内散度矩阵为每个类别的散度矩阵之和$$\boldsymbol S_w=\sum_{i=1}^N\boldsymbol S_{w_i}=\sum_{i=1}^N\sum_{\boldsymbol x\in X_i}(\boldsymbol x-\boldsymbol \mu_i)(\boldsymbol x-\boldsymbol \mu_i)^T$$由此可得$$\boldsymbol S_b=\boldsymbol S_t-\boldsymbol S_w=\sum_{i=1}^Nm_i(\boldsymbol \mu_i-\boldsymbol \mu)(\boldsymbol \mu_i-\boldsymbol \mu)^T$$
常见的一种优化目标为$$\underset{\boldsymbol W}{max}\frac{tr(\boldsymbol W^T\boldsymbol S_b\boldsymbol W)}{tr(\boldsymbol W^T\boldsymbol S_w\boldsymbol W)}$$其中$\boldsymbol W\in \mathbb{R}^{d\times (N-1)}$，上式可通过如下广义特征值问题求解$$\boldsymbol S_b\boldsymbol W=\lambda \boldsymbol S_w\boldsymbol W$$

$\boldsymbol W$的闭式解是$\boldsymbol S_w^{-1}\boldsymbol S_b$的$d'$个最大非零广义特征值所对应的特征向量组成的矩阵，$d'\leq N-1$

若将$\boldsymbol W$视为一个投影矩阵，则多分类LDA将样本投影到$d'$维空间，$d'$通常远小于数据原有的属性$d$，可通过这个投影来降维，且投影过程中使用了类别信息，因此LDA也常被视为一种经典的**监督降维技术**。

## 3.5 多分类学习
有些二分类学习方法可直接推广到多分类，但在更多情形下，可以基于一些基本策略，利用二分类学习器解决多分类学习任务：先对问题进行拆分，为拆分出的每个二分类任务训练一个分类器；在测试时，对这些分类器的预测结果进行集成以获得最终的多分类结果。拆分策略主要有以下三种：

#### 3.5.1 一对一 One vs. One, OvO
OvO将N个类别两两配对，从而产生$\frac{N(N-1)}2$个二分类任务，在测试阶段，新样本将同时提交给所有分类器，得到$\frac{N(N-1)}2$个结果，被预测得最多的类别作为分类结果

#### 3.5.2 一对多 One vs. Rest, OvR
每次将一个类的样例作为正例、所有其它类的样例作为反例来训练N个分类器，若测试时仅有一个分类器预测为正类，则对应的类别标记作为最终结果，若有多个分类器预测为正类，则通常考虑各分类器的预测置信度，选择置信度最大的类别标记为分类结果

- OvO的存储开销和测试时间开销通常比OvR大
- 在类别很多时，OvO的训练时间开销通常比OvR小（OvO每次仅用两个类别的样例，OvR每次用全部训练样例）
- 预测性能取决于具体的数据分布，多数情形下二者差不多
#### 3.5.3 多对多 Many vs. Many, MvM
每次将若干个类作为正类，若干个其它类作为反类，显然OvO、OvR是MvM的特例。MvM的正反类构造必须有特殊的设计，最常用的是纠错输出码（Error Correcting Output Codes, ECOC）

**ECOC**
- 编码：将N个类别做M次划分，每次将一部分类别划分为正例，一部分划分为反例，从而形成M个训练集，训练出M个分类器，编码矩阵（coding matrix）有多种形式，常见的主要有二元码和三元码，前者将每个类别分别制定为正类和反类，后者在正反类外还可制定停用类
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191028225815274.png#pic_center)
- 解码： M个分类器对测试样本的测试结果组成一个编码，将这个编码与每个类别各自的编码进行比较，返回距离最小的类别作为最终结果，对分类器的错误有一定的容忍和修正能力
	- 对同一个学习任务，ECOC编码越长，纠错能力一般也越强，但所需训练的分类器越多，计算、存储开销越大，对有限类别数，可能的组合数目是有限的，码长超过一定范围后就失去了意义
	- 对同等长度的编码，理论上来说任意两个类别之间的编码距离越远，则纠错能力越强，因此在码长较小时，可根据这个原则计算出理论最优编码，码长稍大一些时则很难，但通常我们不需要最优编码，非最优编码在实践中往往已能产生足够好的分类器
	- 并不是编码的理论性质越好，分类性能就越好

## 3.6 类别不平衡问题
为应对分类任务中不同类别的训练样例数差别很大的情况，我们一般有三种做法（j假设正类样例较少，反类样例较多）

#### 3.6.1 欠采样 undersampling
对训练集里的反类样例进行欠采样，即去除一些反例使得正反例数目接近

若随机丢弃反例，可能丢失一些重要信息，代表性算法EasyEnsemble利用集成学习机制，将反例划分为若干个集合供不同学习器使用，这样每个学习器来看都进行了欠采样，但在全局来看却不会丢失重要信息

#### 3.6.2 过采样 oversampling
对训练集里的正类样例进行过采样，即增加一些正例使得正反例数目接近

不能简单地对初始正例样本进行重复采样，否则会导致严重的过拟合，代表性算法SMOTE通过对训练集里的正例进行差值来产生额外的正例

欠采样的时间开销通常小于过采样，因为前者去除样例，后者增加样例

#### 3.6.3 再缩放 rescaling
直接基于原始训练集进行学习，但在用训练好的分类器进行预测时，修改决策过程，称为**阈值移动**（threshold-moving）

从线性分类器的角度考虑，$y=\boldsymbol w^T\boldsymbol x+b$对样本$\boldsymbol x$进行分类时，实际上是用预测出的y与一阈值进行比较，y表达了正例的可能性，几率$\frac y{1-y}$则反映了正例可能性与反例可能性的比值，阈值设置为0.5表明分类器认为真实正反例可能性相同，即分类器决策规则为：$$若\frac y{1-y}>1，则预测为正例$$

当训练集中正反例的数目不同时，令$m^+$表示正例数目，$m^-$表示反例数目，则观测几率是$\frac{m^+}{m^-}$，由于我们通常假设训练集是真实样本总体的无偏采样，因此观测几率就代表了真实几率，因此只要分类器的预测几率高于观测几率就应判为正例：$$若\frac{y}{1-y}>\frac{m^+}{m^-}，则预测为正例$$因此我们做出调整，令$$\frac{y'}{1-y'}=\frac{y}{1-y}\times \frac{m^-}{m^+}$$则可使用上述线性分类器进行决策。

注意：训练集是真实样本总体的无偏采样这个假设往往并不成立