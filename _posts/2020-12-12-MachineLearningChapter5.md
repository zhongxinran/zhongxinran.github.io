---
title: 西瓜书 | 第五章 神经网络
author: 钟欣然
date: 2020-12-11 00:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

## 5.1 神经元模型
- **神经网络**（neural networks）：由局域适应性的简单单元组成的广泛并行互连的网络，它的组织能够模拟生物神经系统对真实世界物体所作出的交互反应（注：此处神经网络指神经网络学习，是机器学习与神经网络这两个领域的交叉部分），可以将其视为包含了许多参数的数学模型，这个模型是若干个函数，例如$y_j=f(\sum_iw_ix_i-\theta_j)$相互嵌套代入而得
- **神经元**（neuron/unit）**模型**：神经网络中最基本的成分，即上述定义中的“简单单元”，把许多个神经元按照一定的层次结构连接起来，就得到了神经网络
- **M-P神经元模型**（阈值逻辑单元 threshold logic unit）：神经元接收到来自n个其他神经元传递过来的输入信号，这些输入信号通过带权重的连接（connection）进行传递，神经元接收到的总输入值将与神经元的阈值（threshold）进行比较，然后通过激活函数（activation function）处理以产生神经元的输出
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191030162957939.png#pic_center)
- **激活函数**：
	- 阶跃函数：将输入值映射为输出值0或1，分别对应神经元抑制和兴奋，但具有不连续，不光滑等不太好的性质
	- Sigmoid函数（挤压函数 squashing function）：把在较大范围内变化的输入值挤压到[0,1]输出值范围内
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191030163024652.png#pic_center)
## 5.2 感知机与多层网络
#### 5.2.1 感知机 Perceptron
感知机由两层神经元组成，输入层接收外界输入信号后传递给输出层，输出层是M-P神经元。

**逻辑与、或、非运算**（输入$x_1,x_2$为0或1，f为阶跃函数）
- 与（$x_1\wedge x_2$）：$w_1=1,w_2=1,\theta=2,y=f(x_1+x_2-2)$
- 或（$x_1\vee x_2$）：$w_1=1,w_2=1,\theta=0.5,y=f(x_1+x_2-0.5)$
- 非（$-x_1$）：$w_1=-0.6,w_2=0,\theta=-0.5,y=f(-0.6x_1+0.5)$

**学习规则**

给定训练数据集，权重$w_i(i=1,2,\dots ,w_n)$以及阈值$\theta$可通过学习得到，阈值$\theta$可看做一个固定输入为-1.0的哑结点（dummy node）所对应的连接权重$w_{n+1}$，由此将权重和阈值的学习统一为权重的学习，学习规则为，对样例$(\boldsymbol x,y)$，若当前感知机的输出为$\hat y$，则感知机权重这样调整$$w_i\leftarrow w_i+\triangle w_i$$$$\triangle w_i=\eta(y-\hat y)x_i$$其中，$\eta\in (0,1)$为学习率（learning rate）

**缺点**

感知机只有输出层神经元能进行激活函数处理，即只拥有一层功能神经元（functional neutron），学习能力非常有限。事实上，上述与或非问题是线性可分（linearly separable）的问题，可以证明，若两类模式是线性可分的，即存在一个超平面能将他们分开，感知机的学习过程中一定会收敛（converge）而求得适当的权向量$\boldsymbol w=(w_1;w_2;\dots ;w_{n+1})$，否则感知机学习过程将会发生震荡（fluctuation），$\boldsymbol w$难以稳定下来，不能求得合适解
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191030164528389.png#pic_center)
#### 5.2.2 多层网络
- **隐层**（隐含层 hidden layer）：输出层与输入层之间的神经元，隐含层和输出层都是拥有激活函数的功能神经元
- **多层前馈神经网络**（multi-layer feedforward neutral networks）：层级结构，每层神经元与下一层神经元全互连，神经元之间不存在同层连接，也不存在跨层连接，其中输入层神经元接收外界输入，隐层和输出层对信号进行加工，最终结果由输出层神经元输出，下图a通常被称为两层网络，为避免歧义也成为单隐层网络
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191030165154237.png#pic_center)
- **学习过程**：根据训练数据来调整神经元之间的连接权（connection weight）以及每个功能神经元的阈值
## 5.3 误差逆传播算法 error BackPropagation, BP
BP算法用于训练多层神经网络，包括多层前馈神经网络和其他类型的神经网络，但BP网络多指前者。

#### 5.3.1 标准BP算法
**数据表示**：
- 训练集$D=\{(\boldsymbol x_1,\boldsymbol y_1),(\boldsymbol x_2,\boldsymbol y_2),\dots ,(\boldsymbol x_m,\boldsymbol y_1m)\},\boldsymbol x_i\in \mathbb{R}^d,\boldsymbol y_i\in \mathbb{R}^l$，即输入示例有$d$个属性描述，输出$l$维实值向量
- $d$个输入神经元，$q$个隐层神经元，$l$个输出神经元
- 隐层第h个神经元的阈值为$\gamma_h$，输出层第j个神经元的阈值为$\theta_i$
- 输入层第i个神经元与隐层第h个神经元之间的连接权为$v_{ih}$，隐层第h个神经元与输出层第j个神经元之间的连接权为$w_{hj}$
- 隐层第h个神经元接收到的输入为$\alpha_h=\sum_{i=1}^dv_{ih}x_i$，输出为$b_h$，输出层第j个神经元接收到的输入为$\beta_j=\sum_{h=1}^qw_{hj}b_h$
- 隐层和输出层神经元都是用Sigmoid函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031170032337.png#pic_center)

**学习过程**：

对训练例$(\boldsymbol x_k,\boldsymbol y_k)$，假设神经网络的输出为$\hat \boldsymbol y_k=(\hat y_1^k,\hat y_2^k,\dots ,\hat y_l^k)$，$\hat y_j^k=f(\beta_j-\theta_j)$，则网络在$(\boldsymbol x_k,\boldsymbol y_k)$上的均方误差为$$E_k=\frac12\sum_{j=1}^l(\hat y_j^k-y_j^k)^2$$

对需要确定的$d\times q+q\times l+q+l$个参数，BP是一个迭代学习算法，在迭代的每一轮中采用广义的感知机学习规则对参数进行更新估计$$v\leftarrow v+\triangle v$$

下面我们以$w_{hj}$为例进行推导

BP算法基于梯度下降策略，以目标函数的负梯度方向对参数进行调整，对误差$E_k$，给定学习率$\eta$，有$$\triangle w_{hj}=-\eta \frac{\partial E_k}{\partial w_{hj}}$$
注意到$w_hj$先影响到第j个输出层神经元的输入值$\beta_j$，再影响到其输出值$\hat y_j^k$，然后影响到$E_k$，因此有$$\frac{\partial E_k}{\partial w_{hj}}=\frac{\partial E_k}{\partial \hat y_j^k}\cdot\frac{\partial \hat y_j^k}{\partial \beta_j}\cdot\frac{\partial \beta_j}{\partial w_{hj}}$$
其中$$\frac{\partial \beta_j}{\partial w_{hj}}=b_h$$
又Sigmoid函数有一个很好的性质：$$f'(x)=f(x)(1-f(x))$$
因此有$$\begin{aligned}g_j&=-\frac{\partial E_k}{\partial \hat y_j^k}\cdot\frac{\partial \hat y_j^k}{\partial \beta_j}\\&=-(\hat y_j^k-y_j^k)f'(\beta_j-\theta_j)\\&=\hat y_j^k(1-\hat y_j^k)(y_j^k-\hat y_j^k)\end{aligned}$$
进一步有$$\triangle w_{hj}=\eta g_jb_h$$

同理$$\triangle \theta_j=-\eta g_j$$$$\triangle v_{ih}=\eta e_hx_i$$$$\triangle \gamma_h=-\eta e_h$$
其中$$\begin{aligned}e_h&=-\frac{\partial E_k}{\partial b_h}\cdot\frac{\partial b_h}{\partial \alpha_h}\\&=-\sum_{j=1}^l\frac{\partial E_k}{\partial \beta_j}\cdot\frac{\partial \beta_j}{\partial b_h}f'(\alpha_h-\gamma_h)\\&=\sum_{j=1}^lw_{hj}g_jf'(\alpha_h-\gamma_h)\\&=b_h(1-b_h)\sum_{j=1}^lw_{hj}g_j\end{aligned}$$

学习率控制着算法每一轮迭代中的更新步长，若太大则容易震荡，太小则收敛速度又会过慢，有时为了精细调节，可以令$\triangle w_{hj},\triangle \theta_j$使用$\eta_1$，$\triangle v_{ih},\triangle \gamma_h$使用$\eta_2$，二者不一定相等

算法总结如下：
```python
输入：训练集D={(x_k,y_k)}_{k=1}^m
      学习率\eta
过程：
在(0,1)范围内随机初始化网络中所有连接权和阈值
repeat
	for all(xk,yk) in D do
		根据当前参数计算当前样本的输出\hat y_k
		计算输出层神经元的梯度项g_j
		计算隐层神经元的梯度项e_h
		更新连接权w_hj,v_ih和阈值\theta_j,\gamma_h
	end for
until 达到终止条件
输出：连接权和阈值确定的多层前馈神经网络
```
#### 5.3.2 累积误差逆传播 accumulated error backpropagation
标准BP算法每次针对一个训练样例更新连接权和阈值，累积BP算法则最小化训练集D上的累积误差$$E=\frac1m\sum_{k=1}^mE_k$$，二者都很常用

**优缺点比较**：
- 标准BP算法参数更新非常频繁，对不同样例进行更新的效果可能互相抵消，为了达到同样的累计误差极小点，往往需要更多次数迭代
- 累积误差下降到一定程度后，进一步下降可能非常缓慢，这时标准BP往往更快获得较好的解，尤其是在训练集D非常大时

#### 5.3.3 其它讨论
**隐层神经元个数**

可证明，只需一个包含足够多神经元的隐层，多层前馈神经网络就能以任意精度逼近任意复杂度的连续函数，但是如何设置隐层神经元个数仍是未决问题，实际应用中常通过试错法（trial-by-error）调整

**解决过拟合问题**

由于其强大的表示能力，BP神经网络经常遭遇过拟合，训练误差持续降低，但测试误差却可能上升，有两种策略解决：
- 早停（early stopping）：将数据划分为训练集和验证集，训练集用来计算梯度、更新连接权和阈值，验证集用来估计误差，若训练集误差降低但验证集误差升高，则停止训练，同时返回具有最小验证集误差的连接权和阈值
- 正则化（regularization）：在误差目标函数中增加一个用于描述网络复杂度的部分，例如连接权与阈值的平方和$$E=\lambda\frac1m\sum_{k=1}^mE_k+(1-\lambda)\sum_iw_i^2$$其中，$w_i$表示连接权和阈值，$\lambda\in (0,1)$用于对经验误差与网络复杂度这两项进行这种，常通过交叉验证来估计

## 5.4 全局最小与局部最小
**局部最小**（local minimum）：对$\boldsymbol w^*$和$\theta^*$，若存在$\epsilon>0$使得$\forall (\boldsymbol w;\theta)\in \{(\boldsymbol w;\theta)\vert \Vert (\boldsymbol w;\theta)-(\boldsymbol w^*;\theta^*)\Vert \leq \epsilon\}$，都有$E(\boldsymbol w;\theta)\geq E(\boldsymbol w^*;\theta^*)$，则$(\boldsymbol w^*;\theta^*)$为局部最小解，相应的$E(\boldsymbol w^*;\theta^*)$为局部极小值，参数空间中梯度为零的点只要其误差函数之小于邻点的误差函数之，就是局部最小点。

**全局最小**（global minimum）：若对参数空间中任意$(\boldsymbol w;\theta)$都有$E(\boldsymbol w;\theta)\geq E(\boldsymbol w^*;\theta^*)$，则$(\boldsymbol w^*;\theta^*)$为全局最小解，相应的$E(\boldsymbol w^*;\theta^*)$为全局最小值

**跳出局部最小的策略**：
- 以多组不同参数值初始化多个神经网络，训练后取误差最小的解为最终参数
- 模拟退火（simulated annealing）技术，在每一步都以一定的概率接受比当前解更差的结果，从而有助于跳出局部极小，在每步迭代中，接受次优解的概率要随着时间的推移逐渐降低，从而保证算法稳定
- 使用随机梯度下降
- 遗传算法（genetic algorithms）

需注意的是，上述技术大多是启发式，理论上缺乏保障
## 5.5 其他常见神经网络
#### 5.5.1 RBF网络
径向基函数 Radial Basis Function

**概念**：

一种单隐层前馈神经网络，使用径向基函数作为隐层神经元激活函数，输出层是对隐层神经元输出的线性组合，可表示为$$\varphi(\boldsymbol x)=\sum_{i=1}^qw_i\rho(\boldsymbol x,\boldsymbol c_i)$$
其中，$\rho(\boldsymbol x,\boldsymbol c_i)$是径向基函数，是某种沿径向对称的标量函数，通常定义为样本$\boldsymbol x$到数据中心$\boldsymbol c_i$之间欧氏距离的单点函数，常用的高斯径向基函数$$\rho(\boldsymbol x,\boldsymbol c_i)=e^{-\beta_i\Vert\boldsymbol x-\boldsymbol c_i\Vert^2}$$

**性质**：

具有足够多隐层神经元的RBF网络能以任意精度逼近任意连续函数

**学习过程**：
- 确定神经元中心$\boldsymbol c_i$，常用的方式包括随机采样，聚类等
- 利用BP算法等来确定参数$w_i,\beta_i$
#### 5.5.2 ART网络
自适应谐振理论 Adaptive Resonance Theory

**胜者通吃原则**（winner-take-all）：

竞争型学习是神经网络中一种常用的无监督学习策略，使用此策略时，网络的输出神经元相互竞争，每一时刻仅有一个竞争获胜的神经元被激活，其他神经元的状态被抑制

**概念**：

由比较层、识别层、识别阈值和重置模块组成
- 比较层：负责接收输入样本，并将其传递给识别层神经元
- 识别层：每个神经元对应一个模式类，神经元之间相互竞争以产生获胜神经元，竞争的最简单方式是，计算输入向量与每个识别层神经元所对应的模式类的代表向量之间的距离，距离最小者获胜，获胜神经元向其它神经元发送信号，抑制其激活，神经元数目可在训练过程中动态增长以增加新的模式类
- 识别阈值：若输入向量与获胜神经元所对应的代表向量之间的相似度大于识别阈值，则将该样本归为该代表向量所属类别，同时更新连接权，使得以后在接收到相似输入样本时该模式类会计算出更大的相似度，从而使该神经元有更大可能获胜
- 重置模块：若相似度不大于识别阈值，则重置模块将在识别层增设一个新的神经元，其代表向量就设置为当前输入向量

**识别阈值**：

识别阈值较高时，输入样本将会被分成比较多、比较精细的模式类，识别阈值较低时，会产生比较少、比较粗略的模式类

**优点**：

较好缓解了竞争型学习中的可塑性-稳定性窘境（stability-plasticity dilemma），可塑性指神经网络要有学习新知识的能力，稳定性指神经网络在学习新知识时要保持对旧知识的记忆，这就使得ART网络可以增量学习（incremental learning）或在线学习（online learning）

**发展脉络**：

早期只能处理布尔型输入数据，此后发展成一个算法族，包括能处理实值输入的ART2网络等

#### 5.2.3 SOM网络
自组织映射 Self-Organizing Map

**概念**：

一种竞争学习型的无监督神经网络，将高维输入数据映射到低维空间（通常为二维），同时保持输入数据在高维空间的拓扑结构，即将高维空间中相似的样本点映射到网络输出层的邻近神经元。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019103119503059.png#pic_center)
**学习过程**：

输出层神经元以矩阵方式排列在二维空间中，每个神经元都有一个权向量，在接收输入向量后计算该样本与自身权向量之间的距离，距离最近的神经元成为获胜神经元，称为最佳匹配单元（best matching unit），它决定了该输入向量在低维空间中的位置，然后最佳匹配单元及其邻近神经元的权向量将被调整，以使得这些权向量与当前输入样本的距离缩小，这个过程不断迭代，直至收敛
#### 5.2.4 级联相关网络
Cascade-Correlation

**结构自适应网络**：

结构自适应网络将学习网络结构当作学习的目标之一，并希望在训练过程中找到最符合数据特点的网络结构，级联相关网络是其重要代表

**概念**：

- 级联：建立层次连接的结构，开始训练时，网络只有输入层和输出层，处于最小拓扑结构，随着训练的进行，新的隐层神经元加入，其输入端连接权值是冻结固定的
- 相关：通过最大化新神经元的输出与网络误差之间的相关性（correlation）来训练相关的参数

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031202801686.png#pic_center)

**优缺点**：
- 无需设置网络层数、隐层神经元数目
- 训练速度较快
- 数据较小时易陷入过拟合
#### 5.2.5 Elman网络
**递归神经网络**：

允许网络中出现环形结构，从而可让一些神经元的输出反馈回来作为输入信号，使得网络在t时刻的输出状态不仅与t时刻的输入有关，还与t-1时刻的网络状态有关，从而能处理与时间有关的动态变化，Elman网络是最常用的递归神经网络之一

**概念**：

结构与多层前馈神经网络相似，但隐层神经元的输出被反馈回来，与下一时刻输入层神经元提供的信号一起，作为隐层神经元在下一时刻的输入，隐层神经元常采用Sigmoid激活函数，训练常采用BP算法
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031202826779.png#pic_center)
#### 5.5.6 Boltzmann机
**基于能量的模型**（energy-based model）：

为网络状态定义一个能量，能量最小化时网络达到理想状态，Boltzmann机是一种基于能量的模型

**概念**：

神经元分两层，显层和隐层
- 显层：用于表示数据的输入和输出
- 隐层：数据的内在表达

神经元都是布尔型的，只能取0、1，分别表示抑制、激活，令$\boldsymbol s\in \{0,1\}^n$表示n个神经元的状态，$w_{ij}$表示神经元i和j之间的连接权，$\theta_i$表示神经元i的阈值，则状态向量$\boldsymbol s$对应的Boltzmann能量为$$E(\boldsymbol s)=-\sum_{i=1}^{n-1}\sum_{j=i+1}^nw_{ij}s_is_j-\sum_{i=1}^n\theta_is_i$$
若网络中的神经元以任意不依赖于输入值的顺序进行更新，则网络最终达到Boltzmann分布，此时状态向量$\boldsymbol s$出现的概率仅有其能量与所有可能状态向量的能量决定$$P(\boldsymbol s)=\frac{e^{-E(\boldsymbol s)}}{\sum_{\boldsymbol t}e^{-E(\boldsymbol t)}}$$
训练过程就是将每个训练样本视为一个状态向量，使得其出现的概率尽可能大

**受限Boltzmann机**（Restricted Boltzmann Machine, RBM）：

标准的Boltzmann机是一个全连接图，网络的复杂度高，难以用于解决现实任务，受限Boltzmann机仅保留显层与隐层之间的连接，结构由完全图简化为二部图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031204248563.png#pic_center)
常用对比散度（Contrasive Divergence, CD）算法来进行训练，假定网络中有d个显层神经元和q个隐层神经元，令$\boldsymbol v$和$\boldsymbol h$分别表示显层与隐层的状态向量，则由于同一层内不存在连接，有$$P(\boldsymbol v\vert\boldsymbol h)=\prod_{i=1}^dP(v_i\vert\boldsymbol h)$$$$P(\boldsymbol h\vert\boldsymbol v)=\prod_{j=1}^qP(h_j\vert\boldsymbol v)$$
CD算法对每个训练样本$\boldsymbol v$，先计算出隐层神经元状态的概率分布，然后根据这个概率分布采样得到$\boldsymbol h$，此后类似地从$\boldsymbol h$产生$\boldsymbol v'$，再从$\boldsymbol v'$产生$\boldsymbol h'$，连接权的更新公式为$$\triangle w=\eta(\boldsymbol v\boldsymbol h^T-\boldsymbol v'\boldsymbol h'^T)$$
## 5.6 深度学习
#### 5.6.1 引入
理论上来说，参数越多的模型复杂度约稿，容量（capacity）越大，这意味着它能完成更复杂的学习任务，但一般情况下，复杂模型的训练效率低，易陷入过拟合，随着云计算、大数据时代的到来，计算能力的大幅提高可缓解训练低效性，训练数据的大幅增加可降低过拟合风险

对神经网络模型，提高容量可以增加隐层数目或增加隐层神经元的数目，前者比后者更有效，因为前者不仅增加了拥有激活函数的神经元数目，还增加了激活函数嵌套的层数，多隐层神经网络难以直接用经典算法（如BP）进行训练，因为误差在多隐层内逆传播时，往往会发散（diverge）而不能收敛到稳定状态
#### 5.6.2 无监督逐层训练 unsupervised layer-wise training
- 预训练（pre-training）：每次训练一层隐结点，训练时将上一层隐结点的输出作为输入，而本层隐结点的输出作为下一层隐结点的输入
- 微调（fine-turning）：预训练全部完成后，对整个网络进行微调，如利用BP算法对整个网络进行训练，如深度信念网络（deep belief network, DBN）每一层都是一个受限Boltzmann机

上述做法可视为将大量参数分组，对每组先找到局部较好设置，然后再基于这些局部较优结果联合起来寻找全局最优，这就利用了模型大量参数所提供的自由度的同时有效节省了训练开销
#### 5.6.3 权共享 weight sharing
让一组神经元使用相同的连接权，这个策略在卷积神经网络（Convolutional Neutral Network, CNN）中发挥重要作用，如CNN进行手写数字识别任务例，此处不详述

#### 5.6.4 特征学习/表示学习 feature learning/representation learning
从另一个角度理解深度学习，多隐层堆叠、每层对上一层的输出进行处理的机制，可以看作是在对输入信号进行逐层加工，从而把初始的、与输出目标之间联系不太密切的输入表示，转化成与输出目标联系更密切的表示，使得原来仅基于最后一层输出映射难以完成的任务成为可能。

换言之，通过多层处理，逐渐将初始的低层特征表示转化为高层特征表示后，用简单模型即可完成复杂的分类等学习任务，由此可将深度学习理解为进行特征学习或表示学习。

以往在机器学习用于现实任务时，描述样本的特征通常需由人类专家来设计，称为特征工程（feature engineering），特征的好坏对泛化性能有至关重要的影响，特征学习通过机器学习技术自身来产生好特征，使机器学习向全自动数据分析又前进了一步。
