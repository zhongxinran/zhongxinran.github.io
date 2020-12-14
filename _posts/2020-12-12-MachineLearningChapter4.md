---
title: 西瓜书 | 第四章 决策树
author: 钟欣然
date: 2020-12-10 00:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

# 1. 基本流程
决策树是基于树结构进行决策的，决策过程的最终结论对应了判定结果，如是或者不是好瓜，决策过程中提出的每个判定问题都是对每个属性的测试，每个测试的结果或是导出最终结论，或是导出进一步的判定问题，其考虑范围是在上次决策结果的限定范围之内。

一棵决策树包含一个根结点、若干个内部结点和若干个叶结点，根结点和内部结点则对应于一个属性测试，叶结点对应于决策结果，根结点包含样本全集，其它每个结点包含的样本集合根据属性测试的结果被划分到子结点中，从根结点到每个叶结点的路径对应一个判定测试序列。决策树的目的是产生一棵繁华能力强，即能处理未见示例能力强的决策树。基本算法如下：



```
输入: 训练集D={(x1,y1),(x2,y2),…(xm,ym)}
      属性集A={a1,a2,…,ad}
过程: 函数TreeGenerate(D,A)
生成结点node;
if D中样本全属于同一类别C then
	将node标记为C类叶结点；return
end if
if A=∅ OR D中样本在A上取值相同 then
	将node标记为叶结点，其类别标记为D中样本最多的类; return
end if
从A中选择最有划分属性a*; 
for a*的每一个值a*v do
	为node生成一个分支; 为Dv表示D中在a*上取值为a*v的样本子集；
	if Dv为空 then
		将分支结点标记为叶结点，其类别标记为D中样本最多的类; return
	else
		以TreeGenerate(Dv,A\{a*})为分支结点
	end if
end for
输出: 以node为根结点的一棵决策树
```



决策树的生成是一个递归过程，在决策树基本算法中，有三种情形会导致递归返回：

- 当前结点包含的样本全属于同一类别，无需划分
- 当前属性集为空，或所有样本在所有属性上取值相同，无法划分（利用当前结点的后验分布，将当前结点标记为叶结点，类别设定为该结点所含样本最多的类别）
- 当前结点包含的样本集合为空，不能划分（利用父结点的样本分布作为当前结点的先验分布，将当前结点标记为叶结点，类别设定为其父结点所含样本最多的类别）



# 2. 划分选择

决策树学习的关键是如何选择最优划分属性，我们希望随着划分过程不断进行，决策树的分支结点所包含的样本尽可能属于同一类别，即结点的纯度（purity）越来越高



## 信息增益 information gain（用于ID3决策树算法）

- 信息熵（information entropy）：度量样本集合纯度最常用的指标，值越小，纯度越高

$$
Ent(D)=-\sum_{k=1}^{\vert \gamma\vert}p_klog_2p_k
$$

- 信息增益：考虑到不同分支结点包含的样本数不同，给分支结点赋予权重$\frac{\vert D^v\vert}{\vert D\vert}$，即样本数越多的分支结点影响越大，因此用属性a对样本集D进行划分所获得的信息增益为

$$
Gain(D,a)=Ent(D)-\sum_{v=1}^V\frac{\vert D^v\vert}{\vert D\vert}Ent(D^v)
$$

- 选择属性$a_*=\underset{a\in A}{argmax}Gain(D,a)$



## 增益率 gain ratio（用于C4.5决策树算法）

- 增益率（gain ratio）：信息增益准则对可取值数目较多的属性有所偏好，因此引入增益率

$$
Gain\_ratio(D,a)=\frac{Gain(D,a)}{IV(a)}
$$

- 固有值（intrinsic value）：IV(a)为属性a的固有值，属性a的可能取值数目越多，即V越大，IV(a)的值通常越大

$$
IV(a)=\sum_{v=1}^V\frac{\vert D^v\vert}{\vert D\vert}log_2\frac{\vert D^v\vert}{\vert D\vert}
$$

- 增益率准则对可取数目较少的属性有所偏好，因此C4.5算法不是直接选择增益率最大的候选划分属性，而是使用了一个启发式：先从候选划分属性中找出信息增益水平高于平均水平的属性，再从中选择增益率最高的



## 基尼指数 Gini index（用于CART决策树算法）

- 基尼值：度量样本集合纯度的指标，反映了从一个数据集中随机抽取两个样本，其类别标记不一致的概率，值越小，纯度越高

$$
Gini(D)=\sum_{k=1}^{\vert \gamma\vert}\sum_{k'\neq k}p_kp_{k'}=1-\sum_{k=1}^{\vert \gamma\vert}p_k^2
$$

- 属性a的基尼指数（Gini index）定义为

$$
Gini\_index(D,a)=\sum_{v=1}^V\frac{\vert D^v\vert}{\vert D\vert}Gini(D^v)
$$

- 选择属性$a_*=\underset{a\in A}{argmin}Gini\_index(D,a)$



# 3. 剪枝处理 pruning

剪枝（pruning）是对付过拟合的主要手段。在决策树学习过程中，为了尽可能正确分类训练样本，结点划分过程将不断重复，有时会造成决策树分支过多，这就可能因训练样本学得太好了，以至于把训练集自身的一些特点当做所有数据都具有的一般性质而导致过拟合。因此，可利用留出法等方法预留出一部分数据用作验证集以进行评估。

决策树桩（decision stump）：只有一层划分的决策树



## 预剪枝 prepruning

在决策树生成过程中，对每个结点在划分前先进性估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点

优点：
- 降低了过拟合的风险
- 显著减少了决策树的训练时间开销和测试时间开销

缺点：
- 基于贪心本质禁止可能导致当前泛化性能不能提升或暂时下降，但在其基础上的后续划分能导致泛化性能显著提高的分支展开，带来了欠拟合的风险



## 后剪枝 postpruning

先从训练集生成一棵完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点替换为叶结点能带来泛化性能的提升，则将该子树替换为叶结点

优点：
- 通常比预剪枝决策树保留了更多的分支，欠拟合风险很小，泛化性能往往优于预剪枝决策树
- 训练时间开销比未剪枝决策树和预剪枝决策树都要大得多



# 4. 连续与缺失值



## 连续值处理

最简单的策略是**二分法**（bi-partition），用于C4.5决策树算法。

给定样本集D和连续属性a，假定a在D上出现了n个不同的取值，将这些值从小到大进行排序，记为$\{a^1,a^2,\dots ,a^n\}$，基于划分点t可将D划分为子集$D_t^-$和$D_t^+$，前者包含在属性a上取值不大于t的样本，后者包含在属性a上取值大于t的样本，对相邻的属性取值$a^i$和$a^{i+1}$来说，t在区间$[a^i,a^{i+1})$中取任意值结果相同，因此对连续属性a，我们可考虑包含n-1个元素的候选划分点集合

$$
T_a=\{\frac{a^i+a^{i+1}}2\vert1\leq i\leq n-1\}
$$

我们就可像考察离散属性值一样来考察这些划分点，选取最优的划分点来进行样本集合的划分

$$
Gain(D,a)=\underset{t\in T_a}{max}Gain(D,a,t)=\underset{t\in T_a}{max}Ent(D)-\sum_{\lambda\in\{-,+\}}\frac{\vert D_t^\lambda\vert}{\vert D\vert}Ent(D_t^\lambda)
$$

与离散属性不同，若当前结点划分为连续属性，该属性还可作为其后代结点的划分属性。



## 缺失值处理

**如何在属性值缺失的情况下进行划分属性选择**

给定训练集D和属性a，$\tilde D$表示D中在属性a上没有缺失值的样本子集，显然我们仅可根据$\tilde D$来判断属性a的优劣

假定属性a有V个可取值$\{a^1,a^2,\dots ,a^V\}$，令$\tilde D^v$表示$\tilde D$中在属性a上取值为$a^v$的样本子集，$\tilde D_k$表示$\tilde D$中属于第k类（$k=1,2,\dots ,\vert \gamma\vert$）的样本子集

假定我们为每个样本$\boldsymbol x$赋予一个权重$w_{\boldsymbol x}$，并定义

$$
\rho=\frac{\sum_{\boldsymbol x\in \tilde D}w_{\boldsymbol x}}{\sum_{\boldsymbol x\in D}w_{\boldsymbol x}}
$$

$$
\tilde p_k=\frac{\sum_{\boldsymbol x\in \tilde D_k}w_{\boldsymbol x}}{\sum_{\boldsymbol x\in \tilde D}w_{\boldsymbol x}}(1\leq k\leq \vert \gamma\vert)
$$

$$
\tilde r_v=\frac{\sum_{\boldsymbol x\in \tilde D^v}w_{\boldsymbol x}}{\sum_{\boldsymbol x\in \tilde D}w_{\boldsymbol x}}(1\leq v\leq V)
$$

直观地看，对属性a，$\rho$表示无缺失值样本所占的比例，$\tilde p_k$表示无缺失值样本中第k类所占的比例，$\tilde r_v$表示无缺失值样本中在属性a上取值为$a^v$的样本所占的比例

因此，我们可以将信息增益的计算式推广为

$$
Gain(D,a)=\rho\times Gain(\tilde D,a)=\rho\times (Ent(\tilde D)-\sum_{v=1}^V\tilde r_vEnt(\tilde D^v))
$$

其中

$$
Ent(\tilde D)=\sum_{k=1}^{\vert \gamma\vert}\tilde p_klog_2\tilde p_k
$$

**给定划分属性，若样本在该属性上的值缺失，如何对样本进行划分**

若样本$\boldsymbol x$在划分属性a上的取值已知，则将$\boldsymbol x$划入与其取值对应的子结点，且样本权值在子结点中保持为$w_{\boldsymbol x}$；若样本$\boldsymbol x$在划分属性a上的取值未知，则将$\boldsymbol x$同时划入所有子结点，且样本权值在与属性值$a^v$对应的子结点中调整为$\tilde r_v·w_{\boldsymbol x}$，直观地看，就是让同一个样本以不同概率划入到不同的子结点中



# 5. 多变量决策树 multivariate decision tree

若我们把每个属性视为坐标空间中的一个坐标轴，则d个属性描述的样本对应了d维空间的一个点，对样本分类意味着在这个坐标空间中寻找不同类样本之间的分类边界，决策树所形成的的分类边界有一个明显的特点是**轴平行**（axis-parallel），即它的分类边界由若干个与坐标轴平行的分段构成，但在学习任务的真实分类边界比较复杂时，必须使用很多段划分才能获得较好的近似，此时的决策树很复杂，由于要进行大量的属性测试，预测时间开销很大。



![在这里插入图片描述](https://img-blog.csdnimg.cn/20191030144602220.png#pic_center)



**多变量决策树**就是实现斜划分甚至更复杂划分的决策树，此时非叶结点不再是仅对某个属性，而是对属性的线性组合进行测试，换言之，每个非叶结点是一个星图$\sum_{i=1}^dw_ia_i=t$的线性分类器，其中$w_i$是属性$a_i$的权重，$w_i$和t可在该结点所含的样本集和属性集上学得。



![在这里插入图片描述](https://img-blog.csdnimg.cn/20191030145038715.png#pic_center)
