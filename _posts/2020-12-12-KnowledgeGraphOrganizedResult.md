---
title: 知识图谱 | 基于监督学习方法的概念图谱构建之英文文本成果梳理
author: 钟欣然
date: 2020-12-12 00:44:00 +0800
categories: [知识图谱, 成果梳理]
math: true
mermaid: true
---


# 1. 数据来源

本报告数据来自于**统计学科的英文教科书**，覆盖了概率论、凸优化、随机过程、统计模型、深度学习等统计学科由浅入深的多个方面。这些课程之间存在明显的先修关系，这些“**先修关系**”的根本来源是教科书中**关键概念**之间的关联。本报告更是将分析粒度细化至**章节**：通过预定义这9本教科书、共150个章节之间的先后修关系，将其作为标签，用章节-关键词词频矩阵作为特征，有监督地学习出这些课程关键词之间的先后修关系，为教师授课、学生上课提供学习路线规划建议。


# 2. 数据处理

本报告建模过程中，所需标签数据为教科书库中所有章节的先后修关系，所需特征数据为所有概念词（组）在各章节出现的频数，旨在通过Liu(2015)提出的有监督学习方法，从课程间的先后修关系学习得到概念之间的先后修关系矩阵。

## 概念提取

对于每一本英文教科书，我们提取教材附录中的概念列表，并通过人工筛选去除与该门课程核心内容无关的概念词组，进而过滤得到关键概念词表。各门课程经过提取所获得的关键概念数表如下：

<br>
表1 九本教科书中分别提取的概念数

Book_names|Concepts_nums
-|-
Convex Optimization|385
Statistical Models  |629 
Categorical Data Analysis|293
Reinforcement Learning|64
Probabiliy|218
Stochastic Processes|230
Deep Learning|393
Computational Statistics|388
Regression Modeling Strategies|130

## 概念处理与清洗

对于九本书获取的全部概念，经过以下步骤进一步处理与清洗：

- 对于九本书得到的所有概念，去重后汇总成关键概念合集
- 对于同义词，指定一个概念词代表此同义词集，以字典形式存入概念合集中，以便统计概念频数时合并同义词频
- 鉴于部分概念词或词组的缩写为单个字符的情况，单个字符不作为该词或词组的同义词录入合集，避免统计词频过程中噪声干扰严重。

经以上处理后，九本教科书合并的概念合集共计2309个关键概念词。

## 概念总频数提取

在九本英文教科书中，查找并统计所有关键概念出现的频次，并进行以下处理：

- 对于同义词：统计各同义概念频数后加总，作为指定概念词的词频
- 对于词组包含：若概念词（组）A包含概念词（组），在分别统计A、B词频后，要从B词频中剔除被包含在A内计算的情况，避免重复计算

在合并的概念集中，出现频次位居前三的概念分别为“概率”（probability)、“均值”（mean)、“分布”（distribution）。图1显示了出现频次最高的50个概念在九本书中出现频次的热力图。从中可以看出，“概率”一次在《概率论》一书中出现次数尤其高，《统计模型》一书中出现频次较高的概念为“均值”、“分布”、“样本”、“方差”等，而“对偶”、“域”两词在《凸优化》教材中出现次数较多。可以看出，每门课程涉及的概念、对应概念出现的次数各有不同，由该门课程的核心内容所决定，最终决定了不同课程之间的先后修关系。

![Alt](https://img-blog.csdnimg.cn/20191114154742653.jpeg#pic_center =400x400)

<center> 图1 九本教科书与出现频次最高的50个概念的分布热力图 </center><br>

<div STYLE="page-break-after: always;"></div>

而不同的概念在每本书中的出现轨迹不同：一本书在开头章节所提出的新概念，很有可能成为之后某个章节的讨论主题；或者，某些概念的定义本身就建立在其他概念的基础上。图2绘制了“随机变量”（random variable）和“均值”（mean）概念在《概率论》一书中出现的分布密度曲线：可见，因为《概率论》一书的核心概念便是“随机变量”，因此其在全书各处出现的概率都不低；而均值的提出建立在随机变量的基础上，《概率论》第四章更是顺次定义了“random variable”和“mean”两个概念，并且在之后的讨论中，随机变量的统计性质——均值——也被频繁地提及。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114154712908.jpeg#pic_center =400x400)
<center>图2 “random variable"与"mean"在《概率论》中的概率密度</center><br>


## 章节先后修关系

本报告建模时，将章节之间的先后修关系作为先验知识，用以训练模型。因而，本报告邀请了中国人民大学统计学院统计学专业的博士学生，对九本书、共150个章节间的先后修关系进行判断，最终统计出577条先后修关系。其中《概率论》（Probability）、《统计模型》（Statistic Models）、《深度学习》（Deep Learning）三本书之间的章节关系如图3所示：可以看出，《概率论》的第十章是《深度学习》第十七章的先修课，同时也是《概率论》第四章的后修课。类似的先后修关系广泛地存在于这九本书的所有章节中，我们据此来构造章节先后修关系矩阵，作为监督学习的标签阵。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114154545381.png#pic_center =400x300)

<center> 图3 三本教科书之间的章节关系示意图 </center>


## 验证概念对的挑选

本报告特邀请中国人民大学统计学院本硕博等数十名学生，根据概念之间的定义关系，从给定的九本英文教科书中挑选出共计184组概念先后修对，以便建模准确性的验证。

<div STYLE="page-break-after: always;"></div>

# 3. 建模分析

## 训练设置

将全量数据划分为训练集和测试集，训练时设置$tolerance=0.01$，初始化$\eta=1.0$，并根据测试集结果选择最优的参数$\lambda$。同时，在构建描述章节先后修关系的三元组时，为了保证三元组关系的准确性，本实验在对负样本（即不存在先后修关系的章节对）进行抽样后方用以构建训练标签。

## 建模结果



从测试集结果来看，表现最优的$\lambda=0.01$。依此参数设置进行全量数据训练，得到最终各个概念之间先后修关系的预测。用auc评价模型性能，如表2：

<br>
表2 建模的准确性结果

| |auc|auc(Liu.2015)|
|-|-|-|
|训练集|0.9696|0.9998|
|测试集|0.9696|0.9535|


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114154520650.png#pic_center =400x400)
<center> 图4 预测的概念先后修关系示意图 </center>
<br>

在预测的概念关系图中，先后修强度最强的前100组概念词对如图4所示。每一个红色的实心圆代表一个概念词，灰色的边表示概念之间的先后修关系，箭头指向为先修词指向后修词，灰色边上的数字表示先后修关系的预测分数，即概念矩阵$A$中$A_{ij}$的取值。

从图4可以看出“markov process”是概念关系密度最为集中的概念。图5为所有与markov process有关的概念对，可以看出：“random variable”、“probability”等词是讨论“markov process”的基础，而“stopping time”是“markov process”的性质，“mcmc”等是“markov process”的应用。这样来看，这些概念词对的关系预测具有较优的准确性。但箭头的指向方向准确度仍有待提高。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114154357249.png#pic_center =400x400)
<center> 图5 “markov process”相关关系 </center>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114154421877.png#pic_center =200x200)
<center> 图6 “Generalized Estimating Equations”相关关系 </center><br>


图6是前100个概念对中，与“Generalized Estimating Equations”（广义估计方程，缩写为gee）有关的概念对图。可以看到“random effects”是“GEE”的先修概念，其强度系数为19.25，而“marginal”与“GEE”也呈现强相关性，关系强度系数为16.53。从广义估计方程的概念来看，这两组关系都符合认知。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114154953777.png#pic_center =400x400)
<center> 图7 三本书中关键概念的关系 </center><br>

图7是《Statistical Models》、《Stochastic Processes》和《Convex Optimization》三本书中几个典型概念的先后修关系，probability、distribution、sample等概念处于中心位置，基本符合认知。

## 探索性分析
表3记录了所有验证概念对预测得分（value）位居前十的情况，最后一列direction表示概念对关系预测结果中先修词（source）与后修词（target）的箭头指向：“-”表示预测结果与先验知识相反，箭头指向错误预测为target指向source；“+”表示预测结果与先验知识相同，表明模型预测的先后修关系正确。

从得分（value）来看，验证概念对的得分较高，表明当前模型可以很好地识别概念词对之间的关系强弱；但从箭头方向（direction）来看，得分最高的前十组概念对有七组预测错误，并且得分最高的前四对概念对预测方向全部错误，这说明当前模型在预测先后修方向上仍然存在较大问题。

<br>
表3 验证概念对的预测结果

|source|target|value|direction|
|:-:|:-:|:-:|:-:|
|independent|brownian motion|9.5311|-|
|markov process|gibbs sampler|9.47|-|
|likelihood|maximum likelihood estimator|8.4775|-|
|random variable|variance|8.1154|-|
|variable|random variable|7.4379|+|
|random variable|stochastic process|7.1443|-|
|random variable|stochastic process|6.6082|+|
|brownian motion|square integrable martingale|5.5485|+|
|descent|newton step|5.0701|-|
|independent|poisson process|5.0372|-|

进一步地，我们对于模型预测的概念词对关系进行“追根溯源”，绘制先后修关系中的两个概念在所有书籍中的频次分布直方图，以及相关章节之间的先后修关系。

图7分别绘制了“descent”和“newton step”在所有教科书中的出现频次直方图。可以看到，“descent”一词主要在《Computational Statistics》、《Convex Optimization》和《Deep Learning》中出现，其中出现频次最多的是《Convex Optimization》一书的第九章。而“newton step”主要在《Computational Statistics》和《Convex Optimization》中出现，其中出现频次最多的是《Convex Optimization》。可以看出，在《Convex Optimization》一书中，先修概念“descent”在后修概念“newton step“之前出现，两者的频次高峰依次出现。这是由于在定义后修概念“newton step”之前，需要先定义先修概念“descent”；并且，在教学过程中，需要在“descent”的基础上提出“newton step”的概念，然后进一步加以讨论。这就解释了“descent”与“newton step”在《Convex Optimization》的第九章共同出现，并且“descent”出现频次达到最高峰，而“newton step”在接下来的第十章到达了自身出现频次的高峰。据此可以推测，“descent”一词在《Convex Optimization》第九章被提出并重点讨论，而“newton step”是在第九章、在“descent”的基础上被定义提出，并在第十章被重点讨论。这样一种概念间的先后修关系，对应到章节层面，表现为《Convex Optimization》第九章为第十章的先修。这就从概念出现频次的角度，解释了课程之间、章节之间的先后修关系与概念间先后修关系的联系。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191114154244353.png#pic_center =400x300)
<center> 图7 “descent”与“newton step”词频直方图及相关章节关系图 </center><br>



## 未来研究方向

未来可以改进的方向如下：

* 如何更好地定义章节之间的先修关系？考虑用机器计算代替专家标注，或者在机器计算的基础上加入专家意见。探索方向：主题模型？
* 怎样更好地利用概念词出现频次的前后关系？
* 如何修正模型中箭头方向不准确的问题？
