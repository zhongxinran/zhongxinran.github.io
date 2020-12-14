---
title: 优化问题 | 梯度下降的知识整理、Python实现及batch_size参数的总结
author: 钟欣然
date: 2020-12-12 00:44:00 +0800
categories: [优化问题, 梯度下降]
math: true
mermaid: true
---
# 1. 综述

梯度下降法是最小化目标函数J(θ)的一种方法，其中θ为参数，利用目标函数关于参数的梯度的反方向更新参数。学习率η决定达到最小值或局部最小值过程中所采用的的步长的大小。

# 2. 三种形式
区别在于计算目标函数的梯度时用到多少数据，根据数据量的不同，在精度和时间两个方面做出权衡。

## 批梯度下降法（BGD）：整个训练集

$$
\theta=\theta-\eta \cdot \nabla_{\theta} J(\theta)
$$

- 新一次速度慢
- 不能在线更新模型（运行过程中不能增加新的样本）
- 凸误差函数收敛到全局最小值，非凸误差函数收敛到局部最小值

```python
for i in range(nb_epochs):
	params_grad = evaluate_gradient(loss_function, data, params)
	params = params - learning_rate * params_grad
```

## 随机梯度下降法（SGD）：1个训练样本

$$
\theta=\theta-\eta \cdot \nabla_{\theta} J\left(\theta ; x^{(i)} ; y^{(i)}\right)
$$

- 更新一次速度快
- 可以在线学习
- 目标函数会出现波动，波动性使SGD可以跳到新的和潜在更好的局部最优，但也使收敛变慢
- 缓慢减小学习率时，SGD与BGD有相同的收敛行为，凸误差函数收敛到全局最小值，非凸误差函数收敛到局部最小值

```python
for i in range(nb_epochs):
	np.random.shuffle(data)
	for example in data:
    	params_grad = evaluate_gradient(loss_function, example, params)
    	params = params - learning_rate * params_grad
```

## 小批量梯度下降法（MBGD，SGD）：n个训练样本

- $\theta=\theta-\eta\times \nabla_\theta J(\theta;x^{(i:i+n)};y^{(i:i+n)})$
- n：50-256

```python
for i in range(nb_epochs):
	np.random.shuffle(data)
	for batch in get_batches(data, batch_size=50):
    	params_grad = evaluate_gradient(loss_function, batch, params)
    	params = params - learning_rate * params_grad
```

## 梯度下降的python实现

本部分包括批梯度下降、随机梯度下降和小批量梯度下降的Python实现

代码使用了最简单的线性模型作为例子：

- 自变量：随机生成的[0,9]上均匀分布的随机数和[0,15]上均匀分布的随机数
- 因变量：前者的四倍 + 后者的二倍 + 标准正态分布的噪音
- 损失函数：$\frac1n\sum_{i=1}^n(y_i-w_1x_{i1}-w_2x_{i2})^2$

```python
import numpy as np
import time
import matplotlib.pyplot as plt

# 梯度下降算法
def batch_gradient_descent(input_data, output_data, eta, tolerance):
    time_start = time.time()
    w = np.ones((1, 2))
    old_w = np.zeros((1,2))
    iteration = 1
    loss_function = []

    while np.sqrt(np.sum(np.square(w - old_w))) > tolerance:
    #while iteration <= 1000:
        error = output_data - np.dot(w, input_data)
        loss_function.append(sum([c*c for c in (output_data - np.dot(w, input_data))][0])/input_data.shape[1])
        old_w = w
        w = w + eta * np.dot(error, input_data.T)/input_data.shape[1]
        iteration = iteration + 1
        
        if iteration%500 == 0:
            print("迭代次数{}参数值{}".format(iteration,w))
        
                
    time_end = time.time()
    print("耗时{}\n迭代次数{}".format(time_end-time_start,iteration))
        
    print("运行结果：参数为{}".format(w.tolist()[0]))
    
    result = {"time":time_end-time_start, "iterations":iteration, 
              "w":w.tolist()[0], "loss_functions":loss_function}
    
    return result
    


# 随机梯度下降算法
def random_gradient_descent(input_data, output_data, eta, tolerance):
    time_start = time.time()
    w = np.ones((1, 2))
    old_w = np.zeros((1,2))
    iteration = 1
    loss_function = []

    while np.sqrt(np.sum(np.square(w - old_w))) > tolerance:
        for i in range(input_data.shape[1]):
            col_rand_x = input_data[:, i]
            col_rand_y = output_data[i]
            error = col_rand_y - np.dot(w, col_rand_x)
            loss_function.append(sum([c*c for c in (output_data - np.dot(w, input_data))][0])/input_data.shape[1])
            old_w = w
            w = w + eta * error * col_rand_x.T
            iteration = iteration + 1
        
            if iteration%500 == 0:
                print("迭代次数{}参数值{}".format(iteration,w))
            
                
    time_end = time.time()
    print("耗时{}\n迭代次数{}".format(time_end-time_start,iteration))
        
    print("运行结果：参数为{}".format(w.tolist()[0]))
    
        
    result = {"time":time_end-time_start, "iterations":iteration, 
              "w":w.tolist()[0], "loss_functions":loss_function}
    
    return result


# 小批量梯度下降算法
def minibatch_gradient_descent(input_data, output_data, eta, tolerance, batch_size):
    time_start = time.time()
    w = np.ones((1, 2))
    old_w = np.zeros((1,2))
    iteration = 1
    loss_function = []

    #while np.sqrt(np.sum(np.square(w - old_w))) > tolerance:
    while iteration <= 500:
        col_rand_array = np.arange(input_data.shape[1])
        np.random.shuffle(col_rand_array)
        input_data = input_data[:, col_rand_array]
        output_data = output_data[col_rand_array]
        
        for i in range(int(input_data.shape[1]/batch_size)):
            col_rand_x = input_data[:, i*batch_size:(i+1)*batch_size]
            col_rand_y = output_data[i*batch_size:(i+1)*batch_size]
            error = col_rand_y - np.dot(w, col_rand_x)
            loss_function.append(sum([c*c for c in (output_data - np.dot(w, input_data))][0])/input_data.shape[1])
            old_w = w
            w = w + eta * np.dot(error, col_rand_x.T)/batch_size
            iteration = iteration + 1
            if iteration%500 == 0:
                print("迭代次数{}参数值{}".format(iteration,w))
            if iteration > 500:
                break
                
    time_end = time.time()
    print("耗时{}\n迭代次数{}".format(time_end-time_start,iteration))
        
    print("运行结果：参数为{}".format(w.tolist()[0]))
     
    result = {"time":time_end-time_start, "iterations":iteration, 
              "w":w.tolist()[0], "loss_functions":loss_function}
    
    return result

if __name__ == "__main__":
    np.random.seed(1)
    x_data = np.dot([[9,0],[0,15]],np.random.rand(2, 10000))
    y_data = np.dot([4, 2], x_data)+np.random.randn(1, 10000)[0]
    print("梯度下降")
    batch_gradient_descent(x_data, y_data,0.01,0.001)
    print("------------------")
    print("随机梯度下降")
    random_gradient_descent(x_data, y_data,0.0001,0.01)
    print("------------------")
    print("小批量梯度下降")
    minibatch_gradient_descent(x_data, y_data,0.001,0.001,100)
```

**学习率参数的测试**

学习速率$\eta$值的选取依赖于经验 ，过大可能使迭代过程无法收敛，太小又容易陷入局部最值的问题，这里对上述的数据模拟进行学习率参数的测试，代码及结果如下。

```python
result_batch_eta = dict()
eta = [0.01,0.001,0.0001,0.00001]
for i in range(len(eta)):
    result_batch_eta[eta[i]] = batch_gradient_descent(x_data, y_data,eta[i],0.0001)
    plt.plot(range(1,len(result_batch_eta[eta[i]]['loss_functions'])+1),result_batch_eta[eta[i]]['loss_functions'], label=eta[i])

plt.xlabel("iteration")
plt.ylabel("loss_function")
plt.legend(loc="upper right")
```

![学习率的测试结果](https://img-blog.csdnimg.cn/20191026223339115.png#pic_center)

![学习率的测试结果可视化](https://img-blog.csdnimg.cn/20191026223253780.png#pic_center)

## 挑战

- 选择一个合适的学习率时困难的，可以采用学习率调整策略，如在下降值小于某个阈值时减小学习率，但策略和阈值需要预先设定，无法适应数据集特点
- 对所有参数使用一个学习率，对出现次数较少的特征，我们希望执行更大的学习率
- 鞍点：在一个维度上递增在另一个维度上递减，被具有相同误差的点包围

# 3. 更好的算法

## 动量法

- 引入原因：SGD很难通过陡谷，在一个维度上的表面弯曲程度远大于其他维度的区域，这种情况通常出现在局部最优点附近，此时SGD摇摆地通过陡谷的斜坡，沿着底部缓慢到达局部最优点，动量法帮助SGD在相关方向上加速并抑制摇摆的方法
- 方法：将历史步长的更新向量的一个分量增加到当前的更新向量中（γ常取0.9）

$$
v_t=\gamma v_{t-1}+\eta\nabla_\theta J(\theta)
$$

$$
\theta=\theta-v_t
$$

- 通俗理解：从山上推下一个球，球在滚下来的过程中累积动量，对于在梯度点处具有相同的方向的维度，动量项增大，对于在梯度点改变方向的维度，动量项减小，收敛更快并减少摇摆

## Nesterov加速梯度下降法-NAG

- 通俗理解：和动量法相比，希望这个球知道它将要去哪，以至于在重新遇到斜率上升时能够知道减速，有预见性的防止我们前进得太快
- 方法：计算参数关于未来的近似位置的梯度，而不是关于当前的参数的梯度（γ常取0.9）

$$
v_t=\gamma v_{t-1}+\eta\nabla_\theta J(\theta-\gamma v_{t-1})
$$

$$
\theta=\theta-v_t
$$

- 图示：动量法首先计算当前的梯度值（图3中的小的蓝色向量），然后在更新的累积梯度（大的蓝色向量）方向上前进一大步，Nesterov加速梯度下降法NAG首先在先前累积梯度（棕色的向量）方向上前进一大步，计算梯度值，然后做一个修正（绿色的向量）。这个具有预见性的更新防止我们前进得太快，同时增强了算法的响应能力，这一点在很多的任务中对于RNN的性能提升有着重要的意义

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026231624288.png#pic_center)

## Adagrad

- 引入原因：希望更新能适应每一个单独参数，根据每个参数的重要性决定大的或小的更新，Adagrad让学习率适应参数，对于出现次数较少的特征采用更大的学习率，对出现次数较多的特征采用较小的学习率，因此非常适合处理稀疏数据
- 方法：

$$
g_{t,i}=\nabla_\theta J(\theta_i)
$$

在t时刻，基于对$\theta_i$计算过的历史梯度，Adagrad修正了对每一个参数$\theta_i$的学习率

$$
\theta_{t+1,i}=\theta_{t,i}-\frac\eta{\sqrt{G_{t,ii}+\epsilon}}\times g_{t,i}
$$

其中，$G_t\in\mathbb{R}^{d\times d}$是一个对角矩阵，对角线上的元素是直到t时刻为止，所有关于$\theta_i$的梯度的平方和，$\epsilon$是平滑项，用于防止除数为0（t通常大约设置为$e^{-8}$）。比较有意思的是，如果没有平方根的操作，算法的效果会变得很差。整理得

$$
\theta_{t+1}=\theta_t-\frac\eta{\sqrt{G_t+\epsilon}}\odot g_t
$$

- 优点：无需手动调整学习率，通常采用常数0.01
- 缺点：分母中累积梯度的平方，每增加一个正项累加的和会持续增长，学习率最终变得无限小，无法取得额外的信息

## Adadelta

- 引入原因：处理Adagrad学习率单调递减的问题，将梯度的平方递归表示成所有历史梯度平方的均值
- 方法：t时刻的均值只取决于先前的均值和当前的梯度

$$
E[g^2]_t=\gamma E[g^2]_{t-1}+(1-\gamma)g_t^2
$$

现在，我们简单将对角矩阵$G_t$替换成历史梯度的均值$E[g^2]_t$

$$
\triangle\theta_t=-\frac\eta{\sqrt{E[g^2]_t+\epsilon}}·g_t
$$
由于分母仅仅是梯度的均方根（root mean squared, RMS）误差，我们可以简写为

$$
\triangle\theta_t=-\frac\eta{RMS[g]_t+\epsilon}·g_t
$$

问题：是不是⊙
- 问题：作者指出上述更新公式中的每个部分（与SGD，动量法或者Adagrad）并不一致，即更新规则中必须与参数具有相同的假设单位。为了实现这个要求，作者首次定义了另一个指数衰减均值，这次不是梯度平方，而是参数的平方的更新：

$$
E[\triangle\theta^2]_t=\gamma E[\triangle\theta^2]_{t-1}+(1-\gamma)\triangle\theta_t^2
$$

因此，参数更新的均方误差为：

$$
RMS[\triangle\theta]_t=\sqrt{E[\triangle\theta^2]_t+\epsilon}
$$

由于$RMS\left[\triangle\theta\right]_t$是未知的，我们利用参数的均方根误差来近似更新。

利用$RMS[\triangle\theta]_{t-1}$替换先前的更新规则中的学习率$\eta$，最终得到Adadelta的更新规则：

$$
\triangle\theta_t=-\frac{RMS[\triangle\theta]_{t-1}}{RMS[\triangle\theta]_t}g_t
$$

$$
\theta_{t+1}=\theta_t+\triangle\theta_t
$$

- 无需再设置学习率

## RMSprop

- RMSprop是Adadelta的第一个更新向量的特例：

$$
E[g^2]_t=0.9E[g^2]_{t-1}+0.1g_t^2
$$

$$
\theta_{t+1}=\theta_t-\frac\eta{\sqrt{E[g^2]_t+\epsilon}}·g_t
$$

- 同样，RMSprop将学习率分解成一个平方梯度的指数衰减的平均。Hinton建议将γ设为0.9，对于学习率η，一个好的固定值为0.001

## Adam

- 自适应矩估计（Adaptive Moment Estimation）
- 对每一个参数都计算自适应的学习率
- mt和vt分别是对梯度的一阶矩（均值）和二阶矩（非确定的方差）的估计

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

- mt和vt初始化为0向量时，Adam的作者发现它们都偏向于0，尤其是在初始化的步骤和当衰减率很小的时候（例如$β_1$和$β_2$趋向于1）
通过计算偏差校正的一阶矩和二阶矩估计来抵消偏差：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

正如我们再Adadelta和RMSprop中看到的那样，他们利用上述的公式更新参数，由此生成了Adam的更新规则：

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat v_t}+\epsilon}\hat m_t
$$

作者建议$\beta_1$取默认值0.9，$\beta_2$为0.999，$\epsilon$为$10^{-8}$。他们从经验上表明Adam在实际中表现很好，同时，与其他的自适应学习算法相比，其更有优势。

## 可视化

（原图请参照参考资料1或2）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026232258474.png#pic_center)


- a图可以看到不同算法在损失曲面的等高线上走的不同路线。所有的算法都是从同一个点出发并选择不同路径到达最优点。Adagrad，Adadelta和RMSprop能够立即转移到正确的移动方向上并以类似的速度收敛，而动量法和NAG会导致偏离，NAG能够在偏离之后快速修正其路线，因为NAG通过对最优点的预见增强其响应能力。
- b图展示了不同算法在鞍点出的行为，鞍点即为一个点在一个维度上的斜率为正，而在其他维度上的斜率为负。SGD，动量法和NAG在鞍点处很难打破对称性，尽管后面两个算法最终设法逃离了鞍点。而Adagrad，RMSprop和Adadelta能够快速想着梯度为负的方向移动，其中Adadelta走在最前面。

## 算法比较与选择

- RMSprop与Adadelta是Adagrad的扩展形式，用于处理在Adagrad中急速递减的学习率，二者不同的是Adadelta在更新规则中使用参数的均方根进行更新。
- Adam是将偏差校正和动量加入到RMSprop中
- RMSprop、Adadelta和Adam是很相似的算法并且在相似的环境中性能都不错，在优化后期由于梯度变得越来越稀疏，偏差校正能够帮助Adam微弱地胜过RMSprop。综合看来，Adam可能是最佳的选择。
- 通常SGD能够找到最小值点，但是比其他优化的SGD花费更多的时间，SGD可能会陷入鞍点，而不是局部极小值点。

# 4. batch_size参数设置

## 为什么需要batch_size这个参数

**全数据集 Full Batch Learning**

- 数据集较小时的好处
	- 由全数据集确定的方向能够更好地代表总体，从而更准确地朝向极值所在的方向
	- ==Q==由于不同权重的梯度值差别巨大，很难选择一个全局的学习率，全数据集可以使用Rprop只基于梯度符号并且针对性单独更新各权重值
- 数据集较大时的坏处
	- 随着数据集的海量增长和内存限制，一次性载入所有的数据越来越不可行
	- ==Q==以Rprop的方式迭代，会由于各个batch之间的采样差异性，各次梯度修正值相互抵消，无法修正，因此有了RMSProp的妥协方案

**单个样本 Online Learning**

- 坏处
	- 每次修正方向为各自样本的梯度方向，横冲直撞各自为政，达到收敛时间变长

## batch_size的权衡

如果数据集足够充分，那么用一半（甚至少得多）的数据训练算出来的梯度与用全部数据训练出来的梯度是几乎一样的

- 在合理范围内，增大batch_size的好处
	- 内存利用率提高
	- 跑完一次epoch（全数据集）所需的迭代次数减少，相同数据量的处理速度加快
	- 在一定范围内，batch_size越大，其确定的下降方向越准，引起训练震荡越小，模型训练曲线会更加平滑
- 盲目增大batch_size的坏处
	- 内存容量可能不支持
	- 跑完一次epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，花费的时间大大增加，对参数的修正更加缓慢
	- Batch_size增大到一定程度，其确定的下降方向已经基本不再变化
	- 可能会陷入局部最小，小batch_size引入更大的随机性，有可能有更好的效果
	- 模型泛化能力下降，泛化能力指机器学习方法训练出来一个模型，对于已知的数据(训练集)性能表现良好，对于未知的数据(测试集)也应该表现良好的机器能力。 

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026233145579.png#pic_center)

	- 大的batch_size收敛到sharp minimum，而小的batch_size收敛到flat minimum，后者具有更好的泛化能力。两者的区别就在于变化的趋势，一个快一个慢，如下图，造成这个现象的主要原因是小的batch_size带来的噪声有助于逃离sharp minimum（参考文献4）

	![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026233332604.png#pic_center)
	- •	大的batchsize性能下降是因为训练时间不够长，本质上并不是batchsize的问题，在同样的epochs下的参数更新变少了，因此需要更长的迭代次数（参考文献5）
	
## 学习率和batch_size的关系

- 通常当我们增加batchsize为原来的N倍时，要保证经过同样的样本后更新的权重相等，按照线性缩放规则，学习率应该增加为原来的N倍。但是如果要保证权重的方差不变，则学习率应该增加为原来的sqrt(N)倍，目前这两种策略都被研究过，使用前者的明显居多。
- 衰减学习率可以通过增加batch_size来实现类似的效果
- 对于一个固定的学习率，存在一个最优的batch_size能够最大化测试精度

## 小批量梯度下降中batch_size参数的测试

小批量梯度下降中，在一定范围内，batch_size越大，下降的方向越准，所需的迭代次数越小，处理速度越快，但盲目增大也可能有诸多弊端，这里对上述的数据模拟进行batch_size参数的测试，代码及结果如下。

![batch_size参数的测试结果](https://img-blog.csdnimg.cn/20191026223438792.png#pic_center)

![batch_size参数的测试结果可视化](https://img-blog.csdnimg.cn/20191026225155796.png#pic_center)

# 5. 参考资料
1. [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html#fn:7)
2. [梯度下降优化算法综述](https://blog.csdn.net/google19890102/article/details/69942970)，An overview of gradient descent optimization algorithms的中文翻译
3. [深度学习中的batch的大小对学习效果有何影响](https://www.zhihu.com/question/32673260)
4. Keskar N S, Mudigere D, Nocedal J, et al. On large-batch training for deep learning: Generalization gap and sharp minima[J]. arXiv preprint arXiv:1609.04836, 2016.
5. Hoffer E, Hubara I, Soudry D. Train longer, generalize better: closing the generalization gap in large batch training of neural networks[C]//Advances in Neural Information Processing Systems. 2017: 1731-1741.
