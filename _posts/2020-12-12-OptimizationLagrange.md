---
title: 优化问题 | 约束优化问题的KKT条件、拉格朗日对偶法、内外点罚函数法
author: 钟欣然
date: 2020-12-12 00:44:00 +0800
categories: [优化问题, 拉格朗日]
math: true
mermaid: true
---

# 1. KKT条件

## 什么是KKT条件
对于具有等式和不等式约束的一般优化问题

$$
\begin{aligned}&minf(\pmb x) \\ &s.t.\;g_j(\pmb x)\leq0(j=1,2,\dots ,l)\\&\;\;\;\;\;\;h_k(\pmb x)=0(k=1,2,\dots ,m)\end{aligned}
$$

KKT条件（Karush-Kuhn-Tucker conditions）给出了判断$\pmb x^*$是否为最优解的必要条件

$$
\begin{aligned}&\frac{\partial f}{\partial x_i}+\sum_{j=1}^m\mu_j\frac{\partial g_j}{\partial x_i}+\sum_{k=1}^l\lambda_k\frac{\partial h_k}{\partial x_i}=0(i=1,2,\dots ,n)\\ &h_k(\pmb x)=0(k=1,2,\dots ,l)\\&\mu_jg_j(\pmb x)=0(j=1,2,\dots ,m)\\&\mu_j\geq 0\end{aligned}
$$

## 等式约束优化问题（Lagrange乘数法）

等式约束优化问题

$$
\begin{aligned}&minf(x_1,x_2,\dots ,x_n) \\ &s.t.\;h_k(x_1,x_2,\dots ,x_n)=0(k=1,2,\dots ,m)\end{aligned}
$$

Lagrange函数

$$
L(\pmb x,\lambda)=f(\pmb x)+\sum_{k=1}^k\lambda_kh_k(\pmb x)
$$

其中$\lambda$为Lagrange乘子

等式约束的极值必要条件：

$$
\begin{aligned}&\frac{\partial L}{\partial x_i}=0(i=1,2,\dots ,n)\\ &\frac{\partial L}{\partial \lambda_k}=0(k=1,2,\dots ,m)\end{aligned}
$$

**理解**：

在无约束优化问题$minf(x_1,x_2,\dots ,x_n)$中，$x_i$为优化变量，我们根据极值的必要条件$\frac{\partial f}{\partial x_i}=0$求出可能的极值点；

在等式约束优化问题中，Lagrange乘数法引入了m个Lagrange乘子，我们可以把$\lambda_k$也看作优化变量，相当于优化变量个数从n增加到n+m个，均对它们求偏导。

## 不等式约束优化问题

- 主要思想：转化——将不等式约束条件变成等式约束条件
- 具体做法：引入松弛变量，松弛变量也是优化变量，也需要一视同仁求偏导

以一元函数为例，

$$
\begin{aligned}&minf(x) \\ &s.t.\;g_1(x)=a-x\leq 0\\&\;\;\;\;\;\;g_2(x)=x-b\leq 0\end{aligned}
$$

对于约束$g_1$和$g_2$，我们引入两个松弛变量$a_1^2$和$b_1^2$，得到

$$
\begin{aligned}&h_1(x)=g_1(x)+a_1^2=a-x+a_1^2=0\\ &h_2(x)=g_2(x)+b_1^2=x-b+a_1^2=0\end{aligned}
$$

取平方项而非$a_1,b_1$是因为$g_1$和$g_2$必须加上一个非负数才能变为等式，若取后者还需加限制条件$a_1\geq 0,b_1\geq 0$，使问题更复杂

由此我们将不等式约束转为等式约束，新的Lagrange函数为

$$
L(x,a_1,b_1,\mu_1,\mu_2)=f(x)+\mu_1(a-x+a_1^2)+\mu_2(x-b+b_1^2)
$$

按照等式约束条件对其求解，得联立方程

$$
\begin{aligned}&\frac{\partial L}{\partial x}=\frac{\partial f}{\partial x}+\mu_1\frac{\partial g_1}{\partial x}+\mu_2\frac{\partial g_2}{\partial x}=0\\&\frac{\partial L}{\partial \mu_1}=g_1+a_1^2=0,\frac{\partial L}{\partial \mu_2}=g_2+b_1^2=0\\&\frac{\partial L}{\partial a_1}=2\mu_1a_1=0,\frac{\partial L}{\partial b_1}=2\mu_2b_1=0\\&\mu_1\geq 0,\mu_2\geq 0\end{aligned}
$$

利用第二行和第三行的四个式子，针对$2\mu_1a_1=0$，我们可得到如下两种情形：

- 情形一：$a_1\neq 0$，则有$\mu_1=0$，约束$g_1$不起作用，且有$g_1<0$
- 情形二：$a_1=0$，则有$\mu_1\geq 0$，约束$g_1$起作用，且有$g_1=0$

合并两种情形，有$\mu_1g_1=0$，约束起作用时，$\mu_1\geq 0$，$g_1=0$，约束不起作用时，$\mu_1=0$，$g_1<0$；

$2\mu_2b_1=0$同理；

由此，必要条件转化为

$$
\begin{aligned}&\frac{\partial f}{\partial x}+\mu_1\frac{\partial g_1}{\partial x}+\mu_2\frac{\partial g_2}{\partial x}=0\\&\mu_1g_1(x)=0,\mu_2g_2(x)=0\\&\mu_1\geq 0,\mu_2\geq 0\end{aligned}\begin{aligned}&\frac{\partial f}{\partial x}+\mu_1\frac{\partial g_1}{\partial x}+\mu_2\frac{\partial g_2}{\partial x}=0\\&\mu_1g_
$$

针对多元的情况，则有

$$
\begin{aligned}&\frac{\partial f(x)}{\partial x_i}+\sum_{j=1}^m\mu_j\frac{\partial g_j(x)}{\partial x}=0(i=1,2,\dots ,n)\\&\mu_jg_j(x)=0(j=1,2,\dots ,m)\\&\mu_j\geq 0(j=1,2,\dots ,m)\end{aligned}
$$

上式即为不等式优化问题的KKT条件，$\mu_j$为KKT乘子，约束起作用时，$\mu_j\geq 0$，$g_j=0$，约束不起作用时，$\mu_j=0$，$g_j<0$

**KKT乘子必须大于等于0**

由于$\frac{\partial f(x)}{\partial x_i}+\sum_{j=1}^m\mu_j\frac{\partial g_j(x)}{\partial x}=0(i=1,2,\dots ,n)$，写成梯度形式有

$$
\nabla f(x)+\sum_{j\in J}\mu_j\nabla g_j(x)=0
$$

J为起约束作用的集合，移项得

$$
-\nabla f(x)=\sum_{j\in J}\mu_j\nabla g_j(x)
$$

注意到梯度为向量，上式表明在约束极小值点处，$f(x)$的梯度一定可以表示成所有起作用的约束在该点的梯度的线性组合

假设只有两个起作用约束，且约束起作用时$g_j(\pmb x)=0$，此时约束在几何上是一簇约束平面，我们假设在$x^k$处取极小值，则$x^k$一定在这两个平面的交线上，且$-\nabla f(x^k),\nabla g_1(x^k),\nabla g_2(x^k)$共面

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101114750284.png#pic_center)

在点$x^k$处沿$x_1Ox_2$平面的截图如下，有两种情况：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101115004646.png#pic_center)

若$-\nabla f$落在$\nabla g_1$和$\nabla g_2$所形成的的锥角区外的一侧，如情形b，作等值面$f(\pmb x)=C$在点$\pmb x^k$的切平面（与$-\nabla f$垂直），我们发现，沿着与负梯度$-\nabla f$成锐角的方向移动，$f(\pmb x)$总能减小，因此$\pmb x^k$仍可沿约束曲面移动，既可减小目标函数值，又不破坏约束条件，所以$\pmb x^k$不是局部极值点

若$-\nabla f$落在$\nabla g_1$和$\nabla g_2$所形成的的锥角内，如情形a，同样作等值面$f(\pmb x)=C$在点$\pmb x^k$的切平面（与$-\nabla f$垂直），沿着与负梯度$-\nabla f$成锐角的方向移动，虽然能使目标函数值减小，但此时任何一点都不在可行区域内，所以此时$\pmb x^k$就是局部极值点

由于$-\nabla f$和$\nabla g_1,\nabla g_2$在一个平面内，所以前者可以看成是后两者的线性组合，又$-\nabla f$落在$\nabla g_1$和$\nabla g_2$所形成的的锥角内，所以线性组合的系数为正，有

$$
-\nabla f(\pmb x^k)=\mu_1\nabla g_1(\pmb x^*)+\mu_2\nabla g_2(\pmb x^*),且\mu_1>0,\mu_2>0
$$

类似地，当有多个不等式约束起作用时，要求$-\nabla f$落在$\nabla g_j$形成的超角锥内。

# 2. 拉格朗日对偶法

在约束最优化问题中，常常利用拉格朗日对偶性（Lagrange duality）将原始问题转化为对偶问题，通过求解对偶问题得到原始问题的解。

## 原始问题

假设$f(x),g_i(x),h_j(x)$为连续可微函数，以下约束最优化问题称为原始问题

$$
\begin{aligned}&minf(x) \\ &s.t.\;g_j(x)\leq0(j=1,2,\dots ,l)\\&\;\;\;\;\;\;h_k(x)=0(k=1,2,\dots ,m)\end{aligned}
$$

引进广义拉格朗日函数（generalized Lagrange function）

$$
L(x,\mu,\lambda)=f(x)+\sum_{j=1}^l\mu_jg_j(x)+\sum_{k=1}^m\lambda_kh_k(x)
$$

考虑x的函数

$$
\theta_P(x)=\underset{\mu,\lambda,\mu_i\geq0}{max}L(x,\mu,\lambda)
$$

其中下标P表示原始问题

假设给定某个x，

- 如果x违反原始问题的约束条件，则$\theta_P(x)=+\infty$
	- 若存在某个j使得$g_j(x)>0$，则可令$\mu_j\rightarrow+\infty$，其余$\mu_j,\lambda_k$均为0
	- 存在某个k使得$h_k(x)\neq 0$，则可令$\beta_j$满足$\beta_jh_j(x)\rightarrow+\infty$，其余$\mu_j,\lambda_k$均为0
- 如果x满足原始问题的约束条件，则$\theta_P(x)=f(x)$

因此

$$
\theta_P(x)=\left\{\begin{array}{l}f(x),x\mathrm{满足原始问题约束}\\+\infty，\mathrm{其他}\end{array}\right.
$$

考虑极小化问题

$$
\underset{x}{min}\theta_P(x)=\underset{x}{min}\underset{\mu,\lambda,\mu_i\geq0}{max}L(x,\mu,\lambda)
$$

它与原始最优化问题等价，即它们有相同的解，问题$\underset{x}{min}\underset{\mu,\lambda,\mu_i\geq0}{max}L(x,\mu,\lambda)$称为广义拉格朗日函数的极小极大问题。

定义原始问题的最优值为原始问题的解：

$$
p^*=\underset{x}{min}\theta_P(x)
$$

## 对偶问题
定义

$$
\theta_D(\mu,\lambda)=\underset{x}{min}L(x,\mu,\lambda)
$$

再考虑极大化$\theta_D(\mu,\lambda)$ 

$$
\underset{\mu,\lambda,\mu_i\geq0}{max}\theta_D(\mu,\lambda)=\underset{\mu,\lambda,\mu_i\geq0}{max}\underset{x}{min}L(x,\mu,\lambda)
$$

问题$\underset{\mu,\lambda,\mu_i\geq0}{max}\underset{x}{min}L(x,\mu,\lambda)$称为拉格朗日函数的极大极小问题

以下约束最优化问题称为原始问题的对偶问题

$$
\begin{aligned}&\underset{\mu,\lambda}{max}\theta_D(\mu,\lambda)=\underset{\mu,\lambda}{max}\;\underset{x}{min}L(x,\mu,\lambda)\\&s.t. \mu_i\geq0,i=1,2,\dots ,l\end{aligned}
$$

定义对偶问题的最优值为对偶问题的值

$$
d^*=\underset{\mu,\lambda,\mu_i\geq0}{max}\theta_D(\mu,\lambda)
$$

**对偶函数与共轭函数**

$$
\left.\begin{array}{rl}原函数：&{f(x): R^{n} \rightarrow R} \\共轭函数：&{f^{*}(y): R^{n} \rightarrow R}\end{array}\right\} \rightarrow f^{*}(y)=\sup _{x \in \mathrm{dom} f}\left(y^{T} x-f(x)\right)
$$

其中，$\sup (g(x))$表示函数$g(x)$在整个定义域$x\in \mathrm{dom} g$中的最大值，y则取任意常数或常向量，对每一个y值，$f^*(y)$都会对应一个值，将y拓展到整个取值范围后即可得到关于y的函数$f^*(y)$，即函数$f(x)$的共轭函数。

考虑线性约束下的最优化问题

$$
\begin{array}{l}{\min f(x)} \\ s.t.\;Ax\leq b,Cx=d\end{array}
$$

将其对偶函数凑出共轭函数形式：

$$
\begin{aligned}g(\lambda, \nu)&=\inf _{x}\left(f(x)+\lambda^{T}(A x-b)+\nu^{T}(C x-d)\right)\\&=-\lambda^{T} b-\nu^{T} d+\inf _{x}\left(f(x)+\lambda^{T} A x+\nu^{T} C x\right) \\&=-\lambda^{T} b-\nu^{T} d-\sup _{x}\left(\left(-A^{T} \lambda-C^{T} \nu\right)^{T} x-f(x)\right) \\&=-\lambda^{T} b-\nu^{T} d-f^{*}\left(-A^{T} \lambda-C^{T} \nu\right)\end{aligned}
$$

即线性约束下的对偶函数可以用共轭函数表示，其自变量为拉格朗日乘子的线性组合。

## 原始问题与对偶问题的关系

**定理1**（弱对偶性：$d^*\leq p^*$）

若原始问题和对偶问题都有最优值，则

$$
d^*=\underset{\mu,\lambda,\mu_i\geq0}{max}\underset{x}{min}L(x,\mu,\lambda)\leq \underset{x}{min}\underset{\mu,\lambda,\mu_i\geq0}{max}L(x,\mu,\lambda)=p^*
$$

证明：对任意的$\mu,\lambda,x$，有

$$
\theta_D(\mu,\lambda)=\underset{x}{min}L(x,\mu,\lambda)\leq L(x,\mu,\lambda)\leq \underset{\mu,\lambda,\mu_i\geq0}{max}L(x,\mu,\lambda)=\theta_P(x)
$$

由于原始问题和对偶问题均有最优值，所以

$$
d^*=\underset{\mu,\lambda,\mu_i\geq0}{max}\theta_D(\mu,\lambda)\leq \underset{x}{min}\theta_P(x)=p^*
$$

**推论1**（强对偶性：$d^*=p^*$）

设$x^*$和$\mu^*,\lambda^*$分别是原始问题和对偶问题的可行解，并且$d^*=p^*$，则$x^*$和$\mu^*,\lambda^*$分别是原始问题和对偶问题的最优解

在某些条件下，原始问题和对偶问题的最优值相等，即$d^*=p^*$，此时可以用解对偶问题替代解原始问题

**定理2**

假设$f(x),g_j(x)$是凸函数，$h_k(x)$是仿射函数（由一阶多项式构成的函数，$h_k(x)=Ax+b$，A是矩阵，x、b是向量），并且假设不等式约束$g_j(x)$是严格可行的，即存在x，对所有$j$有$g_j(x)<0$，则存在$x^*,\mu^*,\lambda^*$使$x^*$是原始问题的解，$\mu^*,\lambda^*$是对偶问题的解，并且$d^*=p^*=L(x^*,\mu^*,\lambda^*)$

**定理3**

假设$f(x),g_j(x)$是凸函数，$h_k(x)$是仿射函数，并且假设不等式约束$g_j(x)$是严格可行的，则$x^*$和$\mu^*,\lambda^*$分别是原始问题和对偶问题的解的充要条件是$x^*,\mu^*,\lambda^*$满足KKT条件

 - [ ] 推论1、定理2、定理3的证明

## 对偶上升法

**换个角度理解拉格朗日对偶函数**

对优化问题

$$
\begin{array}{l}{\min f_0(x)} \\ {\text {s.t.} f_{i}(x) \leq 0(i=1,2, \ldots, m)} \\ {\;\;\;\;\;h_{i}(x)=0(i=1,2, \ldots, p)}\end{array}
$$

拉格朗日函数为

$$
L(x, \lambda, \nu)=f_{0}(x)+\sum_{i=1}^{m} \lambda_{i} f_{i}(x)+\sum_{i=1}^{p} \nu_{i} h_{i}(x)
$$

拉格朗日对偶函数为

$$
g(\lambda, \nu)=\inf _{x \in D} L(x, \lambda, \nu)
$$

可以将其看作是x取不同值时一簇曲线的下界（绿线）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191104141625504.png#pic_center)

当$\lambda\geq 0$时，对于最优化问题的解$\bar x$，两个约束条件都非正：

$$
\lambda_{i} f_{i}(\bar{x})<=0, \nu_{i} h_{i}(\bar{x})=0
$$

于是，该解对应的曲线不超过原始问题最优解：

$$
L(\bar{x}, \lambda, \nu) \leq f_{0}(\bar{x})
$$

进一步，所有曲线的下界不超过原问题最优解：

$$
g(\lambda, \nu) \leq f_{0}(\bar{x})
$$

换言之，拉格朗日对偶函数是最优化值的下界。又上图绿线上的最高点，是对最优值下界的最好估计，这个问题即为原始问题的拉格朗日对偶问题。如果强对偶条件成立，对偶问题存在最优解$\bar \lambda,\bar \nu$，则原始问题$f_0(x)$的最优解也是$L(x,\bar \lambda,\bar \nu)$的最优解。$L(x,\bar \lambda,\bar \nu)$是x的函数，相当于在图中$[\lambda,\nu]=[\bar \lambda,\bar \nu]$对应的竖线上，查找值最小曲线对应的x。求解步骤归纳如下：

- 求解$\max g(\lambda,\mu)$得到$\bar \lambda,\bar \nu$
- 求解$\min L(x,\bar \lambda,\bar \nu)$得到$\bar x$

**对偶上升法**

设第k次迭代得到原始问题解为$x^k$，对偶问题的解为$\lambda^k,\nu^k$

1. 假设$\lambda^k,\nu^k$已为对偶问题的最优解，最小化$L(x,\lambda^k,\nu^k)$得到原问题最优解$x^{k+1}$

$$
x^{k+1}=\arg \min _{x} L\left(x, \lambda^{k}, \nu^{k}\right)
$$

2. 在该位置使用梯度上升法更新对偶问题的解：

$$
\begin{aligned} \lambda^{k+1} &=\lambda^{k}+\left.\alpha \cdot \frac{\partial L(x, \lambda, \nu)}{\partial \lambda}\right|_{x=x^{k+1}, \lambda=\lambda^{k}, \nu=\nu^{k}} \\ \nu^{k+1} &=\lambda^{k}+\left.\alpha \cdot \frac{\partial L(x, \lambda, \nu)}{\partial \nu}\right|_{x=x^{k+1}, \lambda=\lambda^{k}, \nu=\nu^{k}} \end{aligned}
$$

下图灰线展示出求解的变化：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191104145648390.png#pic_center)

## 对偶分解法

假设目标函数是可分解的：

$$
\begin{array}{l}{f(x)=\sum_{i=0}^{n} f_{i}\left(x_{i}\right)} \\ {\text { s.t. } \quad A x=b, \forall x=\left(x_{0}, x_{1}, \cdots, x_{n}\right)^{T}}\end{array}
$$

则拉格朗日函数是可分解的：

$$
\begin{aligned} L(x, \lambda) &=\sum_{i=0}^{n} L\left(x_{i}, \lambda\right) \\ &=\sum_{i=0}^{n} f_{i}\left(x_{i}\right)+\lambda A_{i} x-\lambda b_{i} \end{aligned}
$$

所以对偶上升法中的第一步可以修改为

$$
x_{i}^{k+1}=\arg \min _{x_{i}} L_{i}\left(x_{i}, \lambda^{k}\right)
$$

即可并行计算$x^{k+1}$的每个元素，从而快速得到$x^{k+1}$

# 3. 内外点罚函数法
罚函数的基本思想是构造辅助函数，把原来的约束问题转化为求极小化辅助函数的无约束问题，如何构造辅助函数是求解问题的首要问题。

## 外点罚函数法
构造辅助函数$F_\mu:\mathbb{R}^n\rightarrow\mathbb{R}(\mu>0)$，构造函数在可行域内部与原问题的取值相同，在可行域外部取值远远大于目标函数的取值

- 对于等式约束问题：

$$
\begin{aligned}&minf(x) \\ &s.t.\;h_k(x)=0(k=1,2,\dots ,m)\end{aligned}
$$

可定义辅助函数：

$$
F(x,\mu)=f(x)+\mu\sum_{k=1}^mh_k^2(x)
$$

- 对于不等式约束问题：

$$
\begin{aligned}&minf(x) \\ &s.t.\;g_j(x)\leq0(j=1,2,\dots ,l)\end{aligned}
$$

可定义辅助函数：

$$
F(x,\mu)=f(x)+\mu\sum_{j=1}^l(max\{0,g_j(x)\})^2
$$

- 对于一般问题，可定义辅助函数：

$$
F(x,\mu)=f(x)+\mu P(x)
$$

其中

$$
P(x)=\sum_{j=1}^l\phi(g_j(x))+\sum_{k=1}^m\psi(h_k(x))
$$

$$
\phi(z)\left\{\begin{array}{l}=0,z\leq0\\>0,z>0\end{array}\right.,\psi(z)\left\{\begin{array}{l}=0,z=0\\>0,z\neq0\end{array}\right.
$$

典型取法有

$$
\phi=(max\{0,g_j(x)\})^\alpha,\psi=\vert h_k(x)\vert^\beta,\alpha\geq1,\beta\geq1
$$

通过这些辅助函数，可以把约束问题转换为无约束问题$minF(x,\mu)$，其中$\mu$是很大的数，通常取一个趋向于无穷大的严格递增正数列$\{\mu_k\}$

**具体步骤**：

1. 给定初始点$x^{(0)}$，初始罚因子$\mu_1$，放大系数$c>0$，允许误差$\epsilon>0$，设$k=1$
2. 以$x^{(k-1)}$为初始点，求解无约束问题$minF(x,\mu)=f(x)+\mu_kP(x)$，得极小点$x^{(k)}$
3. 若$\mu_kP(x^{(k)})<\epsilon$，停止，得极小点$x^{(k)}$；否则，令$\mu_{k+1}=c\mu_k,k=k+1$，转步骤二

## 内点罚函数法
从可行域内部逼近问题的解，构造辅助函数，使得该函数在严格可行域外无穷大，当自变量趋于可行域边界时，函数值趋于无穷大。适用于不等式约束问题：

$$
\begin{aligned}&minf(x) \\ &s.t.\;g_j(x)\leq0(j=1,2,\dots ,l)\end{aligned}
$$

可行域为$S=\{x\vert g_j(x)\leq0,j=1,2,\dots ,l\}$

定义障碍函数：

$$
F(x,\mu)=f(x)+\mu B(x)
$$

当自变量趋于可行域边界时，$B(x)\rightarrow +\infty$，当$\mu\rightarrow0$时，$minF_\mu$的解趋于原始问题的解

常用辅助函数

$$
B(x)=\sum_{j=1}^l-\frac1{g_j(x)}
$$

$$
B(x)=-\sum_{j=1}^l\ln (-g_j(x))
$$

**具体步骤**：

1. 给定初始点$x^{(0)}$，初始罚因子$\mu_1$，缩小系数$\beta\in (0,1)$，允许误差$\epsilon>0$，设$k=1$
2. 以$x^{(k-1)}$为初始点，求解无约束问题$minF(x,\mu)=f(x)+\mu_kB(x)$，得极小点$x^{(k)}$
3. 若$\mu_kB(x^{(k)})<\epsilon$，停止，得极小点$x^{(k)}$；否则，令$\mu_{k+1}=\beta\mu_k,k=k+1$，转步骤二

# 4. 参考资料
1. [浅谈最优化问题的KKT条件](https://zhuanlan.zhihu.com/p/26514613)
2. [《统计学习方法》（李航）附录C](https://vdisk.weibo.com/s/sLX7IJK7Id7D)
3. [拉格朗日函数、对偶上升法、对偶分解法](https://blog.csdn.net/deepinC/article/details/79341632)
4. [【优化】对偶上升法(Dual Ascent)超简说明](https://blog.csdn.net/shenxiaolu1984/article/details/78175382)
5. [罚函数法求解约束问题最优解](https://blog.csdn.net/xuehuafeiwu123/article/details/53726930)
