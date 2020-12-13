---
title: 西瓜书 | 第一章 绪论
author: 钟欣然
date: 2020-12-7 00:44:00 +0800
categories: [机器学习, 西瓜书]
math: true
mermaid: true
---

## 1.1 引言

#### 1.1.1 机器学习的概念 
致力于研究如何通过计算的手段，利用经验来改善系统自身的性能，关于在计算机上从数据中产生模型的算法，即学习算法。

## 1.2 基本术语

 - 数据集 data set
    * 示例 instance / 样本 sample / 特征向量 feature vector
      + 属性 attribute / 特征 feature
        + 属性值 attribute value
        + 属性空间 attribute space / 样本空间 sample space
      + 维数 dimensionality
     * 训练数据 training data
      + 训练样本 training sample
  * 测试 testing
    + 测试样本 testing sample
  - 学习 learning / 训练 training
  * 假设 hypothesis
  * 真相 / 真实 ground truth
  * 学习器 learner
  * 预测 prediction
    + 标记 label
    + 标记空间 / 输出空间 label space
    + 样例（有标记信息的示例） example
  * 监督学习 supervised learning
    + 分类 classification
      + 二分类 binary classification
        + 正类 positive class
        + 反类 negative class
      + 多分类 multi-class classification
    + 回归 regression
  * 无监督学习 unsupervised learning
    + 聚类 clustering
      + 簇 cluster
- 泛化能力 generalization

## 1.3 假设空间

#### 1.3.1 归纳与演绎
- 归纳 induction：从特殊到一般的泛化过程 generalization
- 演绎 deduction：从一般到特殊的特化过程 specialization

#### 1.3.2 归纳学习
- 广义：从样例中学习
- 狭义：从训练数据中学得概念（concept），也成为概念学习 / 概念形成
  * 最基本的：布尔概念学习（是，不是）

#### 1.3.3 假设空间
把学习过程看做一个在所有假设组成的空间中进行搜索的过程，搜索目标是找到与训练集匹配（fit）的假设，即能够将训练集中的数据判断正确的假设

#### 1.3.4 假设空间示例
以西瓜数据集为例，假设色泽、根蒂、敲声分别有三种可能取值，加上无论取什么值都合适（*），及无论取什么值都不合适（∅），假设空间规模大小为4×4×4+1=65

![西瓜问题的假设空间](https://img-blog.csdnimg.cn/20191021130554553.png#pic_center)

#### 1.3.5 版本空间 version space
与训练集一致的假设集合

## 1.4 归纳偏好

#### 1.4.1 归纳偏好
机器学习算法在学习过程中对某种类型假设的偏好，如“尽可能特殊”的模型或“尽可能一般”的模型，任何一个有效的机器学习算法都有归纳偏好

#### 1.4.2 奥卡姆剃刀 Occam's razor
若有多个假设与观察一致，则选最简单的那个，但更简单如何定义无法确定

#### 1.4.3 NFL定理 No Free Lunch Theorem
不同算法的期望性能相同

以二分类问题为例进行证明：



![NFL定理二分类问题证明1](https://img-blog.csdnimg.cn/20191021134322716.png#pic_center)![NFL定理二分类问题证明2](https://img-blog.csdnimg.cn/20191021134409770.png#pic_center)



**启示：**
若考虑所有潜在的问题，则所有学习算法都一样好，要谈论算法的相对优劣，必须要针对具体的学习问题

## 1.5 发展历程
推理期→知识期→学习期
具体略

## 1.6 应用现状
略
