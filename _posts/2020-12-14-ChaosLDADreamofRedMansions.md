---
title: 主题模型 | LDA方法分析红楼梦各回主题与宝玉CP
author: 钟欣然
date: 2020-12-11 00:44:00 +0800
categories: [杂谈, 主题模型]
math: true
mermaid: true
---

[点击此处下载数据](https://github.com/zhongxinran/LDA_in_The_Dream_of_Red_Mansion)

# 1. 数据预处理
## 准备工作

引入需要的模块，设置显示方式，修改工作路径，如未安装jieba包需要在命令行中输入pip install jieba安装


```python
import os
import re
import jieba
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.image as mpimg
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.interpolate import spline
import random
from scipy.cluster import hierarchy
from sklearn import decomposition as skldec

# 设置字体
font = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\STXINWEI.TTF')
# 设置pandas显示方式
pd.set_option("display.max_rows",10)
pd.options.mode.chained_assignment = None  # default='warn'

# 设置显示图像的方式
%matplotlib inline
%config InlineBackend.figure_format = "retina"
%config InlineBackend.figure_format = "retina"

os.chdir("C:\Aruc\learning\大四上\贝叶斯\LDA")
print(os.getcwd())
```

    C:\Aruc\learning\大四上\贝叶斯\LDA


## 读入文本

首先读入红楼梦文本，并按照章节处理为列表，再读入停用词表（合并了红楼梦停用词表、哈工大停用词表、四川大学停用词表、中文停用词表等）


```python
# 读入红楼梦文本
f = open('红楼梦.txt',encoding='utf-8')
quanwen = f.read()
quanwen.replace('\n','')
# 出现“第xx回 ”则断开
quanwen_list = re.split('第[\u4e00-\u9fa5]{1,3}回 ',quanwen)[1:121]

# 读入停用词表
f = open('stopword1.txt')
stop_words = f.readlines()
for i in range(len(stop_words)):
    stop_words[i] = stop_words[i].replace('\n','')
    
print('红楼梦共有{}章，{}个停用词'.format(len(quanwen_list),len(stop_words)))
```

    红楼梦共有120章，1604个停用词


## 分词、去停词

每章分别分词、去停词，同时将长度为1的字符删掉，将结果分章节保存


```python
# 分词、去停词
jieba.load_userdict('红楼梦分词词典.txt')
quanwen_fenci = []
for i in range(len(quanwen_list)):
    temps = list(jieba.cut(quanwen_list[i]))
    quanwen_fenci.append([])
    for temp in temps:
        # 将长度为1的字符删掉
        if len(temp) > 1 and temp not in stop_words:
            quanwen_fenci[i].append(temp)

# 将结果分章节保存
for i in range(len(quanwen_fenci)):
    with open('红楼梦分章节分词/{}.json'.format(i),'w',encoding='utf-8') as f:
        json.dump(quanwen_fenci[i],f)

# 查看第一章前100分分词结果
print(quanwen_fenci[0][0:100])
```

    ['甄士隐', '梦幻', '识通灵', '贾雨村', '风尘', '闺秀', '开卷', '第一回', '作者', '自云', '因曾', '历过', '梦幻', '之后', '真事', '隐去', '通灵', '撰此', '石头记', '一书', '故曰', '甄士隐', '但书中', '所记', '何事', '何人', '自又云', '风尘碌碌', '一事无成', '念及', '当日', '女子', '细考', '觉其', '行止', '见识', '之上', '堂堂', '须眉', '诚不若', '裙钗', '实愧', '有余', '无益', '之大', '无可如何', '之日', '自欲', '已往', '所赖', '天恩祖', '锦衣', '纨绔', '饫甘餍肥', '父兄', '教育', '之恩', '师友', '规谈', '之德', '今日', '一技无成', '半生', '潦倒', '之罪', '编述', '一集', '以告', '天下人', '罪固', '闺阁', '中本', '历历', '有人', '不可', '不肖', '自护己', '一并', '泯灭', '今日', '之茅', '椽蓬', '瓦灶', '绳床', '晨夕', '风露', '阶柳庭花', '未有', '襟怀', '笔墨', '未学', '下笔', '无文', '假语', '村言', '演出', '一段', '故事', '亦可', '闺阁']



```python
# 绘制各章节分词后词数
chapter_lens = [len(fenci) for fenci in quanwen_fenci]
p1 = plt.bar(range(len(chapter_lens)),chapter_lens)
plt.title("各章节分词后词数", fontproperties=font,size = 15)
plt.show()
```


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230170620475.png)

每个章节的词数在1500上下波动



# 2. 基本情况

## 全文词频及词云图

统计全文的词频并展示结果，绘制全文及部分章节的词云图


```python
# 统计全文的词频
word_df = pd.DataFrame({'word':np.concatenate(quanwen_fenci)})
word_stat = word_df.groupby(by = 'word')['word'].agg({'number':np.size})
# 调整为按词频由高到低展示
word_stat = word_stat.reset_index().sort_values(by='number',ascending=False)
word_stat
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214224409947.png#pic_center)

全文共有39882个词语，宝玉、贾母、凤姐、袭人、黛玉等主角人物出现次数最多


```python
# 将词频数据框转化为字典
word_dict = {}
for key,value in zip(word_stat.word,word_stat.number):
    word_dict[key] = value
    
# 绘制词云
back_coloring = mpimg.imread('词云图片.jpg')
wc = WordCloud(font_path='C:\Windows\Fonts\STXINWEI.TTF',
               margin=5, width=2000, height=2000,
               background_color="white",
               max_words=800,
               mask=back_coloring,
               max_font_size=400,
               random_state=42,
               ).generate_from_frequencies(frequencies=word_dict)

plt.figure(figsize=(8,8))
plt.imshow(wc)
plt.axis('off')
plt.title('红楼梦词云图',FontProperties=font,size = 15)
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230170718562.png)



## 人物出场情况
读入人物名，查看各人物在各章的出场情况，并按照总出现次数由多到少排序，绘制出场情况曲线


```python
# 读入人物名
f = open('红楼梦人物名.txt',encoding='utf-8')
names = f.readlines()
for i in range(len(names)):
    names[i] = names[i].replace('\n','')
    
# 输出前50个人物名
print(names[:50])
```

    ['黛玉', '宝钗', '贾演', '贾寅', '贾源', '贾法', '贾代化', '贾代善', '贾代儒', '贾代修', '贾敷', '贾敬', '贾赦', '贾政', '贾敏', '贾敕', '贾效', '贾敦', '贾珍', '贾琏', '贾珠', '贾母', '贾宝玉', '宝玉', '贾环', '贾瑞', '贾璜', '贾琮', '贾珩', '贾㻞', '贾珖', '贾琛', '贾琼', '贾璘', '贾元春', '贾迎春', '贾探春', '贾惜春', '贾蓉', '贾兰', '贾蔷', '贾菌', '贾芸', '贾芹', '贾萍', '贾菖', '贾菱', '贾蓁', '贾藻', '贾蘅']



```python
# 计算每个词在每章的词频
for i in range(len(quanwen_fenci)):
    word_cha_stat = pd.DataFrame({'word':quanwen_fenci[i]}).groupby(by = 'word')['word'].agg({'chapter{}'.format(i):np.size})
    word_stat = pd.merge(word_stat, word_cha_stat, how='left', on='word')
word_stat = word_stat.where(word_stat.notnull(), 0)

# 仅保留人物的词频
word_name_stat = pd.DataFrame(columns = word_stat.columns)
for i in range(len(word_stat.word)):
    if word_stat.word[i] in names:
        word_name_stat = word_name_stat.append(word_stat.loc[i])

# 输出人物-词频矩阵
word_name_stat
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214224719700.png#pic_center)



```python
# 绘制出场最多的八个人物的出场曲线
fig = plt.figure(figsize=(10,5))
for i in range(8):
    plt.subplot(2,4,1+i)
    # 平滑
    plt.plot(np.linspace(1,120,300),spline(range(120),word_name_stat.iloc[i,2:122].tolist(),np.linspace(1,120,300)))
    plt.title(word_name_stat.iloc[i,0],FontProperties=font,size = 15)
    plt.xticks([])
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230170802544.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzk1MTMyOA==,size_16,color_FFFFFF,t_70)
      
    




## 宝玉的cp

在林妹妹、宝姐姐和袭人中，谁和宝玉的cp感最强呢？首先关注每个女性人物和宝玉的词频曲线波动趋势的吻合性，再关注每个女性人物和宝玉的cp指数，指数计算方式为计算每章中该女性人物的出现次数，乘以这一章中宝玉出场的比例（当前章节出现次数/全文总出现次数），最后对全文120个章节求和


```python
# 计算各女性人物的词频和cp指数
cps = ['黛玉','宝钗','袭人']
cp_nums = {}
baoyu_num = [num/word_name_stat.iloc[word_name_stat.word.tolist().index('宝玉'),1] for num in word_name_stat.iloc[word_name_stat.word.tolist().index('宝玉'),2:122]]
for cp in cps:
    cp_num = word_name_stat.iloc[word_name_stat.word.tolist().index(cp),2:122]
    cp_index = round(sum(np.multiply(np.array(baoyu_num),np.array(cp_num)).tolist()),2)
    cp_nums[cp] = [cp_num,cp_index]

# 分别绘制各女性角色和宝玉的词频曲线
fig = plt.figure(figsize=(10,3))
for i in range(len(cps)):
    plt.subplot(1,3,1+i)
    l1 = plt.plot(range(120),word_name_stat.iloc[word_name_stat.word.tolist().index('宝玉'),2:122])
    l2 = plt.plot(range(120),cp_nums[cps[i]][0])
    plt.title(cps[i],FontProperties=font,size = 15)
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230170826739.png)


上图中蓝色线表示宝玉的词频曲线，橙色线表示女性角色的词频曲线，不难发现，袭人的词频曲线和宝玉的波动情况吻合度最高，接下来关注cp指数


```python
cp_df = pd.DataFrame(cp_nums).drop([0])
cp_df.index = ["cp指数"]
cp_df
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214224813817.png#pic_center)




cp指数再一次印证了，袭人和宝玉一起出场的频率最高，其次是黛玉

# 3. LDA分析红楼梦各回的主题
## 数据准备


```python
# 将数据调整为CountVectorizer可调用的形式
articals = []
for fenci in quanwen_fenci:
    articals.append(" ".join(fenci))
# 展示前500个字符
articals[0][0:500]
```




    '甄士隐 梦幻 识通灵 贾雨村 风尘 闺秀 开卷 第一回 作者 自云 因曾 历过 梦幻 之后 真事 隐去 通灵 撰此 石头记 一书 故曰 甄士隐 但书中 所记 何事 何人 自又云 风尘碌碌 一事无成 念及 当日 女子 细考 觉其 行止 见识 之上 堂堂 须眉 诚不若 裙钗 实愧 有余 无益 之大 无可如何 之日 自欲 已往 所赖 天恩祖 锦衣 纨绔 饫甘餍肥 父兄 教育 之恩 师友 规谈 之德 今日 一技无成 半生 潦倒 之罪 编述 一集 以告 天下人 罪固 闺阁 中本 历历 有人 不可 不肖 自护己 一并 泯灭 今日 之茅 椽蓬 瓦灶 绳床 晨夕 风露 阶柳庭花 未有 襟怀 笔墨 未学 下笔 无文 假语 村言 演出 一段 故事 亦可 闺阁 昭传 复可悦 世之目 破人 愁闷 宜乎 故曰 贾雨村 此回 中凡用 提醒 阅者 眼目 此书 立意 本旨 列位 看官 此书 从何而来 说起 根由 虽近 荒唐 深有 趣味 来历 注明 方使 阅者 了然 不惑 原来 女娲 炼石补天 大荒山 无稽 崖练成 高经 十二 方经 二十四丈 顽石 三万 六千五百 一块 皇氏 只用 三万 六千五百 一块 便弃 此山 青'




```python
# 建立能用于模型训练的章节-词频矩阵
tf_vectorizer = CountVectorizer(max_features=10000)
tf = tf_vectorizer.fit_transform(articals)
# 查看词
print(tf_vectorizer.get_feature_names()[10:20])
# 查看词频
tf.toarray()[20:50,200:800]
```

    ['一两天', '一两日', '一两样', '一个个', '一个头', 
    '一个月', '一串', '一丸', '一乘', '一事']
    array([[0, 1, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 1, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0]], dtype=int64)




## 训练模型


```python
# 主题数目
n_topics = 5
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=50, 
                                learning_method='online',                 
                                learning_offset=50., random_state=0)

# 模型应用于数据
lda.fit(tf)
# 得到每个章节属于某个主题的可能性
chapter_top = pd.DataFrame(lda.transform(tf),index=range(120),columns=np.arange(n_topics)+1)
chapter_top
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214225050689.png#pic_center)



## 模型结果


```python
# 每一行的和
chapter_top.apply(sum,axis=1).values
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1.])




```python
# 查看每一列的最大值
chapter_top.apply(max,axis=1).values
```




    array([0.9993818 , 0.99919544, 0.89569005, 0.83185936, 0.99929602,
           0.99939649, 0.99941162, 0.99931902, 0.99921217, 0.99904521,
           0.99926698, 0.99892742, 0.9991443 , 0.9992524 , 0.99919204,
           0.99942508, 0.99934276, 0.73443672, 0.99946176, 0.99911223,
           0.99926225, 0.99932175, 0.99919958, 0.99949226, 0.99947284,
           0.99940219, 0.99926442, 0.9995133 , 0.99906163, 0.99926482,
           0.99934065, 0.99919775, 0.99916241, 0.99940487, 0.99946205,
           0.99934675, 0.99938311, 0.99909089, 0.99922734, 0.99950176,
           0.9992686 , 0.99937141, 0.9993522 , 0.99934852, 0.99936737,
           0.99945854, 0.99936432, 0.9992998 , 0.99942645, 0.9992734 ,
           0.9993323 , 0.99944179, 0.99945389, 0.99910061, 0.99947922,
           0.99947456, 0.99958927, 0.99933946, 0.99910388, 0.99938082,
           0.99927962, 0.99952662, 0.99957228, 0.99958762, 0.99937388,
           0.99911102, 0.999547  , 0.99941656, 0.99939294, 0.99878177,
           0.99953144, 0.99940166, 0.99938949, 0.99959678, 0.99952475,
           0.999307  , 0.99957224, 0.9706532 , 0.99912097, 0.99936234,
           0.99936806, 0.99948991, 0.9995026 , 0.99944047, 0.99949171,
           0.99916148, 0.9992805 , 0.99937959, 0.99930451, 0.99932157,
           0.99920972, 0.99507871, 0.99935829, 0.9994835 , 0.99939538,
           0.99940215, 0.99960147, 0.99934493, 0.99920179, 0.99924936,
           0.999453  , 0.99906016, 0.99935927, 0.99931767, 0.99922597,
           0.99930695, 0.999353  , 0.99940975, 0.99954764, 0.99935427,
           0.99939983, 0.99940343, 0.99936619, 0.99909992, 0.99939138,
           0.95543441, 0.9994236 , 0.99945312, 0.99959164, 0.88327963])




```python
# 每个主题生成各个词的概率
Red_lda = pd.DataFrame(lda.components_,index=np.arange(n_topics)+1,columns=tf_vectorizer.get_feature_names())
Red_lda
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214225207273.png#pic_center)





```python
# 查看每个主题的前20个关键词
n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()
fig = plt.figure(figsize=(18,12))
j = 0
for topic_id,topic in enumerate(lda.components_):
    j += 1
    topword = pd.DataFrame({"word":[tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]],
                            "componets":topic[topic.argsort()[:-n_top_words - 1:-1]]})
    topword.sort_values(by = "componets").plot(kind = "barh",x = "word",y = "componets",legend=False,
                                               color='cornflowerblue',subplots=True,ax=fig.add_subplot(2,3,j))
    plt.yticks(FontProperties = font,size=15)
    plt.ylabel("")
    plt.title("Topic %d" %(topic_id+1),FontProperties = font,size= 15)
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230170922359.png)



```python
# 查看每个主题的词云
n_top_words = 100

back_coloring = mpimg.imread('词云图片.jpg')
wc = WordCloud(font_path='C:\Windows\Fonts\STXINWEI.TTF',
               margin=5, width=2000, height=2000,
               background_color="white",
               max_words=800,
               mask=back_coloring,
               max_font_size=400,
               random_state=42,
               )
fig = plt.figure(figsize=(18,12))

j = 0
for topic_id,topic in enumerate(lda.components_):
    j += 1
    topword = pd.DataFrame({"word":[tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]],
                            "componets":topic[topic.argsort()[:-n_top_words - 1:-1]]})
    topword = topword.sort_values(by = "componets")
    
    word_dict = {}
    for key,value in zip(topword.word,topword.componets):
        word_dict[key] = round(value)
    plt.subplot(2,3,j)
    plt.imshow(wc.generate_from_frequencies(frequencies=word_dict))
    plt.axis('off')
    plt.title("Topic %d" %(topic_id+1),FontProperties = font,size= 15)
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230170939162.png)



# 4. 回应课程任务

## 任务1

随机抽取40篇作为训练样本，训练完成后，对剩下的40回进行分类，判断和训练样本中的那些回比较接近


```python
# 生成训练集和测试集的序号
test_index = [i for i in range(80)]
train_index = random.sample(test_index,40)
train_fenci = []
for i in train_index:
    test_index.remove(i)
    train_fenci.append(quanwen_fenci[i])
    
# 将数据调整为CountVectorizer可调用的形式
articals = []
for fenci in train_fenci:
    articals.append(" ".join(fenci))
# 建立能用于模型训练的章节-词频矩阵
tf_vectorizer = CountVectorizer(max_features=10000)
tf = tf_vectorizer.fit_transform(articals)
# 主题数目
n_topics = 5
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=500, 
                                learning_method='online',                 
                                learning_offset=50., random_state=0)
# 模型用于数据
lda.fit(tf)
# 得到每个章节属于某个主题的可能性
chapter_top = pd.DataFrame(lda.transform(tf),index=train_index,columns=np.arange(n_topics)+1)
chapter_top.sort_index()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214225313813.png#pic_center)







```python
# 计算测试集的主题
test_fenci = []
for i in test_index:
    test_fenci.append(quanwen_fenci[i])
articals_test = []
for fenci in test_fenci:
    articals_test.append(" ".join(fenci))
tf_test = tf_vectorizer.fit_transform(articals_test)
chapter_top_test = pd.DataFrame(lda.transform(tf),index=test_index,columns=np.arange(n_topics)+1)
chapter_top_test.sort_index()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214225402471.png#pic_center)







```python
# 对每个测试集章节，找到和它最像的训练集章节
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    param vector_a: 向量 a 
    param vector_b: 向量 b
    return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
result = []
for i in range(len(test_index)):
    cos_sims = []
    for j in range(len(train_index)):
        cos_sims.append(cos_sim(chapter_top_test.loc[test_index[i]],chapter_top.loc[train_index[j]]))
    result.append([test_index[i]+1,train_index[cos_sims.index(max(cos_sims))]+1,max(cos_sims)])
result_df = pd.DataFrame(result,columns=["测试集章节","最相似的训练集章节","相似度"])
result_df
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214225523881.png#pic_center)






## 任务2

使用全部八十回，在LDA分析基础上，进行分类，判断哪些回在主题分布上比较接近


```python
# 生成训练集和测试集的序号
train_index_2 = [i for i in range(80)]
train_fenci_2 = []
for i in train_index_2:
    train_fenci_2.append(quanwen_fenci[i])
# 将数据调整为CountVectorizer可调用的形式
articals_2 = []
for fenci in train_fenci_2:
    articals_2.append(" ".join(fenci))
len(articals_2)
# 建立能用于模型训练的章节-词频矩阵
tf_vectorizer = CountVectorizer(max_features=10000)
tf_2 = tf_vectorizer.fit_transform(articals_2)
# 主题数目
n_topics_2 = 5
lda_2 = LatentDirichletAllocation(n_topics=n_topics_2, max_iter=25, 
                                  learning_method='online',
                                  learning_offset=50., random_state=0)
# 模型用于数据
lda_2.fit(tf_2)
# 得到每个章节属于某个主题的可能性
chapter_top_2 = pd.DataFrame(lda_2.transform(tf_2),index=train_index_2,columns=np.arange(n_topics_2)+1)
chapter_top_2.sort_index()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201214225613754.png#pic_center)


```python
disMat = hierarchy.distance.pdist(chapter_top_2,'euclidean') 
#进行层次聚类:
Z = hierarchy.linkage(disMat,method='average') 
#将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
fig = plt.figure(figsize=(10,5))
P=hierarchy.dendrogram(Z)
# plt.savefig('plot_dendrogram.png')
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230171035926.png)


