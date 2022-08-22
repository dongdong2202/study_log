## 词袋模型 Bag-of-words
-----------
John likes to watch movies. Mary likes too

John also likes to watch football games.

以上两句可以构造一个词典，**{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10} **
那么第一句的向量表示为：[1,2,1,1,1,0,0,0,1,1]，其中的2表示likes在该句中出现了2次，依次类推。

词袋模型同样有一下缺点：

    词向量化后，词与词之间是有大小关系的，不一定词出现的越多，权重越大。
    词与词之间是没有顺序关系的。
## n-gram 保持词的顺序， n表示滑动窗口大小
--------------
John likes to watch movies. Mary likes too

John also likes to watch football games.

以上两句可以构造一个词典，{"John likes”: 1, "likes to”: 2, "to watch”: 3, "watch movies”: 4, "Mary likes”: 5, "likes too”: 6, "John also”: 7, "also likes”: 8, “watch football”: 9, "football games": 10}

那么第一句的向量表示为：[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]，其中第一个1表示John likes在该句中出现了1次，依次类推。

**缺点：**随着n的大小增加，词表会成指数型膨胀，会越来越大。

## 共现矩阵 用于LSA
------------------
把词的频次填写在矩阵中。
存在的问题：

    向量维数随着词典大小线性增长。
    存储整个词典的空间消耗非常大。
    一些模型如文本分类模型会面临稀疏性问题。
    模型会欠稳定，每新增一份语料进来，稳定性就会变化。
# word2Vec

CBOW： 用周围的词来预测中间的词，预测目标词 ，O（n）快，不太准
可以看出，skip-gram进行预测的次数是要多于cbow的：因为每个词在作为中心词时，都要使用周围词进行预测一次。这样相当于比cbow的方法多进行了K次（假设K为窗口大小），因此时间的复杂度为O(KV)，训练时间要比cbow要长。
n-gram 用周围词来预测中间词，复杂度O（kn），但会相对转确一些



上采样upsampling

上采样就是以数据量多的一方的样本数量为标准，把样本数量较少的类的样本数量生成和样本数量多的一方相同，称为上采样。

下采样subsampled

下采样，对于一个不均衡的数据，让目标值(如0和1分类)中的样本数据量相同，且以数据量少的一方的样本数量为准。获取数据时一般是从分类样本多的数据中随机抽与少数量样本等数量的样本。


重要性采样是，使用另外一种分布来逼近所求分布一种方法。 用矩形逼近圆的曲线一求解面积

拒绝法采样的原理非常简单，如果我们难以直接在一个空间A均匀采样，那么我们可以考虑在一个更大的空间B内均匀采样，然后判断采样点是否在空间A中，如果在，则接受该采样点；否则拒绝该采样点

变换采样（英语：inverse transform sampling），又称为逆采样（inversion sampling）、逆概率积分变换（inverse probability integral transform）、逆变换法（inverse transformation method）、斯米尔诺夫变换（Smirnov transform）、黄金法则（golden rule）等，是伪随机数采样的一种基本方法。在已知任意概率分布的累积分布函数时，可用于从该分布中生成随机样本。

# lr

）线性回归要求变量服从正态分布，logistic回归对变量分布没有要求。
2）线性回归要求因变量是连续性数值变量，而logistic回归要求因变量是分类型变量。
3）线性回归要求自变量和因变量呈线性关系，而logistic回归不要求自变量和因变量呈线性关系
4）logistic回归是分析因变量取某个值的概率与自变量的关系，而线性回归是直接分析因变量与自变量的关系
————————————————
##       弱平稳过程（weak sence stationary  process ）表示：

        （1）平均值\mu是不变的，不随着时间t的变化而变化

        （2）标准差\sigma是不变的，不随着时间t的变化而变化

        （3）没有季节性和周期性特点

        满足上面3种情况即具备弱平稳性
原文链接：https://blog.csdn.net/qq_36171491/article/details/124829257


查全率=查全了么， 在正样本中查到几个，查到的真实正样本占真实正样本多少
查准率=查到的正样本的准确情况如何， 查到的真实正样本占查到的正样本多少