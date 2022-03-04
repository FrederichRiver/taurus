# taurus

NLP应用项目

## Corpus预处理

文本预处理

## 语料库

toDo：构建语料库

## 分词工具

toDo：构建分词工具

## word embedding

toDo：开发SkipGram词向量工具

### SkipGram方法

损失函数：

$$
\begin{align}
\frac{\partial logP(w_{o}|w_{c})}{\partial v_{c}}&=u_{o}-\sum_{j\in v}{\frac{exp(u_{j}^{T}v_{c})}{\sum_{i \in v}{{exp(u_{i}^{T}v_{c})}}}u_{j}}\\
&=u_{o}-\sum_{j \in V}{P(w_{o}|w_{c})u_{j}}
\end{align}
$$

这个概率函数计算开销过大
