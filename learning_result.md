### 数据分析

- 真实分类多

```
basic_statistics(min_val=1, max_val=1344, median=20, mean_val=27.61187490936296)
```
共1475666个不同的y，最少的覆盖1个样本，最大的覆盖1344（0.1%）个样本，中位数是20，均值是27.6，呈现左偏，但不是太严重。

- 特征维度高

特征维度近20万。这些特征都是词条，可以设合理的tf-idf阈值来降维。

- 真实分类所包含叶结点的关系
据官方文档，真实分类只包含层次关系的叶结点。层次关系是图结构。某条分类中的若干个叶结点可能属于同一个或多个父结点。

---

### 数据处理细节、模型训练过程及效果

2016-01-03

1. 取前10000行作为样本
2. 对样本用tf-idf进行降维，阈值为1
3. 将y中的多label转成单label, 对应的x有冗余
4. 将单label沿层次有向图往上最多100层进行转换，目的是减少类别数目
5. 选取2个单label用来测试
6. 用liblinear进行训练，参数为'-s 0 -c 0.03'. 
7. 对选出来的2个单label做macro评估，结果如下为(0.4097222222222222, 0.010944131128044779)

2016-01-03

1. 取前10000行作为样本
2. 对样本用tf-idf进行降维，阈值为1
3. 将label串中的各个label沿层次有向图往上100层进行转换，输出仍是label串
4. 选取label 2193976用作预测目标，label串中包含2193976的为正样本，否则为负样本
5. 用liblinear进行训练，参数为'-s 0 -c 0.03'.
6. 在2000个测试样本中，label 2193976的macro precision / recall 分别是 (0.44549763033175355, 0.2789317507418398), confusion matrix为measure(true_pos=94.0, false_pos=117.0, true_neg=1546.0, false_neg=243.0)

2016-01-04

1. 取前10行作为样本
2. 对样本用tf-idf进行降维，阈值为1
3. 将y中的多label转成单label, 对应的x有冗余
4. 将x转为csr形式的稀疏矩阵（DataDesc(sample_size=70, feature_dimension=690, class_number=70)）
5. 用sklearn.MultinomialNB对稀疏形式的x和array_like的y进行训练
6. 总耗时12.003s, 在单一label下的测试集误判率为 49/56 （56 = 70 * 0.8）
结论：慢且不准

1. 取前80行作为样本
2. 对样本用tf-idf进行降维，阈值为1
3. 将x转为csr形式的稀疏矩阵（DataDesc(sample_size=100, feature_dimension=3421, class_number=100)）
5. 用sklearn.MultinomialNB对稀疏形式的x和array_like的y进行训练
6. 总耗时21.064 seconds, 在单一label下的测试集误判率为 0/80
结论：还不如不拆分单label 

抽样比例0.01    DataDesc(sample_size=23422, feature_dimension=5112, class_number=21412) 57.795 seconds  22727/23422
抽样比例0.1时，DataDesc(sample_size=237287, feature_dimension=409599, class_number=187152)
用阈值为4.0的tf-idf降维后，DataDesc(sample_size=237287, feature_dimension=222, class_number=187152)
程序崩溃。

2016-01-06
重构

2016-01-07
1. 抽样比例0.0005
2. 对样本用tf-idf进行降维，阈值为1
3. 按8：2拆分训练集和测试集
4. 将训练集的multi-label拆分成单label, 此时对应的x有重复（多个分类）
5. 用拆分单label的y和x进行训练
6. 用5中得到的模型对训练集和测试集进行预测，并将预测结果转回拆分前的样本
7. 评估。结果如下：
训练集上的单label准确率
accuracy: 0.0607353906763, 185 out of 3046.
训练集上的multi-label macro precision / recall 
(0.0012031962807839576, 0.0012259524707657489)
测试集上的单label准确率
accuracy: 0.0482846251588, 38 out of 787.
测试集上的multi-label macro precision / recall
(0.00019272024627618842, 0.001364256480218281)
主要的问题：对每条x只给一个预测结果（将mnb当成线性分类器来用，选决策函数值最高的类别作为预测结果）。
而且在类别不均衡时，mnb本身会倾向于多数类，所以预测出来几乎是训练集中出现频率最高的单label。

2016-01-08
类别不均衡的问题仍未解决。

---

### 一些可以尝试的新方法

- Multinomial Naive Bayes 
- 修改mnb预测机制，从返回决策函数最高的类别变为返回决策函数top k的类别；处理类别不均衡带来的影响（并不能）
- 降低类别c先验概率在分类决策函数中的权重