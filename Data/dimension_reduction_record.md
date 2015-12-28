目的是去掉tf-idf值低的特征。
用数据前100000行来测试，总共有197496个特征，但当过滤掉tf-idf值小于1.0的特征之后，只余下10001行，9834个特征。
看来是某些样本所包含的特征过于稀疏。

需要看降维后还有多少样本以及真实类别，否则会丢失太多信息。

降维前：
DataDesc(sample_size=100000, feature_size=197496, label_size=76595)

用tf-idf值为1降维后：
DataDesc(sample_size=10001, feature_size=9834, label_size=7841)

果然真实类别数目已经变成原来的10%. 
看来tf-idf值不能设得太高。

用tf-idf值为0.5降维后：
DataDesc(sample_size=59747, feature_size=49417, label_size=42132)
