读入数据后，依次进行以下的数据处理步骤：

阈值为1.0的tf-idf降维 -> 拆成单个label -> 将label转换成最多100层前的祖先节点
 
前100万行经过如上处理后的数据结果如下：

DataDesc(sample_size=1000000, feature_dimension=957093, class_number=739308)
DataDesc(sample_size=3710074, feature_dimension=957093, class_number=266262)
DataDesc(sample_size=3710074, feature_dimension=957093, class_number=8489)

