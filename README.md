# newscnn
## 数据集来自NLPCC2017新闻标题分类
## 运行步骤：
首先 
> python cut_txt.py
这一步是分词并保存分词结果
然后 
> python train_w2v.py
这一步是训练一个word2vec模型，仅使用本项目中提供的语料，如果有另外的语料效果会更好
最后 
> train.py
训练一个分类模型
## 准确率能到74%
