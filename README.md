# LabelSemantics
Code for Label Semantics for Few Shot Named Entity Recognition

参考paper ,`Our label representations are updated with every update while training.`对FewShot_NER进行了修改。
修改前后说明如下：

LabelSemantics_detach_label_representation.py    label representation只计算了一次，init model的时候直接detach，在训练过程中也没有更新。

LabelSemantics.py      训练过程中不断迭代label representation，能在一定程度上缓解fewshot 小数据集训练时过拟合。因为在训练过程中不断更新label representation,所以训练会变慢

