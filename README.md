# LabelSemantics
Code for Label Semantics for Few Shot Named Entity Recognition

参考paper ,`Our label representations are updated with every update while training.`对FewShot_NER进行了修改。
修改前后说明如下：

LabelSemantics_detach_label_representation.py    label representation只计算了一次，init model的时候直接detach，在训练过程中也没有更新。

LabelSemantics.py      训练过程中不断迭代label representation，能在一定程度上缓解fewshot 小数据集训练时过拟合。因为在训练过程中不断更新label representation,所以训练会变慢

## Pipelines

关于大家关心的，如何将source dataset 的标签知识迁移到target dataset的问题：

> paper里面说了，先在source dataset 上pre-finetuning，然后在target dataset finetune。
代码实现上基本一致，只需要在target finetune时加载上一阶段保存的模型就行了。具体见代码(LabelSemantics.py)里的说明
