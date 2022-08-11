# LabelSemantics
> Code for Label Semantics for Few Shot Named Entity Recognition

经过修改，现在的方案已与原paper中思路一样，采用两个独立Bert Encoder分别对token和label进行编码


## 代码结构如下
```
├── fewshot.json
├── LabelSemantics_fewshot.py
    ├── finetune in target dataset
├── LabelSemantics.py
    ├── pretrain in source dataset
├── pretrain.json
```



## 实验结果如下

> 我们再自己的数据集上做了实验，基本效果如下。

|  K-Shots | 5-shot | 20-shot | 50-shot | 100-shot | 200-shot |  500-shot| Full dataset |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |  :----:  | 
| F1  | 0.53 |0.61 | 0.64 | 0.67 | 0.71 |0.76 | 0.89 |
| epoch  | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
