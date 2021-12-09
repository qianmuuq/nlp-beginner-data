# nlp-beginner-data

### 任务一：基于机器学习的文本分类

| Model     | Dev accuracy | Test accuracy |
| :---:     | :---:        | :---:         |
| Logistic regression | 51.75 | 48.50 |
| Softmax regression | 71.50 | 71.25 |

Softmax regression为自己写的，Logistics regression为sklearn库函数，有很大差距

### 任务二：基于深度学习的文本分类

| Model     | Dev accuracy | Test accuracy |
| :---:     | :---:        | :---:         |
| CNN | 65.95 | 65.64 |
| RNN | 67.37 | 67.00 |

结果基于Glove-50d

### 任务三：基于注意力机制的文本匹配

| Model     | Dev accuracy | Test accuracy |
| :---:     | :---:        | :---:         |
| Conditional Encoding | 59.31 | 56.54 |
| Attention | 59.24 | 56.67 |
| Word-by-word Attention | 59.07 | 55.99 |
| ESIM | 59.88 | 57.72 |

结果基于Glove-50d

### 任务四：基于LSTM+CRF的序列标注

| Model     | Dev F1 | Test F1 |
| :---:     | :---:        | :---:         |
| LSTM+CRF | 77.69 | 84.75 |
| LSTM+CRF R-Drop 0.8 | 78.27 | 85.94 |

F1值为字符级别匹配，实体级别待补充

### 任务五：基于神经网络的语言模型

bleu基本为0，看预测输出效果是否能有诗词基本规则。
