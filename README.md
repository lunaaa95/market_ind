# market_ind
## 一、 建立data文件夹，文件夹结构：
```
+-- data
｜  +-- raw_data
    |   +--stock_code.npy
    |   +--stock_date.npy
    |   +--stock_concept.npy
    |   +--stock_style.npy
    |   +--stock_observation.npy
    |   +--stock_concept_label.csv
    |   +--style_list.pkl
    +--market
```

其中，style_list记录了style文件从第0列到第-1列按顺序的column名（在读数据库时都可以直接获取）


## 三、 参考indicator中的语句来获取indicators：
`cd ..`

`python indicator.py`
如果文件中 mode == "train", 则会进行deepwalk、LSTM等模型重新训练；
如果文件中 mode == "update", 则不进行deepwalk，采用训好的模型进行测试；
update时，请按文件中样本形式输入当日的新数据，无需再次输入历史数据。
