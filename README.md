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


## 二、 根据原始数据生成必要文件：

`cd code`

`python preprocess.py`

## 三、 参考indicator中的语句来获取indicators：
`cd ..`

`python indicator.py`
