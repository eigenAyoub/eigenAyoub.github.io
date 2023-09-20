---
layout: default
title:
permalink: /blogs/python-mini-task
---



In this task, we are asked to clean, and process a dataset.


### *A quick (manual) inspection. A few remarques:*
    
* The first dummy columns, kind of looks obsolute?!
* The the other columns
* Then everything seems to be regular


### *Loading the dataset:*
As the default separator in Pandas’ `read_csv()` method is `sep=","`, we have to specify the correct separator, in this case `sep = ';'` does the job.

###  *Some quick EDA:* 

* Shape: `data.shape` outputs:  `2009 x 40`
* Head:  `data.head(20)` outputs:

![Head](/src/head_20.png)

Apparently, there are 3 `dummy_header` columns, the first got picked as the `data.columns` (default behavior of Pandas). We can fix this by adding `header=None` as an argument to `read_csv()`. Reading the csv is done as follows: 

`data = pd.read_csv("input.csv", sep = ';', header=None)` 



### *Create a new header, clean rows, and reset index.*

Code:

```
data = pd.read_csv("input.csv", sep = ';', header=None)`
data.columns = data.iloc[6,:]+'_'+data.iloc[7,:]
data = data.iloc[9:,].reset_index(drop = True)
```

Result: 

![Head](/src/head_clean.png) 

*P.S:* the focus here is to get stuff done quickly, without worrying much about efficiency, as the dataset is small. For bigger datasets, I would've probably thought of more efficient ways.



### *Some visual:*




