---
layout: default
title:
permalink: /blogs/python-mini-task
---

I'll detail in this mini-blog the steps that I took to pre-precess/clean the data.

### *A quick (manual) inspection. A few remarks:*
    
* The first dummy columns, kind of looks useless?
* The the other columns up row 8/9 do not seem in sync with the rest of the columns.
* Then everything seems to be regular starting from row 8/9.


### *Loading the dataset:*
As the default separator in Pandas’ `read_csv()` method is `sep=","`, we have to specify the correct separator, in this case `sep = ';'` does the job.

###  *Some quick EDA:* 

* Shape: `data.shape` outputs:  `2009 x 40`
* Head:  `data.head(20)` outputs:

![Head](/src/head_20.png)

Apparently, there are 3 `dummy_header` columns, the first got picked as the `data.columns` (default behavior of Pandas). We can fix this by adding `header=None` as an argument to `read_csv()`. Reading the csv (for now) is done as follows : 

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

*P.S:* The focus here is to get stuff done quickly, without worrying much about efficiency, as the dataset is small. For bigger datasets, I would've probably thought of more efficient ways.

### *Some final touches*
Clearly, all columns are to be numerical. We have two problems here:
1. Columns types are object (we can easily check this using `data.dtypes`), we would rather want the true type to be reflected.
2. Each entry is a `str` type.
3. Decimal part is represented by a `,`, and `.` represents thousands separator.

We fix this with the following  code snipit:

```
data.to_csv("data2.csv", sep = ";")
data = pd.read_csv("data2.csv", sep=";" , decimal = ",", thousands='.').drop(["Unnamed: 0"], axis=1)
```
*Remark:* This can't be done from the first call, as the dataset is mixed.

And now, we get the following `data.head()`:

![Head](/src/head_clean2.png)

And all types are now either `float` or `int`:

![Head](/src/head_types.png)

*P.S*: This is absolutely not efficient, as I'm saving and loading the same data. I'll try later on to come up with a second method.

### *Some visual:*

Below we plot a few variables:

![Head](/) 

Obviously, the third variables has a lot of volatility, let's visulize it better using a moving average or resampling. 

![Head](/src/head_avg.png)

![Head](/src/head_resample.png)



### *Final touches:*

The notebook, has some other improvements that I didn't include here. 


