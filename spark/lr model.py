#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
start spark
"""
import numpy as np
import pandas as pd
import functools
get_ipython().run_line_magic('run', 'header.ipynb')
sparkInit=InitSpark()
spark=sparkInit.start("demo","1",True).sqlc


# In[2]:


def scaling(c,label=[],significance=0):
    """
    Col function:Normalization columns into 0 to 1
    """
    if isinstance(c,str):
        c=col(c)
    w=Window.partitionBy(label)
    pcnt=cume_dist().over(w.orderBy(c.asc()))
    limited_c=when(pcnt.between(significance,1-significance),c)
    max_c=max(limited_c).over(w)
    min_c=min(limited_c).over(w)
    scaled_c=(c-min_c)/(max_c-min_c)
    return when(scaled_c>1,1).when(scaled_c<0,0).otherwise(scaled_c)


# In[88]:


"""利用spark内置ml包建模"""
X=np.random.randn(10000,3)*100
W=np.random.randn(3,1)
bias=np.random.randn(10000,1)
Y=X.dot(W)+bias  #创建初始数据

data=(
    spark
    .createDataFrame(
        pd.DataFrame([[i for i in np.append(x,y)] for x,y in zip(X,Y)]
            ,columns=[f"x{i}" for i in range(len(X[-1]))]+["y"]))
    )  #转为spark表格

columns=data.columns
filter_data=(
    data
    .select(*[col(c).astype("float").alias(c) for c in columns]) #因为测试只使用了数值型变量，将变量格式统一为float
    .where(functools.reduce(lambda x,y:x+y,[col(c) for c in columns]).isNotNull()) #去除空值
    .select(*[scaling(c).alias(c) for c in columns if c!="y"],col("y")) #对自变量归一化
)   #对数据进行简单的处理

training_data,testing_data=(
    filter_data.randomSplit([0.7,0.3])
    )
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

features=[c for c in columns if c!="y"]
label="y"
# 构建pipeline
vectorAseembler_features=VectorAssembler(inputCols=features,outputCol='features')
lr=LinearRegression(maxIter=100,regParam=0.0,labelCol=label)
pipeline=Pipeline(stages=[vectorAseembler_features,lr])
# pipeline拟合
model = pipeline.fit(training_data)
# 预测
prediction=model.transform(testing_data)
# 模型评估
evaluation=RegressionEvaluator(labelCol=label,predictionCol='prediction',metricName='rmse')
r2=evaluation.evaluate(prediction,{evaluation.metricName:'r2'})
print(f"模型偏差度r2：{r2}")

