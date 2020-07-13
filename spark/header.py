#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
import pandas as pd
from pyspark import SparkConf, SparkContext, HiveContext
from pyspark.sql import SparkSession, SQLContext,HiveContext, DataFrame, Window
from pyspark.sql import functions
from pyspark.sql.functions import *
import pandas as pd
import numpy as np
import getpass
from pyspark.sql.types import ArrayType,IntegerType,LongType,StringType


# In[2]:


class InitSpark:
    '''
    Class for Spark initialization
    '''
    #初始化类
    def __init__(self):
        
        self.user_name = getpass.getuser()
        self.spark_master = "spark://spark-namenode1:7077"
    def start(self, appName, cores="2", crossJoin = False, local = True):
        '''
        Build a new Spark session
        :param appName: name of Spark Session
        :param cores: CPU Cores assign to current Spark Session
        :param crossJoin: Optionally enable cross join
        :param local: Optionally create a local spark session for temporary request
        '''
        if local:
            self.spark_master = "local[" + cores + "]"
        if crossJoin:
            crossJoinSetting = "True"
        else:
            crossJoinSetting = "False"
        self.conf = SparkConf()
        
        (self.conf.setMaster(self.spark_master)
            .setAppName(appName + "_" +  self.user_name)
            .set("spark.sql.crossJoin.enabled", crossJoin))
        self.sc = SparkContext(conf=self.conf)
        self.sqlc = SQLContext(self.sc)
        return self
        
    def stop(self):
        '''
        Stop current Spark session
        '''
        self.session.stop()
        return self

def visualize(data_frame,N=5):
    """
    Transform spark dataframe to pandas dataframe then display 
    """
    df_1=(data_frame.limit(N))
    pd.options.display.max_rows=None
    pd.options.display.max_columns=None   
    return pd.DataFrame(df_1.collect(),columns=df_1.columns)

