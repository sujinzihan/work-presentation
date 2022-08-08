import pandas as pd
import numpy as np
from typing import Any, List, Tuple, Union
import math
import time
import tensorflow as tf
from tensorflow.python import keras
import logging
import os

def raiseModelPath(path,model_keywords):
    """
    return the path of most recent pmml model
    """
    if not os.path.exists(path):
        os.mkdir(path)
    lists = [i for i in os.listdir(path) if model_keywords in i]
    if len(lists)==0:
        return None
    lists.sort(key= lambda f: os.path.getmtime(os.path.join(path,f)))
    file_path = os.path.join(path, lists[-1])
    return file_path

def raiseLogger(log_path):
    log=logging.getLogger()
    log.setLevel(logging.DEBUG)
    hander=logging.FileHandler(log_path,mode='w')
    formatter=logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    hander.setFormatter(formatter)
    log.addHandler(hander)
    return log

def currentTime(time_format='%Y-%m-%d %H:%M:%S'):
    """
    获取当前时间
    """
    return time.strftime(time_format,time.localtime(time.time()))

def loadText(filename):
    """读取指定数据文件的数据格式，返回pd.Series"""
    try:
        data=[]
        f=open(filename)
        line=f.readline()
        while line:
            col=[i.replace("\n","") for i in line.split(" ") if len(i)>0]
            data.append(col)
            line = f.readline()
        return pd.DataFrame(data,columns=["name","dtype","classes","length"])
    except:
        return None


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
        
def python_type_to_tf_type(input_type):
    if type(input_type) == str:
        return eval(f"tf.{input_type}")
    else:
        return eval(f"tf.{input_type.__name__}32")
    
def generate_input_config(data_format, inputs):
    input_config = {}
    for col_name, dtype, feature_name, length in data_format.values:
        if col_name in inputs and length != "id":
            if length != "Unlimited" and length is not None:
                input_config[feature_name] = [input_config.get(feature_name, [0, tf.string])[0] + int(length), python_type_to_tf_type(dtype)]
            else:
                input_config[feature_name] = [None, python_type_to_tf_type(dtype)]
    return input_config
