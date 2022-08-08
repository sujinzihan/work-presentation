import pandas as pd
import tensorflow as tf
import csv
import numpy as np


"""common csv to tfrecord data process"""
def csvLineCounter(path):
    """csv行数统计"""
    reader= csv.reader(open(path, "r"))
    lines=0
    for item in reader:
        lines+=1
    return lines

def stringSplit(x,sep):
    # split_strings = tf.strings.to_number(tf.strings.split(x, sep))  # 分割字符串
    split_strings = tf.strings.split(x, sep)
    return split_strings

def parseCsvLine(dtypes,delim,sep):
    def parseLine(line,dtypes=dtypes,delim=delim,sep=sep):
        """
        csv文件逐行读取
        n_fields 列数，必须添加。待优化为自动监测
        fields 为tensor，可以进行tf.nn.embedding_lookup等tensor操作，但tensor中的数据类型必须统一，推荐统一为float32
        """
        defs=[tf.constant("")] * len(dtypes) #为了简易处理，默认所有列为string格式
        fields = tf.io.decode_csv(line,record_defaults=defs,field_delim=delim)
        try:
            fields=list(map(lambda field:stringSplit(field,sep=sep),fields))
        except:
            print(fields)
            for field in fields:
                print(field)
                print(stringSplit(field,sep=sep))
        return fields
    return parseLine

def csvReaderDataset(filenames, batch_size=None, n_readers = 1, shuffle_buffer_size=1,count=None,dtypes=None,sep=",",delimiter=""):
    """
    csv转tf模型用dataset
    n_readers 每次apply lambda函数的元素数，即元素重复次数
    batch_size 每个batch的大小
    shuffle_buffer_size 指定buffer大小，数据会在指定buffer内随机排序，若shuffle_buffer_size为1相当于未重新排序
    """
    if isinstance(filenames,str):
        filenames=[filenames]
    dataset = tf.data.Dataset.list_files(filenames) 
    dataset = dataset.repeat(count=count)
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1), 
        cycle_length=n_readers 
    )
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)
    #pandas读取数据解析文件结构
    sample_data=pd.read_csv(filenames[-1],nrows =1,sep=delimiter)
    lines=sum([csvLineCounter(filename) for filename in filenames])
    if dtypes is None:
        dtypes=sample_data.dtypes
    dataset = dataset.map(parseCsvLine(dtypes=dtypes,delim=delimiter,sep=sep), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if batch_size is not None:
        return dataset.batch(batch_size),dtypes,lines
    return dataset,dtypes,lines

"""specific data case"""
def formatType(data,dtype):
    if "int" in str(dtype):
        return data.astype(float).astype(int)
    elif "float" in str(dtype):
        return data.astype(float)
    else:
        return np.apply_along_axis(lambda x:[i.decode() for i in x], 0, data)
        
def loadTrainData(filenames,data_format,batch_size=1000,sep=",",delimiter=""):
    dtypes=pd.Series(data=data_format.dtype.values,index=data_format.name.values)
    dataset,dtypes,lines=csvReaderDataset(filenames,batch_size=batch_size,dtypes=dtypes,sep=sep,delimiter=delimiter)
    dataset=dataset.as_numpy_iterator()
    while True:
        x_train={}
        y_train=None
        datas=dataset.next()
        for data,name,dtype,classes in zip(datas,data_format.name,data_format.dtype,data_format.classes):
            if classes not in ["is_click","y"]:
                if classes in x_train:
                    x_train[classes]=np.concatenate((x_train[classes],formatType(data,dtype)),axis=1)
                else:
                    x_train[classes]=formatType(data,dtype)
            else:
                y_train=formatType(data,dtype)
        # y_train=np.squeeze(y_train)
        yield (x_train,y_train)