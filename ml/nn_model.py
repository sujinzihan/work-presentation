#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import  os
from utils.utils import *


# # data_process

#以下为文本处理
def setFreqWords(freq_words_path=None):
    """
    set freq words for splitWords function
    """
    if freq_words_path:
        try:
            freq_words=loadText(freq_words_path)
            for word in freq_words:
                jieba.suggest_freq(word, tune=True)
        except Exception as e:
            print("unable to load freq words cause ",e)
    return

def splitWords(x,remove_zzts=True):
    """
    split words and filter word which is not in chinese,letters and digtial
    """
    if remove_zzts:
        cop=re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]") # 匹配不是中文、大小写、数字的其他字符
    if x and type(x)==str:
        return [i for i in jieba.lcut(cop.sub("",x), HMM=True)]
    else:
        return []

def removeStopWords(words,stop_words=None):
    """
    remove stop words from word list
    """
    if stop_words:
        return [word for word in words if word not in stop_words]
    else:
        return words

def corpusReader(df,context_features,stop_words=None):
    """
    transform sentence into separate words
    """
    for context_feature in context_features:
        df[context_feature]=df[context_feature].map(lambda x:splitWords(x))
        if stop_words:
            df[context_feature]=df[context_feature].map(lambda x:removeStopWords(x,stop_words))
    return   




#以下为通用函数
def categoryExtract(dataset,feature):
    """
    extract all categories from feature columns in vertor or string
    """
    feature_category=pd.DataFrame(dataset[feature].value_counts()).index.to_list()
    assert len(feature_category)>0
    for i in feature_category:
        assert type(i)==type(feature_category[-1])
    return set(reduce(lambda x,y:x+y,[i if type(i) in [list] else [str(i)] for i in feature_category])) 

def categoryEncode(categories):
    """
    encode category with unique 
    """
    initial_code=1 #
    category_dict={}
    for i in categories:
        category_dict[i]=initial_code
        initial_code+=1
    return category_dict

def fillNaFeatures(dataset,features,value=0):
    """
    fill null features with assign value
    """
    for feature in features:
        dataset[feature]=dataset[feature].fillna(value)
    return 

def revalueOutlier(x,mean,stddev,alpha=0.95):
    """
    replace null and outlier with mean
    """
    if abs(float(x)-mean)<=1.96*stddev:
        return float(x)
    else:
        return float(mean)

def rescale(dataset,features,rescale=True):
    """
    rescale scaled value and set outliars/null values to mean if necessary
    """
    for feature in features:
        if rescale:
            mean=dataset[feature].mean()
            stddev=dataset[feature].std()
            dataset[feature]=dataset[feature].apply(lambda x:revalueOutlier(x,mean,stddev))
    return

def replaceCatgoryVectory(x,encodedCatgory,append_header=True):
    """
    replace category label with boolean vector
    """
    if append_header:
        pos=[0]*(len(encodedCatgory)+1)
    else:
        pos=[0]*len(encodedCatgory)
    if x is None:
        pos[0]=1
    else:
        if isinstance(x,(str,int,float)):
            if str(x) in encodedCatgory.keys():
                if append_header:
                    pos[encodedCatgory[str(x)]]=1
                else:
                    pos[encodedCatgory[str(x)]-1]=1
            else:
                pos[0]=1
        else:
            for i in x:
                if i in encodedCatgory.keys():
                    if append_header:
                        pos[encodedCatgory[i]]=1
                    else:
                        pos[encodedCatgory[i]-1]=1
                else:
                    pos[0]=1
    return pos


#以下为特殊函数，只针对特定数据结构
def extractGodCatIds(x):
    """
    extract god_cat_ids
    """
    if not x:
        return ["-1"]
    if x!=0:
        return [i.split("=")[0] for i in x.split(",")]
    else:
        return ["-1"]
    

def encodeLabels(x):
    if x==-1:
        return [1,0]
    else:
        return [0,1]


# data loading



import jieba
from functools import reduce
import datetime

def processData(dataset=None,dataset_table="dwd_bixin_content_interact_features",
               stop_words_path="data/chineseStopWords.txt",
               user_freq_words_path="data/userFreqWords.txt",
               feature_json_path="data/featureJson",
               category_feature_len_path="data/category_feature_len_path",
               feature_description_path="data/feature_description",
               is_traning=True,odps=None):
    if dataset is None:
        ts_start = time.time()
        print("preparing dataset...")
        # dataset=loadOdpsData(dataset_table,o)
        dataset=runOdpsSql(f"select * from {dataset_table}'",odps)
        setFreqWords(user_freq_words_path)
        stop_words=loadText(stop_words_path)
        ts_end = time.time()
        print("load dataset used: {:.2f} s".format(ts_end - ts_start)) 
    ts_start = time.time()
    print("working on features...")
    #feature types
    category_columns=["content_compose","author_gender","author_is_god","gender","is_god"]
    category_vector_columns=["label_name","relation_typename","author_god_cat_ids","play_category"]
    sacled_columns=["comm_cnt","praise_cnt","reward_cnt","author_fans_counts","author_followers_counts","author_publish_dongtai_counts"]
    resacled_columns=["author_age","age","author_register_days","register_days"]
    content_columns=["content"]
    label_columns=["behavior_type"]
    
    #process features
    fillNaFeatures(dataset,category_vector_columns,"")
    dataset["author_god_cat_ids"]=dataset["author_god_cat_ids"].apply(lambda x:extractGodCatIds(x))
    dataset["label_name"]=dataset["label_name"].apply(lambda x:[i for i in x.split(",") if len(i)>0])
    dataset["relation_typename"]=dataset["relation_typename"].apply(lambda x:[i for i in x.split(",") if len(i)>0])
    dataset["play_category"]=dataset["play_category"].apply(lambda x:[i for i in x.split(",") if len(i)>0])
    fillNaFeatures(dataset,["author_gender","gender"],2)
    fillNaFeatures(dataset,category_columns,-1)
    fillNaFeatures(dataset,["age","author_age"],20)
    fillNaFeatures(dataset,sacled_columns,0)
    rescale(dataset,resacled_columns)
    rescale(dataset,["age"]) #年龄异常数据比较多，需要再重置一次
    if os.path.exists(feature_json_path):
        json_dict=loadJson(feature_json_path)
        category_feature_len_dict=loadJson(category_feature_len_path)
    else:
        json_dict={}
        category_feature_len={}
        for feature in category_vector_columns+category_columns:
            json_dict[feature]=categoryEncode(categoryExtract(dataset,feature))
            if feature in category_vector_columns:
                category_feature_len_dict[feature]=len(json_dict[feature]+1)
            else:
                category_feature_len_dict[feature]=len(json_dict[feature])
        json_dict["play_category"]=json_dict["author_god_cat_ids"]  #保持接单品类和玩家游戏品类的一致，方便模型
        category_feature_len_dict["play_category"]=category_feature_len_dict["author_god_cat_ids"] 
        saveJson(feature_json_path,json_dict)
        saveJson(category_feature_len_path,category_feature_len_dict)

    for feature in category_vector_columns:
        dataset[feature]=dataset[feature].apply(lambda x:replaceCatgoryVectory(x,json_dict[feature],append_header=True))
    for feature in category_columns:
        dataset[feature]=dataset[feature].apply(lambda x:replaceCatgoryVectory(x,json_dict[feature],append_header=False))

    #discribe feature
    if os.path.exists(feature_description_path):
        feature_description=loadJson(feature_description_path)
    else:
        feature_description={}
        feature_types=["category_vector","sacled","embedding"]
        feature_columns=[category_columns+category_vector_columns,sacled_columns+resacled_columns,content_columns]
        for feature_type,feature_column in zip(feature_types,feature_columns):
            feature_description[feature_type]=feature_column
        feature_description["label"]="behavior_type"
        saveJson(feature_description_path,feature_description)
        
    if is_traning:
        fillNaFeatures(dataset,label_columns,-1)
        for label_column in label_columns:
            dataset[label_column]=dataset[label_column].apply(lambda x:encodeLabels(x))

    ts_end = time.time()
    print("prepare feature used: {:.2f} s".format(ts_end - ts_start)) 
    del feature_description["embedding"]
    return dataset,feature_description,category_feature_len_dict


# # nn network

import tensorflow as tf

class nnNetwork(object):
    def __init__(self,dataset,features_description,vectory_feature_length,model_save_path=None,is_training=True):
        """
        initialization
        loading data:
        dataset       pandas table
        features_description    a dict which contains save path of features and type of features,load from json
        vectory_feature_length  a dict which contains length of vector variables
        """
        #initial tensorflow settings and deal with version problems
        tf.logging.set_verbosity(tf.logging.ERROR)
        #set defalt values
        self.embedding_length=16
        self.nn_hidden_length=32
        self.scaled_vevtors_length=64
        self.embedding_vevtors_length=64
        self.cross_hideen_length=1
        self.nn_hidden_length_layer_1_length=32
        self.output_layer_length=2
        self.learning_rate=0.0001
        self.model_save_path=model_save_path
        
        #trainning default settings
        self.batch_size=8192
        self.episodes=100
        
        #load pandas dataset
        if is_training:
            self.dataset,self.valid_dataset=self.__splitDataFrame(dataset)
        else:
            assert model_save_path is not None
            self.dataset,self.valid_dataset=dataset,None
            
        self.train_losses=[]
        self.vectory_feature_length=vectory_feature_length
        
        #initial values
        self.user_feature_num,self.user_feature_size = self.dataset.shape
        
        #column type:vector,category,boolen,scaled value
        self.category_vector_features=[]
        self.scaled_vector_features=[]
        self.scaled_features=[]
        self.embedding_features=[]
        for key in features_description.keys():
            if key=="category_vector":
                self.category_vector_features=features_description[key]
            elif key=="sacled":
                self.scaled_features=features_description[key]
            elif key=="embedding":
                self.embedding_features=features_description[key]
            elif key=="scaled_vector":
                self.scaled_vector_features=features_description[key]
            elif key=="label":
                self.label_column_title=features_description[key]
            else:
                print(f"unable to classify {key} data,please check typo errors")

        #pretreatment for different feature types
        self.__data_process(self.dataset)
        if self.valid_dataset is not None:
            self.__data_process(self.valid_dataset)
        
        
        #batch settings
        self.batch_index = None
        self.valid_batch_index = None
        
        #reset default graph before stack
        tf.reset_default_graph()
        
        #set model initializers
        self.tf_category_vectors,self.tf_scaled_vectors,self.tf_scaled_vector,self.tf_embedding_vectors,self.is_training,self.y=self.__model_init()
        
        #build blocks
        self.result, self.loss, self.train_op = self.__model_builder()
        
        #start session    
        self.gpuConfig = self.__gpuConfig()
        self.sess = tf.Session(config=self.gpuConfig)
        
        #save model if necessary
        self.saver = tf.train.Saver()
        
        #load model if exists
        if model_save_path:
            self.saver.restore(self.sess,tf.train.latest_checkpoint(model_save_path))
            
    def __data_process(self,dataset=None):
        """
        process dataset,should be applied on any input dataset
        """
        if dataset is None:
            dataset=self.dataset
        self.__rescale(dataset,self.scaled_features,self.__zScore)
        return 
        
    def __model_init(self):
        """
        create placeholder
        """
        #init catgory vector features
        if self.category_vector_features:
            tf_category_vectors={}
            for feature in self.category_vector_features:
                tf_category_vectors[feature]=tf.placeholder(tf.float32, shape=(None, self.vectory_feature_length[feature]), name=feature)
        else:
            tf_category_vectors=None
            
        #init scaled vector features    
        if self.scaled_vector_features:
            tf_scaled_vectors={}
            for feature in self.scaled_vector_features:
                tf_scaled_vectors[feature]=tf.placeholder(tf.float32, shape=(None, self.vectory_feature_length[feature]), name=feature) 
        else:
            tf_scaled_vectors=None
            
        #init scaled features       
        if self.scaled_features:
            tf_scaled_vector=tf.placeholder(tf.float32, shape=(None, len(self.scaled_features)), name="scaled_vector")
        else:
            self.scaled_features=None
            
        #init embedding 
        if self.embedding_features:
            tf_embedding_vectors={}
            for feature in self.embedding_features:
                tf_embedding_vectors[feature]=tf.placeholder(tf.float32, shape=(None, self.vectory_feature_length[feature]), name=feature)
        else:
            tf_embedding_vectors=None
            
        #init y and others
        y = tf.placeholder(tf.float32, shape=(None, 2))
        is_training = tf.placeholder(tf.bool, name="training")
        return tf_category_vectors,tf_scaled_vectors,tf_scaled_vector,tf_embedding_vectors,is_training,y
    
    def __model_builder(self):
        """
        building blocks
        """
        print("initialing model...")
        ts_start = time.time()
        
        #build catgory vector part
        catgory_vector_len=0
        cross_net=None
        if self.category_vector_features:
            #creat embedding vector for each category vector
            category_nets={}
            embeddings={}
            for feature in self.category_vector_features:
                catgory_vector_len+=self.embedding_length
                with tf.variable_scope(feature, reuse=tf.AUTO_REUSE) as scope:
                    embeddings[feature]=self.__raise_embeddings(shape=(self.vectory_feature_length[feature], self.embedding_length), random=True, zero_mask=False)
                    category_nets[feature]=tf.matmul(self.tf_category_vectors[feature],embeddings[feature])

            #use cross layer to handle category features
            with tf.variable_scope("category_vector_scope", reuse=tf.AUTO_REUSE) as scope:
                catgory_net=tf.concat([category_nets[key] for key in category_nets.keys()],axis=1)
                cross_weight = tf.get_variable("cross_weight", shape=(catgory_vector_len,self.cross_hideen_length),
                                                initializer=tf.random_normal_initializer())
                cross_bias = tf.get_variable("cross_bias", shape=(catgory_vector_len,),
                                               initializer=tf.random_normal_initializer())       
                cross_net = self.__raise_cross_layer(catgory_net, catgory_net, cross_weight, bias=cross_bias, activate=True)
                
        #build scaled vectores
        scaled_vector_len=0
        scaled_vec_net=None
        if self.scaled_vector_features:
            for feature in self.scaled_vector_features:
                scaled_vector_len+=self.vectory_feature_length[feature]
            with tf.variable_scope("scaled_vector_scope", reuse=tf.AUTO_REUSE) as scope:
                scaled_vec_nets=tf.concat([self.tf_scaled_vectors[key] for key in self.scaled_vector_features],axis=1)
                scaled_vec_weight=tf.get_variable("scaled_vec_weight", shape=(scaled_vector_len,self.scaled_vevtors_length),
                                            initializer=tf.random_normal_initializer())
                scaled_vec_bias = tf.get_variable("scaled_vec_bias", shape=(self.scaled_vevtors_length,),
                                               initializer=tf.random_normal_initializer())
                scaled_vec_net=tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(scaled_vec_nets, scaled_vec_weight) + scaled_vec_bias,
                                                                    training=self.is_training))
            scaled_vector_len=self.scaled_vevtors_length
    
        #build scaled values
        scaled_value_len=0
        scaled_net=None
        if self.scaled_features:
            with tf.variable_scope("scaled_values", reuse=tf.AUTO_REUSE) as scope:
                dense_weight = tf.get_variable("dense_weight", shape=(len(self.scaled_features),self.nn_hidden_length), dtype=tf.float32,
                                           initializer=tf.random_normal_initializer())
                dense_bias = tf.get_variable("dense_bias", shape=(1,self.nn_hidden_length), dtype=tf.float32,
                                         initializer=tf.random_normal_initializer())
                scaled_net=tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.tf_scaled_vector, dense_weight) + dense_bias,
                                                                        training=self.is_training))
            scaled_values_len=self.nn_hidden_length
          
        #build (word) embedding vectors
        embedding_len=0
        embedding_vec_net=None
        if self.embedding_features:
            for feature in self.embedding_features:
                embedding_vector_len+=self.vectory_feature_length[feature]
            with tf.variable_scope("embedding_vector_scope", reuse=tf.AUTO_REUSE) as scope:
                embedding_vec_nets=tf.concat([self.tf_embedding_vectors[key] for key in self.embedding_features],axis=1)
                embedding_vec_weight=tf.get_variable("embedding_vec_weight", shape=(embedding_vector_len,self.embedding_vevtors_length),
                                            initializer=tf.random_normal_initializer())
                embedding_vec_bias = tf.get_variable("embedding_vec_bias", shape=(self.embedding_vevtors_length,),
                                               initializer=tf.random_normal_initializer())
                embedding_vec_net=tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(embedding_vec_nets, embedding_vec_weight) + embedding_vec_bias,
                                                                    training=self.is_training))
            embedding_len=self.embedding_vevtors_length            
        
        #add layes
        with tf.variable_scope("dense_scope", reuse=tf.AUTO_REUSE) as scope:
            dense_length=catgory_vector_len+scaled_values_len+scaled_vector_len+embedding_len
            
            dense_weight_1 = tf.get_variable("dense_weight_1", shape=(dense_length,self.nn_hidden_length_layer_1_length),
                                             dtype=tf.float32, initializer=tf.random_normal_initializer())
            dense_weight_2 = tf.get_variable("dense_weight_2", shape=(self.nn_hidden_length_layer_1_length,self.nn_hidden_length_layer_1_length),
                                             dtype=tf.float32, initializer=tf.random_normal_initializer())
            
            dense_weight_3 = tf.get_variable("dense_weight_3", shape=(self.nn_hidden_length_layer_1_length,self.output_layer_length),
                                 dtype=tf.float32, initializer=tf.random_normal_initializer())
            dense_bias_1 = tf.get_variable("dense_bias_1", shape=(1,self.nn_hidden_length_layer_1_length), dtype=tf.float32,
                                           initializer=tf.random_normal_initializer())
            dense_bias_2 = tf.get_variable("dense_bias_2", shape=(1,self.nn_hidden_length_layer_1_length), dtype=tf.float32,
                                           initializer=tf.random_normal_initializer())
            
            dense_net = tf.concat([i for i in [cross_net,scaled_vec_net,scaled_net,embedding_vec_net] if i is not None], axis=1)  

            dense_net = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(dense_net, dense_weight_1) + dense_bias_1,
                                                                    training=self.is_training))
            dense_net = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(dense_net, dense_weight_2) + dense_bias_2,
                                                                    training=self.is_training))
            logits = tf.matmul(dense_net, dense_weight_3)
            result = tf.nn.softmax(logits, name="softmax_result")

        with tf.variable_scope("optimize_stuff", reuse=tf.AUTO_REUSE):
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=logits))  # + l2_loss
            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        
        ts_end = time.time()
        print("Model has been established in {:.2}s!".format(ts_end - ts_start))
        return result, loss, train_op
        
            
    
    def train(self ,batch_size=None, episodes=None,learning_rate=None,model_save_path=None, reset=True, reset_batches=False, valid=True, weighted=True):
        """
        trainning model
        """
        if not batch_size:
            batch_size=self.batch_size
            
        if not episodes:
            episodes=self.episodes
            
        if not learning_rate:
            learning_rate=self.learning_rate
        
        if not model_save_path:
            model_save_path=self.model_save_path
            
        if reset:
            self.sess.run(tf.global_variables_initializer())

        if reset_batches:
            self.batch_index = self.__raise_batches(batch_size=batch_size, total_num=self.user_feature_num,
                                                    weighted=weighted, shuffle=True)
            
            self.valid_batch_index = self.__raise_batches(batch_size=batch_size, total_num=self.valid_dataset.shape[0],
                                                          weighted=False, shuffle=False)

            print("Batch Features has Raised ..")

        if self.batch_index is None:
            self.batch_index = self.__raise_batches(batch_size=batch_size, total_num=self.user_feature_num,
                                                    weighted=weighted, shuffle=True)

        if self.valid_batch_index is None and valid:
            self.valid_batch_index = self.__raise_batches(batch_size=batch_size, total_num=self.valid_dataset.shape[0],
                                                          weighted=False, shuffle=False)

        print("Start Training...  ", datetime.datetime.now())
        ts_start = time.time()
        for epi in range(1, episodes + 1):
            print(f"running {epi} round epi")
            for i, batch in enumerate(self.batch_index):
#                 print("value_counts",self.dataset[self.label_column_title][batch].value_counts())
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=self.__raise_feed_dict(batch=batch))
                self.train_losses.append(loss)
    
            if epi % 10 == 0:
                print("Train Episode %d has loss: %.6f  " % (epi, np.mean(self.train_losses)), datetime.datetime.now())

            t_time = datetime.datetime.now()

            if (epi % 10 == 0) & (valid):
                valid_losses = []
                # valid_results = []
                for valid_batch in self.valid_batch_index:
                    valid_loss = self.sess.run(self.loss, feed_dict=self.__raise_feed_dict(dataset=self.valid_dataset,batch=valid_batch,is_training=False,is_validation=True))
                    valid_losses.append(valid_loss)
                    # valid_results.append(valid_result[:, 1].sum())

                # result_value = np.sum(valid_results)
                loss_value = np.mean(valid_losses)
                print("Ep %d loss: %s" % (epi, loss_value), t_time)

                if epi % 100 == 0:
                    prediction=self.sess.run(self.result, feed_dict=self.__raise_feed_dict(dataset=self.valid_dataset,is_training=False))
                    print("Ep %d loss: %s,current accurate rate is %s" % (epi, loss_value,self.__accurate_rate(pd.Series(np.argmax(prediction,axis=1)),\
                        pd.Series(np.argmax(np.stack(self.valid_dataset[self.label_column_title]),axis=1)))*100), t_time)
                    if model_save_path:
                        self.__model_saver(model_save_path,t_time)
        ts_end = time.time()
        print("traning accomplished,used {:.2}s".format(ts_end - ts_start))
        return

    def predict(self,dataset=None):
        """
        raise prediction of input data
        """
        if dataset is not None:
            self.__data_process(dataset)
        else:
            dataset=self.dataset
        predictions = self.sess.run(self.result, feed_dict=self.__raise_feed_dict(dataset,is_training=False))
        return predictions
    

    def saveModel(self,save_path=None):
        """
        save model to target path
        """
        if save_path is None:
            save_path=self.model_save_path
        self.__model_saver(save_path,datetime.datetime.now())
        return
    
    def __model_saver(self,model_path, t_time):
        """
        save model for further usage
        """
        now = t_time.strftime("%Y%m%d%H%M%S")
        path = f"{model_path}/ctr_model_{now}"
        self.saver.save(self.sess, path)
        print("model_%s has been saved !" % now)
        return  
    
    def __loadModel(self,save_path):
        """
        load model from target path
        """
        self.saver.restore(self.sess, save_path)
        return
    
    
    def __raise_word_vectorize(self):
        """
        transform word index into target word
        """
        return 

    def __raise_cross_layer(self, original_net, current_net, weight, bias=None, activate=False):
        """
        raise cross layer
        """
        net = tf.multiply(tf.matmul(original_net, weight), current_net)

        if bias is not None:
            # net = net + bias + tf.expand_dims(original_net, 1)
            net = net + bias + current_net
        else:
            # net = net + tf.expand_dims(original_net, 1)
            net = net + current_net

        if activate:
            net = tf.nn.relu(net)

        return net
    
        
    def __raise_embeddings(self, shape, random=True, zero_mask=False):
        """
        raise embedding matrixs
        """
        if (random) & (not zero_mask):
            embedding = tf.get_variable("embedding", shape=shape, dtype=tf.float32,
                                        initializer=tf.random_normal_initializer())
            
        if not random:
            constant_initialize = tf.constant_initializer(self.word_vectorize, dtype=tf.float32)
            embedding = tf.get_variable("embedding", shape=shape, dtype=tf.float32
                                        , initializer=constant_initialize)

        if (zero_mask) & (random):
            embedding = tf.get_variable("embedding", shape=(shape[0] - 1, shape[1]), dtype=tf.float32,
                                        initializer=tf.random_normal_initializer())
            zeros = tf.get_variable("embedding_mask", shape=(1, shape[1]), dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(), trainable=False)
            embedding = tf.concat((zeros, embedding), axis=0)
        return embedding
            
        
    def __embedding_lookup_sparse(self, ids, embedding, method="avg", expand_dims=False):
        """
        merge columns in index 
        """
        assert method in ['avg', 'sum', 'norm']

        embeddings = tf.gather(embedding, ids)
        embeddings_sum = tf.reduce_sum(embeddings, axis=1)

        if method == "avg":
            nonzero_count = tf.count_nonzero(ids, axis=1, keepdims=True, dtype=tf.float32)
            one_tensor = tf.ones_like(nonzero_count, dtype=tf.float32)
            base_count = tf.concat((one_tensor, nonzero_count), axis=1)
            max_base = tf.reduce_max(base_count, axis=1, keepdims=True)
            embedding_output = embeddings_sum / max_base

        if method == "sum":
            embedding_output = embeddings_sum

        if method == 'norm':
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings_sum), axis=1, keepdims=True))
            embedding_output = embeddings_sum / norm

        if expand_dims:
            embedding_output = tf.expand_dims(embedding_output, axis=1)

        return embedding_output
    
    def __rescale(self,dataset,features,method):
        """
        rescale scaled feature accordinf to selected method
        """
        for feature in features:
            mean=dataset[feature].mean()
            stddev=dataset[feature].std()
            max_val=dataset[feature].max()
            min_val=dataset[feature].min()
            dataset[feature]=dataset[feature].apply(lambda x:method(x,mean,stddev,min_val,max_val))
        return
    

    def __zScore(self,x,mean,stddev,min_val,max_val):
        """
        z scoce of scaled value
        """
        if stddev!=0:
            return (x-mean)/stddev
        else:
            return x-mean
    
    def __standardNonarmal(self,x,mean,stddev,min_val,max_val):
        """
        max->1 min->0 of scaled value
        """
        return (x-min_val)/(max_val-min_val)
    
    def __gpuConfig(self):
        """
        gpu config settings
        """
        gpuConfig = tf.ConfigProto(allow_soft_placement=True)
        gpuConfig.gpu_options.allow_growth = True
        # gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        return gpuConfig
    
    
    def __raise_batches(self, total_num, batch_size, weighted, shuffle=True):
        """
        raise batchs
        """
        sample_index = np.arange(total_num)

        if weighted:
            batch_index = self.__raise_sample(batch_size, sample_index,shuffle=shuffle)
            print("Random Batches has been raised !  ", datetime.datetime.now())
        else:
            if shuffle:
                np.random.shuffle(sample_index)
            batch_num = total_num // batch_size + 1 if total_num % batch_size > 0 else total_num // batch_size
            batch_index = [list(sample_index[x * batch_size: (x + 1) * batch_size]) for x in range(batch_num)]
            print("Valid Batches has been raised !", datetime.datetime.now())

        return batch_index

    def __raise_sample(self, batch_size, original_index,shuffle=True):
        """
        raise sample with even percent for each labels
        """
        label_num=len(self.dataset[self.label_column_title][0])
        even_weight=1/label_num
        label_indexs={}
        max_batch_num=0
        for label_pos in range(label_num):
            label_index = original_index[self.dataset[self.label_column_title].apply(lambda x:x[label_pos]) == 1]
            if shuffle:
                np.random.shuffle(label_index)
            label_batch_num = label_index.shape[0] // int(batch_size*even_weight)+(1 if label_index.shape[0] % int(batch_size*even_weight) > 0 else 0)
            if label_batch_num>max_batch_num:
                max_batch_num=label_batch_num
            label_indexs[label_pos]=[(label_index[x * int(batch_size *even_weight): (x + 1) * int(batch_size *even_weight)])
                                for x in range(label_batch_num)]
        batch_index= [np.concatenate([label_indexs[i][x%len(label_indexs[i])-1] for i in label_indexs.keys()],axis=0)
                      for x in np.arange(max_batch_num)]
        return batch_index
    
    def __raise_feed_dict(self,dataset=None,batch=None,is_training=True,is_validation=False):
        """
        generate feed_dict for model
        """
        if dataset is None:
            dataset=self.dataset
        if batch is None:
            batch=np.arange(dataset.shape[0])
        feed_dict={}
        if self.category_vector_features:
            for feature in self.category_vector_features:
                feed_dict[self.tf_category_vectors[feature]]=np.row_stack(dataset[feature][batch])
        if self.scaled_features:
            feed_dict[self.tf_scaled_vector]=np.array(dataset[self.scaled_features])[np.array(batch)]
        if self.scaled_vector_features:
            for feature in self.scaled_vector_features:
                feed_dict[self.tf_scaled_vectors[feature]]=np.row_stack(dataset[feature][batch])  
        if self.embedding_features:
            for feature in self.embedding_features:
                feed_dict[self.tf_embedding_vectors[feature]]=np.row_stack(dataset[feature][batch])
        if is_training or is_validation:
            feed_dict[self.y]=np.row_stack(dataset[self.label_column_title][batch])
        feed_dict[self.is_training]=is_training
        return feed_dict

    def __confusion_matrix(self,predictions,labels):
        """
        predictions pd seriers
        labels pd seriers
        """
        confusion_matrix=pd.concat([predictions,labels,pd.Series(predictions==labels,name="is_equal")],axis=1)
        confusion_matrix=confusion_matrix.groupby(confusion_matrix.columns.tolist(),as_index=False).size()
        return confusion_matrix
        
    def __accurate_rate(self,predictions,labels):
        """
        predictions pd seriers
        labels pd seriers
        """
        return (predictions==labels).sum()/labels.count()
    
    def __splitDataFrame(self,dataset,frac=0.7):
        """
        split dataset into train data and validation data
        """
        train_data=dataset.sample(frac=frac,random_state=None,axis=0)
        valid_data=dataset[~dataset.index.isin(train_data.index)]
        train_data=train_data.reset_index(drop=True)
        valid_data=valid_data.reset_index(drop=True)
        return train_data,valid_data
    
    def obtain_accurate_rate(self,dataset=None):
        """
        raise accurate rate
        """
        if dataset is None:
            dataset=self.valid_dataset
        else:
            self.__data_process(dataset)
        prediction=self.sess.run(self.result, feed_dict=self.__raise_feed_dict(dataset=dataset,is_training=False))     
        return self.__accurate_rate(pd.Series(np.argmax(prediction,axis=1)),pd.Series(np.argmax(np.stack(dataset[self.label_column_title]),axis=1)))