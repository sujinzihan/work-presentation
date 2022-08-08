from utils import EasyDict,loadText,raiseLogger, generate_input_config
from dataProcess import loadTrainData
import argparse
import tensorflow as tf
from tensorflow import keras
import os
from sys import path


parser = argparse.ArgumentParser()
parser.add_argument("--data_path",type=str,help="path of dataset",default="dataset/training_data.csv")
parser.add_argument("--model_version",type=str,help="model version",default="v2")
parser.add_argument("--embedding_length",type=int,help="length of output embedding",default=16)
parser.add_argument("--losses",type=str,help="type of losses",default="crossentropy")
args = parser.parse_args()


# model config
model_config=EasyDict(
    {
        "city_depth":500,
        "city_embedding_length":32,
        "numerical_features_length":16,
        "truck_type_depth":50,
        "truck_type_embedding_length":16,
        "common_embedding_length":16,
        "user_embedding_length":args.embedding_length,
        "cargo_embedding_length":args.embedding_length,
        "vocab_size":1000,
        "encoder_count":6,
        "learning_rate":0.001,
        "batch_size":1024,
        "max_loops":1000000,
        "data_path":args.data_path,
        "log_path":f"logs/train_log_{args.model_version}",
        "save_path":f"weights/weights_{args.model_version}/",
        "losses":keras.losses.binary_crossentropy if args.losses=="crossentropy" else keras.losses.mse
    }
)

#raise log
log=raiseLogger(model_config.log_path)


#load model
current_path=os.getcwd()
path.append(f"{current_path}/networks/{args.model_version}/")
from model import initModel,saveModel,input_columns


#load data
data_format=loadText("dataset/data_format")
inputs = input_columns()
if inputs is None:
    inputs = list(data_format.name.values)
input_config = generate_input_config(data_format, inputs)
model_config["input_config"] = EasyDict(input_config)
dataset=loadTrainData(model_config.data_path,data_format,batch_size=model_config.batch_size)
x_train,y_train=next(dataset)


#load & train
model=initModel(model_config)
for i in range(model_config.max_loops):
    model.fit(x_train,y_train,epochs=10,verbose=0)
    x_train,y_train=next(dataset)
    if i%1000==0:
        with tf.GradientTape() as gtape:
            yp=model(x_train)
            loss=model_config.losses(y_train,yp)
            grads=gtape.gradient(loss,model.trainable_variables)
        log.info(f"epoch {i}:",yp,"\n",loss,"\n",grads,"\n")
        saveModel(model,model_config)