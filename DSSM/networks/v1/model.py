import tensorflow as tf
from tensorflow import keras
from utils import currentTime,raiseModelPath
from networks.subgroups import EqualLinear,DNNLayer,FMLayer,wordEmbeddingLayer,createLayerModel,modelInputs,saveModel,loadModel

#模型输入列
def input_columns(inputs = None):
    inputs = [
        'user_id',
         'cargo_id',
         'searchscene',
         'actiontrucklengthmin',
         'actiontrucklengthmax',
         'actiontrucktype',
         'oftenrunroutes',
         'currentcity',
         'userlongitude',
         'userlatitude',
         'clickstartcity',
         'clickendcity',
         'clickweigthmax1',
         'clickweigthmax2',
         'clickweigthmax3',
         'clicktrucklength1',
         'clicktrucklength2',
         'clicktrucklength3',
         'clicktrucktype',
         'clickislcl',
         'clickhandlingtype',
         'clickissecuritytran',
         'clickcargocategory',
         'actionfirstcategory',
         'actioncargodescribe',
         'searchstartcity',
         'searchendcity',
         'searchtrucklengthmin',
         'searchtrucklengthmax',
         'searchtrucktype',
         'searchtruckweightmin',
         'searchtruckweightmax',
         'searchlcl',
         'searchnearrange',
         'cargolongitude',
         'cargolatitude',
         'cargostartcity',
         'cargoendcity',
         'cargotrucklength',
         'cargoweight',
         'cargoarrivelongitude',
         'cargoarrivelatitude',
         'cargotrucktype',
         'cargocategory',
         'cargoislcl',
         'cargohandlingtype',
         'cargoissecuritytran',
         'cargodrivedistance',
         'cargoloadtime',
         'cargodescribe',
         'isclick',
         'rnk']
    return inputs

#model construct
class userTower(keras.layers.Layer):
    def __init__(self,model_config):
        """
        用户的塔结构
        将货物的embedding与用户的embedding独立开来，为了方便模型部署
        -----------------------------------------------------------
        Params
        city_depth                     int   城市数,embedding深度
        city_embedding_length          int   城市向量长度
        user_numerical_features_num    int   用户数值向量个数(默认向量已做过归一化处理)
        numerical_features_length      int   数值向量经过FC层后长度
        truck_type_depth               int   货车类型深度
        truck_type_embedding_length    int   货车类型向量长度
        common_embedding_length        int   通用向量长度
        user_embedding_length          int   用户向量长度
        """
        super().__init__()
        self.city_depth=model_config.city_depth
        self.city_embedding_length=model_config.city_embedding_length
        self.user_numerical_features_num=model_config.input_config.user_numerical_features[0]
        self.numerical_features_length=model_config.numerical_features_length
        self.truck_type_depth=model_config.truck_type_depth
        self.truck_type_embedding_length=model_config.truck_type_embedding_length
        self.user_embedding_length=model_config.user_embedding_length
        self.common_embedding_length=model_config.common_embedding_length
        self.vocab_size = model_config.vocab_size
        self.encoder_count = model_config.encoder_count
        
        
    def build(self,input_shapes):
        self.linear_layer=EqualLinear(self.user_numerical_features_num,self.numerical_features_length)
        self.city_embedding_layer=keras.layers.Embedding(self.city_depth,self.city_embedding_length,mask_zero=True)
        self.truck_type_embedding_layer=keras.layers.Embedding(self.truck_type_depth,self.truck_type_embedding_length,mask_zero=True)
        self.lcl_embedding_layer=keras.layers.Embedding(3,self.common_embedding_length,mask_zero=True)
        self.handling_type_embedding_layer=keras.layers.Embedding(3,self.common_embedding_length,mask_zero=True)
        self.security_tran_embedding_layer=keras.layers.Embedding(3,self.common_embedding_length,mask_zero=True)
        self.search_range_embedding_layer=keras.layers.Embedding(5,self.common_embedding_length,mask_zero=True)
        self.cargo_category_embedding_layer=keras.layers.Embedding(50,self.common_embedding_length,mask_zero=True)
        self.search_scene_embedding_layer=keras.layers.Embedding(20,self.common_embedding_length)
        self.flatten_layer=keras.layers.Flatten()
        self.dnn_layer=DNNLayer(self.user_embedding_length)
        self.fm_layer=FMLayer(self.common_embedding_length)
        self.word_embedding_layer = wordEmbeddingLayer(self.vocab_size, self.common_embedding_length, self.encoder_count, attention_head_count = 1)
        super().build(input_shapes)
        
    def call(self,inputs):
        (user_search_scene, user_numerical_features, user_truck_type_labels,
            user_city_labels, user_is_lcl, user_handling_type, user_security_tran,
            user_cargo_category, user_cargo_describe, user_search_range)=inputs
        
        numerical_vector=self.linear_layer(user_numerical_features)                   #(batch,numerical_features_length)
        city_embedding=self.city_embedding_layer(user_city_labels)                    #(batch,19,city_embedding_length)
        truck_type_embedding=self.truck_type_embedding_layer(user_truck_type_labels)  #(batch,-1,truck_type_embedding_length)
        lcl_embedding=self.lcl_embedding_layer(user_is_lcl)                           #(batch,1,common_embedding_length)
        handling_type_embedding=self.handling_type_embedding_layer(user_handling_type)#(batch,1,common_embedding_length)
        security_tran_embedding=self.security_tran_embedding_layer(user_security_tran)#(batch,1,common_embedding_length)
        search_range_embedding=self.search_range_embedding_layer(user_search_range)   #(batch,1,common_embedding_length)
        search_scene_embedding=self.search_scene_embedding_layer(user_search_scene)   #(batch,1,common_embedding_length)
        cargo_category_embedding=self.cargo_category_embedding_layer(user_cargo_category)   #(batch,6,common_embedding_length)
        cargo_describe_embedding=self.word_embedding_layer(user_cargo_describe)       #(batch,3,common_embedding_length)
        
        city_embedding=self.flatten_layer(city_embedding)
        truck_type_embedding=tf.reduce_mean(truck_type_embedding,axis=1)
        lcl_embedding=self.flatten_layer(lcl_embedding)
        handling_type_embedding=self.flatten_layer(handling_type_embedding)
        security_tran_embedding=self.flatten_layer(security_tran_embedding)
        search_range_embedding=self.flatten_layer(search_range_embedding)
        search_scene_embedding=self.flatten_layer(search_scene_embedding)
        cargo_category_embedding=self.flatten_layer(cargo_category_embedding)
        cargo_describe_embedding=self.flatten_layer(cargo_describe_embedding)
        
        
        out=tf.concat([numerical_vector,city_embedding,truck_type_embedding,lcl_embedding,handling_type_embedding,security_tran_embedding,
                       search_range_embedding,search_scene_embedding,cargo_category_embedding,cargo_describe_embedding],axis=1)
        fm_output=self.fm_layer(out)
        out=self.dnn_layer(out)
        out=tf.nn.l2_normalize(0.5*(out+fm_output),axis=-1)
        return out
    
    
class cargoTower(keras.layers.Layer):
    def __init__(self,model_config):
        """
        货物的塔结构
        将货物的embedding与用户的embedding独立开来，为了方便模型部署
        -----------------------------------------------------------
        Params
        city_depth                     int   城市数,embedding深度
        city_embedding_length          int   城市向量长度
        cargo_numerical_features_num   int   货源数值向量个数(默认向量已做过归一化处理)
        numerical_features_length      int   数值向量经过FC层后长度
        truck_type_depth               int   货车类型深度
        truck_type_embedding_length    int   货车类型向量长度
        common_embedding_length        int   通用向量长度
        cargo_embedding_length         int   货源向量长度(点积时应等于用户向量长度)  
        """
        super().__init__()
        self.cargo_numerical_features_num=model_config.input_config.cargo_numerical_features[0]
        self.numerical_features_length=model_config.numerical_features_length
        self.city_depth=model_config.city_depth
        self.city_embedding_length=model_config.city_embedding_length
        self.truck_type_depth=model_config.truck_type_depth
        self.truck_type_embedding_length=model_config.truck_type_embedding_length
        self.cargo_embedding_length=model_config.cargo_embedding_length
        self.common_embedding_length=model_config.common_embedding_length
        self.vocab_size = model_config.vocab_size
        self.encoder_count = model_config.encoder_count        
     
    def build(self,input_shapes):
        self.linear_layer=EqualLinear(self.cargo_numerical_features_num,self.numerical_features_length)
        self.truck_type_embedding_layer=keras.layers.Embedding(self.truck_type_depth,self.truck_type_embedding_length,mask_zero=True)
        self.city_embedding_layer=keras.layers.Embedding(self.city_depth,self.city_embedding_length,mask_zero=True)
        self.lcl_embedding_layer=keras.layers.Embedding(3,self.common_embedding_length)
        self.handling_type_embedding_layer=keras.layers.Embedding(3,self.common_embedding_length)
        self.security_tran_embedding_layer=keras.layers.Embedding(3,self.common_embedding_length)
        self.flatten_layer=keras.layers.Flatten()
        self.dnn_layer=DNNLayer(self.cargo_embedding_length)  
        self.fm_layer=FMLayer(self.common_embedding_length)
        self.category_embedding_layer=keras.layers.Embedding(50,self.common_embedding_length,mask_zero=True)
        self.word_embedding_layer = wordEmbeddingLayer(self.vocab_size, self.common_embedding_length, self.encoder_count, attention_head_count = 1)
        super().build(input_shapes)
        
    
    def call(self,inputs):
        cargo_numerical_features,cargo_city_labels,cargo_truck_type_labels,cargo_category_labels,cargo_is_lcl,cargo_handling_type,cargo_security_tran,cargo_describe=inputs
        
        numerical_vector=self.linear_layer(cargo_numerical_features)                   #(batch,numerical_features_length)
        city_embedding=self.city_embedding_layer(cargo_city_labels)                    #(batch,2,city_embedding_length)
        truck_type_embedding=self.truck_type_embedding_layer(cargo_truck_type_labels)  #(batch,-1,truck_type_embedding_length)
        lcl_embedding=self.lcl_embedding_layer(cargo_is_lcl)                           #(batch,1,common_embedding_length)
        handling_type_embedding=self.handling_type_embedding_layer(cargo_handling_type)#(batch,1,common_embedding_length)
        security_tran_embedding=self.security_tran_embedding_layer(cargo_security_tran)#(batch,1,common_embedding_length)
        category_embedding=self.category_embedding_layer(cargo_category_labels)
        describe_embedding=self.word_embedding_layer(cargo_describe)       #(batch,1,common_embedding_length)
        
        city_embedding=self.flatten_layer(city_embedding)
        truck_type_embedding=tf.reduce_mean(truck_type_embedding,axis=1)
        lcl_embedding=self.flatten_layer(lcl_embedding)
        handling_type_embedding=self.flatten_layer(handling_type_embedding)
        security_tran_embedding=self.flatten_layer(security_tran_embedding)   
        category_embedding=self.flatten_layer(category_embedding)  
        describe_embedding=self.flatten_layer(describe_embedding) 
        
        out=tf.concat([numerical_vector,city_embedding,truck_type_embedding,lcl_embedding,handling_type_embedding,security_tran_embedding,category_embedding,describe_embedding],axis=1)
        fm_output=self.fm_layer(out)
        out=self.dnn_layer(out)
        out=tf.nn.l2_normalize(0.5*(out+fm_output),axis=-1)
        return out
    


class DSSM(keras.layers.Layer):
    def __init__(self,model_config):
        super().__init__()
        self.model_config=model_config
     
    def build(self,input_shapes):
        self.user_embedding_layer=userTower(self.model_config)
        self.cargo_embedding_layer=cargoTower(self.model_config)
        super().build(input_shapes)
    
    def call(self,inputs):
        (
            user_search_scene, user_numerical_features, user_truck_type_labels,
            user_city_labels, user_is_lcl, user_handling_type, user_security_tran,user_cargo_category, user_cargo_describe, user_search_range,
            cargo_numerical_features,cargo_city_labels,cargo_truck_type_labels,cargo_category_labels,cargo_is_lcl,cargo_handling_type,cargo_security_tran,cargo_describe
        )=inputs
        
        user_embedding=self.user_embedding_layer(
            (user_search_scene, user_numerical_features, user_truck_type_labels,user_city_labels, 
             user_is_lcl, user_handling_type, user_security_tran,user_cargo_category, user_cargo_describe, user_search_range)
        )
        
        cargo_embedding=self.cargo_embedding_layer(
            (cargo_numerical_features,cargo_city_labels,cargo_truck_type_labels,cargo_category_labels,cargo_is_lcl,cargo_handling_type,cargo_security_tran,cargo_describe)
        )
        
        out=tf.reduce_sum(tf.multiply(user_embedding,cargo_embedding),axis=1)*tf.constant(100.0)
        out=tf.nn.sigmoid(out)
        return out
    
    def get_config(self):
        config=super().get_config().copy()
        config.update({
            "model_config":self.model_config
        })
        return config
    
class multiplyLayer(keras.layers.Layer):
    def __init__(self,model_config,user_tower,cargo_tower):
        """
        当重新加载DSSM模型时,user_tower与item_tower无法分开保存，
        因此需要分别读取user_tower与item_tower，然后组装为DSSM
        """
        super().__init__()
        self.model_config=model_config
        self.user_embedding_layer=user_tower
        self.cargo_embedding_layer=cargo_tower
     
    def build(self,input_shapes):
        super().build(input_shapes)
    
    def call(self,inputs):
        (
            user_search_scene, user_numerical_features, user_truck_type_labels,
            user_city_labels, user_is_lcl, user_handling_type, user_security_tran,user_cargo_category, user_cargo_describe, user_search_range,
            cargo_numerical_features,cargo_city_labels,cargo_truck_type_labels,cargo_category_labels,cargo_is_lcl,cargo_handling_type,cargo_security_tran,cargo_describe
        )=inputs
        
        user_embedding=self.user_embedding_layer(
            (user_search_scene, user_numerical_features, user_truck_type_labels,user_city_labels, 
             user_is_lcl, user_handling_type, user_security_tran,user_cargo_category, user_cargo_describe, user_search_range)
        )
        
        cargo_embedding=self.cargo_embedding_layer(
            (cargo_numerical_features,cargo_city_labels,cargo_truck_type_labels,cargo_category_labels,cargo_is_lcl,cargo_handling_type,cargo_security_tran,cargo_describe)
        )
        
        out=tf.reduce_sum(tf.multiply(user_embedding,cargo_embedding),axis=1)*tf.constant(100.0)
        out=tf.nn.sigmoid(out)
        return out
    
    def get_config(self):
        config=super().get_config().copy()
        config.update({
            "model_config":self.model_config
        })
        return config


def restoreModel(model_config,user_tower,cargo_tower):
    user_features,cargo_features,y=modelInputs(model_config)
    yp=multiplyLayer(model_config,user_tower,cargo_tower)((*user_features,*cargo_features))
    model=keras.models.Model(
        inputs=[*user_features,*cargo_features],
        outputs=yp
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=model_config.learning_rate), 
        loss=model_config.losses
    )
    return model


def initModel(model_config):
    model_path=raiseModelPath(model_config.save_path,"dssm")
    if model_path is None:
        user_features,cargo_features,y=modelInputs(model_config)
        yp=DSSM(model_config)((*user_features,*cargo_features))
        model=keras.models.Model(
            inputs=[*user_features,*cargo_features],
            outputs=yp
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=model_config.learning_rate), 
            loss=model_config.losses
        )
    else:
        print(f"load model {model_path}")
        user_tower=loadModel(model_config.save_path,"user_tower")
        item_tower=loadModel(model_config.save_path,"item_tower")
        model=restoreModel(model_config,user_tower,item_tower)
    return model

def saveModel(model,model_config):
    """
    将模型存储为3个部分:
    1.dssm （模型整体）
    2.user_tower （模型中将user特征embeddding化的部分）
    3.item_tower  (模型中将item特征embeddding化的部分)
    """
    timestamp=currentTime("%Y%m%d%H%M%S")
    model.save(f'{model_config.save_path}/dssm_{timestamp}')
    dssm_layer=model.layers[-1]
    user_features,cargo_features,y=modelInputs(model_config)
    user_tower=dssm_layer.user_embedding_layer
    item_tower=dssm_layer.cargo_embedding_layer
    user_embedding_model=createLayerModel(user_features,user_tower,model_config)
    item_embedding_model=createLayerModel(cargo_features,item_tower,model_config)
    user_embedding_model.save(f'{model_config.save_path}/user_tower_{timestamp}')
    item_embedding_model.save(f'{model_config.save_path}/item_tower_{timestamp}')
    return
