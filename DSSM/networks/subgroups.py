import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
from utils import currentTime,raiseModelPath

#subgroups
def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.shape.ndims - bias.shape.ndims - 1)
    if input.shape.ndims == 3:
        return (
            tf.nn.leaky_relu(input+tf.reshape(bias,shape=(1, *rest_dim, bias.shape[0])),alpha=negative_slope)*scale
        )
    else:
        return (
            tf.nn.leaky_relu(input+tf.reshape(bias,shape=(1, bias.shape[0],*rest_dim)),alpha=negative_slope)*scale
        )
    
class EqualLinear(keras.layers.Layer):
    """根据输入size进行rescale后的线性变换"""
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0.0, lr_mul=1, activation=None
    ):
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.bias=bias
        self.bias_init=bias_init
        self.lr_mul=lr_mul
        self.activation=activation
        self.scale=(1/math.sqrt(in_dim))*lr_mul
        super().__init__()

    def build(self,input_shapes):
        self.weight=tf.Variable(tf.divide(tf.random.normal([self.out_dim, self.in_dim]),self.lr_mul),trainable=True)
        if self.bias:
            self.bias = tf.Variable(tf.fill(self.out_dim,float(self.bias_init)),trainable=True,dtype=tf.float32)
        else:
            self.bias = None
        super().build(input_shapes)

    def call(self,x):
        if self.activation:
            out=tf.matmul(x,tf.transpose(self.weight*self.scale))
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out=tf.matmul(x,tf.transpose(self.weight*self.scale))
            out=tf.nn.bias_add(out,self.bias * self.lr_mul)
        return out

class FMLayer(keras.layers.Layer):
    def __init__(self, k, w_reg=0.01, v_reg=0.01):
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=keras.regularizers.l2(self.v_reg))

    def call(self, inputs, **kwargs):
        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)

        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, self.k)
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)

        output = linear_part + inter_part
        return output

class DNNLayer(keras.layers.Layer):
    def __init__(
        self, out_dim
    ):
        super().__init__()
        self.out_dim=out_dim

    def build(self,input_shapes):
        self.dense_1=keras.layers.Dense(128,activation="relu")
        self.normal_1=keras.layers.BatchNormalization()
        self.dense_2=keras.layers.Dense(64,activation="relu")
        self.normal_2=keras.layers.BatchNormalization()
        self.dense_3=keras.layers.Dense(self.out_dim)
        super().build(input_shapes)
        
    def call(self,x):
        out=self.dense_1(x)
        out=self.normal_1(out)
        out=self.dense_2(out)
        out=self.normal_2(out)
        out=self.dense_3(out)
        return out

"""
CIN
""" 
class CINLayer(keras.layers.Layer):
    """
    CIN part
    """

    def __init__(self, cin_size=[16,16,16,16,16], l2_reg=0.0001):
        super().__init__()
        self.cin_size = cin_size
        self.l2_reg=l2_reg

    def build(self, input_shapes):
        self.embedding_nums = input_shapes[1]
        self.field_nums = list(self.cin_size)
        self.field_nums.insert(0, self.embedding_nums)
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_uniform',
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }
        super().build(input_shapes)

    def call(self, inputs, **kwargs):
        """
        inputs (batch,embedding_nums,embedding_length)
        """
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)
            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)
            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])
            result_3 = tf.transpose(result_2, perm=[1, 0, 2])
            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')
            result_5 = tf.transpose(result_4, perm=[0, 2, 1])
            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)
        result = tf.reduce_sum(result,  axis=-1)
        return result

"""
DCN
"""   
class crossLayer(keras.layers.Layer):
    def __init__(self, l2_reg=0.0001):
        super(crossLayer, self).__init__()
        self.l2_reg=l2_reg

    def build(self, input_shape):
        # filters
        self.W = self.add_weight(
                name='w',
                shape=(input_shape[-1],),
                initializer='random_uniform',
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            )
        self.b=self.add_weight(
                name='b',
                shape=(input_shape[-1],),
                initializer='random_uniform',
                regularizer=keras.regularizers.l2(self.l2_reg),
                trainable=True
            )

    def call(self, inputs, **kwargs):
        """
        inputs (batch,length)
        """
        cross=inputs*tf.tensordot(tf.reshape(inputs,[-1,1,inputs.shape[-1]]),self.W,axes=1)+inputs+self.b
        return cross

class DCNLayer(keras.layers.Layer):
    """
    DCN part
    """
    def __init__(self,layer_num=5):
        super().__init__()
        self.layer_num=layer_num
    
    def build(self, input_shape):
        self.cross_layers=[crossLayer() for i in range(self.layer_num)]
        
    def call(self,inputs):  
        """
        inputs (batch,length)
        """
        for cross_layer in self.cross_layers:
            inputs=cross_layer(inputs)
        return inputs    
    
def createLayerModel(input_features,layer,model_config):
    """
    layer   keras.layer  keras.model中的一层或连续多层(sequence)
    input_features kera.Input 对应layer的输入
    model_config easyDict  基础模型设置
    """
    out=layer(input_features)
    model=keras.models.Model(
        inputs=[*input_features],
        outputs=out
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=model_config.learning_rate), 
        loss="mse"
    )
    return model

"""
transformer
"""
class Transformer(keras.Model):
    def __init__(self,
                 inputs_vocab_size,
                 target_vocab_size,
                 encoder_count,
                 decoder_count,
                 attention_head_count,
                 d_model,
                 d_point_wise_ff,
                 dropout_prob):
        super(Transformer, self).__init__()

        # model hyper parameter variables
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.encoder_embedding_layer = Embeddinglayer(inputs_vocab_size, d_model)
        self.encoder_embedding_dropout = keras.layers.Dropout(dropout_prob)
        self.decoder_embedding_layer = Embeddinglayer(target_vocab_size, d_model)
        self.decoder_embedding_dropout = keras.layers.Dropout(dropout_prob)

        self.encoder_layers = [
            EncoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob
            ) for _ in range(encoder_count)
        ]

        self.decoder_layers = [
            DecoderLayer(
                attention_head_count,
                d_model,
                d_point_wise_ff,
                dropout_prob
            ) for _ in range(decoder_count)
        ]

        self.linear = keras.layers.Dense(target_vocab_size)

    def call(self,
             inputs,
             target,
             inputs_padding_mask,
             look_ahead_mask,
             target_padding_mask,
             training
             ):
        encoder_tensor = self.encoder_embedding_layer(inputs)
        encoder_tensor = self.encoder_embedding_dropout(encoder_tensor, training=training)

        for i in range(self.encoder_count):
            encoder_tensor, _ = self.encoder_layers[i](encoder_tensor, inputs_padding_mask, training=training)
        target = self.decoder_embedding_layer(target)
        decoder_tensor = self.decoder_embedding_dropout(target, training=training)
        for i in range(self.decoder_count):
            decoder_tensor, _, _ = self.decoder_layers[i](
                decoder_tensor,
                encoder_tensor,
                look_ahead_mask,
                target_padding_mask,
                training=training
            )
        return self.linear(decoder_tensor)


class EncoderLayer(keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(EncoderLayer, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.multi_head_attention = MultiHeadAttentionLayer(attention_head_count, d_model)
        self.dropout_1 = keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff,
            d_model
        )
        self.dropout_2 = keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        output, attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        output = self.dropout_1(output, training=training)
        output = self.layer_norm_1(tf.add(inputs, output))  # residual network
        output_temp = output

        output = self.position_wise_feed_forward_layer(output)
        output = self.dropout_2(output, training=training)
        output = self.layer_norm_2(tf.add(output_temp, output)) #correct

        return output, attention


class DecoderLayer(keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(DecoderLayer, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.masked_multi_head_attention = MultiHeadAttentionLayer(attention_head_count, d_model)
        self.dropout_1 = keras.layers.Dropout(dropout_prob)
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.encoder_decoder_attention = MultiHeadAttentionLayer(attention_head_count, d_model)
        self.dropout_2 = keras.layers.Dropout(dropout_prob)
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
            d_point_wise_ff,
            d_model
        )
        self.dropout_3 = keras.layers.Dropout(dropout_prob)
        self.layer_norm_3 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, decoder_inputs, encoder_output, look_ahead_mask, padding_mask, training):
        output, attention_1 = self.masked_multi_head_attention(
            decoder_inputs,
            decoder_inputs,
            decoder_inputs,
            look_ahead_mask
        )
        output = self.dropout_1(output, training=training)
        query = self.layer_norm_1(tf.add(decoder_inputs, output))  # residual network
        output, attention_2 = self.encoder_decoder_attention(
            query,
            encoder_output,
            encoder_output,
            padding_mask
        )
        output = self.dropout_2(output, training=training)
        encoder_decoder_attention_output = self.layer_norm_2(tf.add(output, query))

        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_3(output, training=training)
        output = self.layer_norm_3(tf.add(encoder_decoder_attention_output, output))  # residual network

        return output, attention_1, attention_2


class PositionWiseFeedForwardLayer(keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.w_1 = keras.layers.Dense(d_point_wise_ff)
        self.w_2 = keras.layers.Dense(d_model)

    def call(self, inputs):
        inputs = self.w_1(inputs)
        inputs = tf.nn.relu(inputs)
        return self.w_2(inputs)


class MultiHeadAttentionLayer(keras.layers.Layer):
    def __init__(self, attention_head_count, d_model):
        super(MultiHeadAttentionLayer, self).__init__()

        # model hyper parameter variables
        self.attention_head_count = attention_head_count
        self.d_model = d_model

        if d_model % attention_head_count != 0:
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero.d_model must be multiple of attention_head_count.".format(
                    d_model, attention_head_count
                )
            )

        self.d_h = d_model // attention_head_count

        self.w_query = keras.layers.Dense(d_model)
        self.w_key = keras.layers.Dense(d_model)
        self.w_value = keras.layers.Dense(d_model)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)

        self.ff = keras.layers.Dense(d_model)

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        output, attention = self.scaled_dot_product(query, key, value, mask)
        output = self.concat_head(output, batch_size)

        return self.ff(output), attention

    def split_head(self, tensor, batch_size):
        # inputs tensor: (batch_size, seq_len, d_model)
        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.attention_head_count, self.d_h)
                # tensor: (batch_size, seq_len_splited, attention_head_count, d_h)
            ),
            [0, 2, 1, 3]
            # tensor: (batch_size, attention_head_count, seq_len_splited, d_h)
        )

    def concat_head(self, tensor, batch_size):
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count * self.d_h)
        )


class ScaledDotProductAttention(keras.layers.Layer):
    def __init__(self, d_h):
        super(ScaledDotProductAttention, self).__init__()
        self.d_h = d_h

    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)

        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)

        return tf.matmul(attention_weight, value), attention_weight


class Embeddinglayer(keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        # model hyper parameter variables
        super(Embeddinglayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = keras.layers.Embedding(vocab_size, d_model)
        
    def call(self, sequences):
        output = self.embedding(sequences) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        output += self.positional_encoding(tf.shape(sequences)[1])
        return output   
    
    def positional_encoding(self, max_len):
        pos = tf.expand_dims(tf.range(max_len, dtype=tf.float32), axis=1)
        index = tf.expand_dims(tf.range(self.d_model, dtype=tf.float32), axis=0)

        pe = self.angle(pos, index)
        

        pe_s = tf.expand_dims(tf.math.sin(pe[:, 0::2]), axis=-1)
        pe_c = tf.expand_dims(tf.math.cos(pe[:, 1::2]), axis=-1)
        pe = tf.reshape(tf.concat([pe_s, pe_c], axis =-1), [tf.shape(pe)[0], -1])

        pe = tf.expand_dims(pe, axis=0)
        return tf.cast(pe, dtype=tf.float32)

    def angle(self, pos, index):
        return pos / tf.pow(10000.0, (index - index % 2) / float(self.d_model))    

class wordEmbeddingLayer(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_length, encoder_count, attention_head_count, dropout_prob=0.2):
        """
        vocab_size hash字典深度
        embedding_length 映射长度
        encoder_count encoder层数
        attention_head_count attention头数
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.encoder_count = encoder_count
        self.attention_head_count = attention_head_count
        self.dropout_prob = dropout_prob
        
    def build(self, input_shapes):
        """
        input_shaps (b, l)
        """
        self.hash_layer = keras.layers.Hashing(self.vocab_size)
        self.embedding_layer = Embeddinglayer(self.vocab_size, self.embedding_length)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(self.dropout_prob)
        self.encoder_layers = [
            EncoderLayer(
                self.attention_head_count,
                self.embedding_length,
                self.embedding_length,
                self.dropout_prob
            ) for _ in range(self.encoder_count)
        ]
        super().build(input_shapes)
        
    def call(self, x, training):
        input_shape = x.shape
        x = tf.reshape(x, [-1])
        x = tf.strings.unicode_decode(x, "UTF-8")
        x = self.hash_layer(x).to_tensor()
        x = tf.concat([tf.ones([tf.shape(x)[0], 1], dtype=x.dtype), x], axis=-1) #增加一个起始字符标记
        inputs_padding_mask = tf.reshape(tf.cast(tf.greater(x, 0), dtype=tf.float32), shape=[-1, 1, 1, tf.shape(x)[-1]])
        x = self.embedding_layer(x)
        encoder_tensor = self.encoder_embedding_dropout(x, training=training)
        for i in range(self.encoder_count):
            encoder_tensor, _ = self.encoder_layers[i](encoder_tensor, inputs_padding_mask, training=training)
        output = self.mask_reduce_mean(encoder_tensor, tf.squeeze(inputs_padding_mask))
        output = tf.reshape(output, [-1, *input_shape[1:], self.embedding_length])
        return output
    
    @staticmethod
    def mask_reduce_mean(input_tensor, mask):
        mask = tf.dtypes.cast(mask, tf.float32)
        mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
        mask_weight = tf.expand_dims(mask, axis=-1)
        output = tf.multiply(input_tensor, mask_weight)
        output = tf.reduce_sum(output, axis=1) / mask_sum
        return output
    
#input
def modelInputs(model_config):
    input_config = model_config["input_config"]
    user_features = [] 
    cargo_features = [] 
    labels = []
    for k,v in input_config.items():
        exec(f"{k} = keras.Input([{v[0]}],dtype=tf.{v[1].name},name='{k}')")
        if "user" in k:
            user_features.append(eval(k))
        elif "cargo" in k:
            cargo_features.append(eval(k))
        else:
            labels.append(eval(k))
    return tuple(user_features), tuple(cargo_features), labels[-1]

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


def loadModel(model_path,key=None):
    if key is not None:
        model_path=raiseModelPath(model_path,key)
    return keras.models.load_model(model_path)
