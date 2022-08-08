import tensorflow as tf
from tensorflow import keras


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    """上采样，下采样，padding，注意指定了x与kernel，可以理解为固定运算操作"""
    out = upfirdn2d_native(
        x, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )

    return out

def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    """upfirdn2d的实际操作函数"""
    _, channel, in_h, in_w = input.shape
    x=tf.reshape(input,shape=(-1, in_h, in_w, 1))

    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape

    out=  tf.reshape(x,shape=(-1, in_h, 1, in_w, 1, minor))
    out = tf.pad(out, [[0,0],[0,0],[0, up_y - 1],[ 0, 0],[0, up_x - 1],[0, 0]])
    out=tf.reshape(out,shape=(-1, in_h * up_y, in_w * up_x, minor))

    out = tf.pad(out,[[0,0],[max(pad_y0, 0), max(pad_y1, 0)],[max(pad_x0, 0), max(pad_x1, 0)],[0,0]])
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]
    out=keras.layers.Permute(( 3, 1, 2),input_shape=out.shape)(out)
    out=tf.reshape(out,shape=[-1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1,1])
    w =tf.reshape(keras.backend.reverse(kernel,axes=[0,1]),shape=( kernel_h, kernel_w,1,1))
    out = tf.nn.conv2d(out,w,strides=1,padding='VALID')
    out=keras.layers.Permute(( 3, 1, 2),input_shape=out.shape)(out)
    out=tf.reshape(out,shape=(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
        ))
    out=keras.layers.Permute((2, 3, 1),input_shape=out.shape)(out)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    out=tf.reshape(out,shape=(-1, channel, out_h, out_w))
    return out


class FusedLeakyReLU(keras.layers.Layer):
    """拼接维度的leakyRelU"""
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        self.channel=channel
        self.negative_slope = negative_slope
        self.scale = scale
        super().__init__()

        
    def build(self,input_shapes):
        self.bias= tf.Variable(tf.zeros(self.channel),trainable=True,dtype=tf.float32)
        super().build(input_shapes)
        

    def call(self, x):
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)
    
    

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