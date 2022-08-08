import math
import random
import tensorflow as tf
from tensorflow import keras
from models.styleGAN2.op import upfirdn2d,FusedLeakyReLU

class PixelNorm(keras.layers.Layer):
    """初始特征向量z会先经过pixelnorm再流向mapping层转换成线性无关的中间特征向量
    沿着不同的轴就是不同的norm方式"""
    def __init__(self):
        super().__init__()

    def call(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(x ** 2, axis=1, keepdims=True) + 1e-8)

    

def make_kernel(k):
    """制作一个kernel """
    k=tf.Variable(k,trainable=False, dtype=tf.float32)
    if k.shape.ndims == 1:
        k = k[None, :] * k[:, None]
    k = k/tf.reduce_sum(k)
    return k



class Upsample(keras.layers.Layer):
    """上采样"""
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        
        self.kernel=kernel
        self.pad = (pad0, pad1)

    def call(self, x):
        out = upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out
    


class Downsample(keras.layers.Layer):
    """下采样"""
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2
        
        self.kernel=kernel
        self.pad = (pad0, pad1)

    def call(self, x):
        out = upfirdn2d(x, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


    
class Blur(keras.layers.Layer):
    """用指定kernel进行图片模糊化处理"""
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
            
        self.kernel=kernel
        self.pad = pad

    def call(self, x):
        out = upfirdn2d(x, self.kernel, pad=self.pad)
        return out

    
class EqualConv2d(keras.layers.Layer):
    """根据输入size进行rescale后的卷积"""
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.bias=bias
        self.kernel_size=kernel_size
        self.in_channel=in_channel
        self.out_channel=out_channel
        
        
    def build(self,input_shapes):
        self.weight=tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size,
            self.in_channel,self.out_channel]),trainable=True)
        if self.bias:
            self.bias = tf.Variable(tf.zeros(self.out_channel),trainable=True)
        else:
            self.bias = None
            
        super().build(input_shapes)

    def call(self,x):
        x=tf.transpose(x,perm=[0,2,3,1])
        x=keras.layers.convolutional.ZeroPadding2D(self.padding)(x)
        out=tf.nn.conv2d(
            x,
            self.weight * self.scale,
            strides=self.stride,
            padding="VALID"
        )
        if self.bias is not None:
            out=tf.nn.bias_add(out,self.bias)
            
        out=keras.layers.Permute((3, 1, 2),input_shape=out.shape)(out)
        return out
    
    
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
    
    
class ScaledLeakyReLU(keras.layers.Layer):
    """根据输入size进行rescale后的leakyRelU变换"""
    def __init__(self,negative_slope=0.2):
        self.negative_slope = negative_slope
        super().__init__()
        
    def build(self,input_shapes):
        self.leaky_relu=keras.layers.LeakyReLU(alpha=self.negative_slope)
        super().build(input_shapes)
        
    def call(self,x):
        out=self.leaky_relu(x)
        out=out*math.sqrt(2)
        return out
    
    
def depthwise_conv2d_transpose(
    value,
    filter,
    strides,
    output_shape,
    data_format='NHWC',
    padding='VALID',
    name='depthwise_conv2d_transpose'):
    """分层反卷积，对应tf.nn.deepwise_conv2d,待优化：已提议tensorflow团队开发"""
    if data_format == 'NHWC':
        channel_axis = 3
        channel_output_shape = [
          output_shape[0],
          output_shape[1],
          output_shape[2],
          1]
    elif data_format == 'NCHW':
        channel_axis = 1
        channel_output_shape = [
          output_shape[0],
          1,
          output_shape[2],
          output_shape[3]]
    else:
        raise ValueError("Unsupported data_format: {}".format(data_format))

    filter_shape = filter.get_shape().as_list()
    if len(filter_shape) != 4:
        raise ValueError("Expected filter of dim 4, got filter shape: {}".format(filter_shape))

    conv_in_channels = filter_shape[2]
    if conv_in_channels is None:
        raise ValueError("Number of convolution input channels in filter (dim 2) " +
                         "must be known at compile time, got: {}".format(conv_in_channels))

    channel_multiplier = filter_shape[3]
    if channel_multiplier is None:
        raise ValueError("Channel multiplier in filter (dim 3) " +
                         "must be known at compile time, got: {}".format(channel_multiplier))

    # collect outputs for all input channels
    c_outputs = []

    # create all operations in scope with name
    with tf.compat.v1.variable_scope(name):
        # regular conv2d_transpose for each input channel
        for c in range(conv_in_channels):
            # grab subset of value
            c_value = tf.gather(value,
                indices=list(range(
                  channel_multiplier*c,
                  channel_multiplier*c+channel_multiplier)),
                axis=channel_axis)

            # grab subset of filter
            c_filter = tf.gather(filter,
                indices=[c],
                axis=2)
            
            # run for this input channel
            c_outputs.append(
            tf.nn.conv2d_transpose(
              c_value,
              c_filter,
              strides=strides,
              output_shape=channel_output_shape,
              data_format=data_format,
              padding=padding,
              name='channel{}'.format(c)))
    
    # concatenate outputs within variable scope
    output = tf.concat(c_outputs, axis=channel_axis, name='concat')
    return output


class ModulatedConv2d(keras.layers.Layer):
    """
    依据输入的style为权重，对图片进行deepthwise 卷积/反卷积/padding的操作，
    style作为卷积kernel先进行数量级修正后与输入图像完成卷积/反卷积
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        ):
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        
        super().__init__()
        
    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def build(self,input_shapes):
        self.weight = tf.Variable(
            tf.random.normal([1, self.out_channel, self.in_channel, self.kernel_size, self.kernel_size]),
            trainable=True
        )
        super().build(input_shapes)
        
    def call(self,x,style, input_is_stylespace=False):
        """
        x (batch, in_channel, height, width)
        style (batch,style_dim)
        """
        batch, in_channel, height, width = x.shape
        
        if not input_is_stylespace:
            style=tf.reshape(self.modulation(style),[batch, 1, in_channel, 1, 1])
            
        weight = self.scale * self.weight * style
        
        if self.demodulate:
            demod = tf.math.rsqrt(tf.reduce_sum(tf.math.pow(weight,2),axis=[2,3,4]) + 1e-8)
            weight = weight * tf.reshape(demod,[batch, self.out_channel, 1, 1, 1])
            
        weight=tf.reshape(weight,[batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size])
        
        if self.upsample:
            x=tf.tile(x,[1,self.out_channel,1,1])
            x=tf.reshape(x,[1,batch *in_channel*self.out_channel, height, width])
            x=tf.transpose(x,perm=[0,2,3,1])

            weight=tf.reshape(weight,[self.out_channel*batch,in_channel ,self.kernel_size, self.kernel_size])
            weight=tf.transpose(weight,perm=[2,3,0,1])

            strides=2
            out=depthwise_conv2d_transpose(x,weight,strides=[1,strides,strides,1],
                    output_shape=[1,strides*height+1,strides*height+1,1])
            out=tf.transpose(out,perm=[0,3,1,2])
            _, _, height, width = out.shape
            out=tf.reshape(out,[batch, self.out_channel, height, width])
            out=self.blur(out)

        elif self.downsample:
            x=blur(x)
            _, _, height, width = x.shape
            
            
            x=tf.tile(x,[1,self.out_channel,1,1])
            x=tf.reshape(x,shape=[1,in_channel*batch *self.out_channel, height, width])
            x=tf.transpose(x,perm=[0,2,3,1])

            weight=tf.transpose(weight,perm=[2,3,0,1])
            weight=tf.reshape(weight,shape=[ self.kernel_size, self.kernel_size,batch*self.out_channel*in_channel,1])


            out=tf.nn.depthwise_conv2d(x,weight,strides=[1,2,2,1],padding='VALID')
            out=tf.transpose(out,perm=[0,3,1,2])
            _, _, height, width = out.shape
            out=tf.reshape(out,shape=[self.out_channel*batch,-1,in_channel,height,width])
            out=tf.reduce_sum(out,axis=2)
            out=tf.reshape(out,shape=[batch,out_channel,height,width])
            
        else:
            x=tf.reshape(x,[1, batch * in_channel, height, width])
            x=tf.pad(x,[[0,0],[0,0],[self.padding,self.padding],[self.padding,self.padding]])
            _,_,height,width=x.shape
            
            x=tf.tile(x,[1,self.out_channel,1,1])
            x=tf.reshape(x,shape=[ 1,in_channel*batch *self.out_channel, height, width])
            x=tf.transpose(x,perm=[0,2,3,1])

            weight=tf.transpose(weight,perm=[2,3,0,1])
            weight=tf.reshape(weight,shape=[ self.kernel_size, self.kernel_size,batch*self.out_channel*in_channel,1])

            out=tf.nn.depthwise_conv2d(x,weight,strides=[1,1,1,1],padding='VALID')
            out=tf.transpose(out,perm=[0,3,1,2])
            _, _,height,width= out.shape
            out=tf.reshape(out,shape=[self.out_channel*batch,-1,in_channel,height,width])
            out=tf.reduce_sum(out,axis=2)
            out=tf.reshape(out,shape=[batch,self.out_channel,height,width])
        return out, style
    
class NoiseInjection(keras.layers.Layer):
    """为输入图像增加噪音，噪音不可训练"""
    def __init__(self):
        super().__init__()
        
    def build(self,input_shapes):
        self.weight=tf.Variable(tf.zeros(1),trainable=True,dtype=tf.float32)
        super().build(input_shapes)
        
    def call(self,x,noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = tf.random.normal(shape=x.shape)
            return x + self.weight * noise

class ConstantInput(keras.layers.Layer):
    """制造一个与输入图像同形状的随机数，可训练"""
    def __init__(self,channel,size=4):
        self.channel=channel
        self.size=size
        super().__init__()
        
    def build(self,input_shapes):
        self.Input=tf.Variable(tf.random.normal(shape=[1, self.channel, self.size, self.size]),trainable=True,dtype=tf.float32)
        super().build(input_shapes)
    
    def call(self,x):
        batch=x.shape[0]
        _,channel,height,width=self.Input.shape
        out=tf.broadcast_to(self.Input,shape=[batch,channel,height,width])
        return out
    

class StyledConv(keras.layers.Layer):
    """
    应用style卷积
    style卷积-加噪音-激活层
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.kernel_size=kernel_size
        self.style_dim=style_dim
        self.upsample=upsample
        self.blur_kernel=blur_kernel
        self.demodulate=demodulate
        super().__init__()
    
    def build(self,input_shapes):
        self.conv = ModulatedConv2d(
            self.in_channel,
            self.out_channel,
            self.kernel_size,
            self.style_dim,
            upsample=self.upsample,
            blur_kernel=self.blur_kernel,
            demodulate=self.demodulate,
        )
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(self.out_channel)
        super().build(input_shapes)
    
    def call(self,x, style, noise=None, input_is_stylespace=False):
        out, style = self.conv(x, style, input_is_stylespace=input_is_stylespace)
        out=self.noise(out, noise=noise)
        out = self.activate(out)
        return out, style


class ToRGB(keras.layers.Layer):
    """3通道style卷积-加偏差-上采样"""
    def __init__(self, in_channel, style_dim, isUpsample=True, blur_kernel=[1, 3, 3, 1]):
        self.in_channel=in_channel
        self.style_dim=style_dim
        self.isUpsample=isUpsample
        self.blur_kernel=blur_kernel
        super().__init__()
        
    def build(self,input_shapes):
        if self.isUpsample:
            self.upsample = Upsample(self.blur_kernel)
        self.conv = ModulatedConv2d(self.in_channel, 3, 1, self.style_dim, demodulate=False)
    
        self.bias=tf.Variable(tf.zeros(shape=[1,3,1,1]),trainable=True,dtype=tf.float32)
        super().build(input_shapes)
        
    def call(self, x, style, skip=None, input_is_stylespace=False):
        out, style = self.conv(x, style, input_is_stylespace=input_is_stylespace)
        out=out+self.bias
        
        if skip is not None:
            skip=self.upsample(skip)
            
            out = out + skip
        return out, style

    
class Generator(keras.layers.Layer):
    """生成器"""
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        #获取变量
        self.size=size
        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1 
        self.n_latent = self.log_size * 2 - 2
        self.style_dim=style_dim #风格化参量长度
        self.n_mlp=n_mlp   #图层数
        self.channel_multiplier=channel_multiplier 
        self.blur_kernel=blur_kernel
        self.lr_mlp=lr_mlp

        self.channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * channel_multiplier,
                128: 128 * channel_multiplier,
                256: 64 * channel_multiplier,
                512: 32 * channel_multiplier,
                1024: 16 * channel_multiplier,
            }
        super().__init__()
    
    
    def build(self,input_shapes):
        #逐层创建
        layers=[PixelNorm()]

        for i in range(self.n_mlp):
            layers.append(
                EqualLinear(
                    self.style_dim, self.style_dim, lr_mul=self.lr_mlp, activation='fused_lrelu'
                )
            )
        
        self.style= keras.Sequential(layers)
        
        self.style_input=ConstantInput(self.channels[4])
        self.conv1=StyledConv(
            self.channels[4], self.channels[4], 3, self.style_dim, blur_kernel=self.blur_kernel
        )
        self.to_rgb1=ToRGB(self.channels[4], self.style_dim, isUpsample=False)
        
        self.convs = []
        self.upsamples = []
        self.to_rgbs = []
        self.noises = []
        
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            #noises 需要被记录在模型中，但不参与训练
            #trainable=False 是否有效需要在打出核查
            self.noises.append(tf.Variable(tf.random.normal(shape=shape),trainable=False,dtype=tf.float32)) 
            
        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    self.style_dim,
                    upsample=True,
                    blur_kernel=self.blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, self.style_dim, blur_kernel=self.blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))

            in_channel = out_channel

        super().build(input_shapes)
        
    def make_noise(self):
        noises = [tf.random.normal(shape=[1, 1, 2 ** 2, 2 ** 2])]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(tf.random.normal(shape=[1, 1, 2 ** i, 2 ** i]))
        return noises
    
    def mean_lantent(self):
        lantent_in=tf.random.normal(shape=[self.n_latent, self.style_dim])
        lantent=tf.reduce_mean(self.style(latent_in),axis=0,keepdims=True)
        return latent   
        
    def get_lantent(self,x):
        return self.style(x)
    
    
    def reshape_lantent(self,latent,repeats):
        assert latent.shape.ndims<3
        latent=tf.expand_dims(latent,axis=1)
        if latent.shape.ndims<3:
            latent=tf.expand_dims(latent,axis=0)
        latent=tf.tile(latent,[1,repeats,1])
        return latent
    
    
    def call(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        input_is_stylespace=False,
        noise=None,
        randomize_noise=True,
    ):
        """
        styles 风格输入，list of tensor
        """
        if not input_is_latent and not input_is_stylespace:
            styles=tf.map_fn(self.style,styles) #线性变化以后的style输入们
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = self.noises
        
        #使用truncation_latent与styles按照截断比例混合
        if truncation < 1 and not input_is_stylespace:
            style_t = []
            
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
                
            styles = style_t
        
        #这里把style[0]取出作为lantent(当取出的维度不足时进行了维度填充)
        if input_is_stylespace:
            latent = styles[0]
        elif len(styles) < 2:
            inject_index = self.n_latent
            
            if styles[0].shape.ndims < 3:
                latent=self.reshape_lantent(styles[0],inject_index)
            else:
                latent=styles[0]       
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            
            latent=self.reshape_lantent(styles[0],inject_index)
            latent2=self.reshape_lantent(styles[1],self.n_latent - inject_index)
            
            latent = tf.concat([latent, latent2],axis=1)
        #初始化循环
        style_vector = []
        if not input_is_stylespace:
            out = self.style_input(latent)
            out, out_style = self.conv1(out, latent[:, 0], noise=noise[0])
            style_vector.append(out_style)

            skip, out_style = self.to_rgb1(out, latent[:, 1])
            style_vector.append(out_style)

            i = 1
        else:
            out = self.style_input(latent[0])
            out, out_style = self.conv1(out, latent[0], noise=noise[0], input_is_stylespace=input_is_stylespace)
            style_vector.append(out_style)

            skip, out_style = self.to_rgb1(out, latent[1], input_is_stylespace=input_is_stylespace)
            style_vector.append(out_style)

            i = 2
        #循环对lantent的各层风格化卷积
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):   
            if not input_is_stylespace:
                out, out_style1 = conv1(out, latent[:, i], noise=noise1)
                out, out_style2 = conv2(out, latent[:, i + 1], noise=noise2)

                skip, rgb_style = to_rgb(out, latent[:, i + 2], skip)

                style_vector.extend([out_style1, out_style2, rgb_style])

                i += 2
            else:
                out, out_style1 = conv1(out, latent[i], noise=noise1, input_is_stylespace=input_is_stylespace)
                out, out_style2 = conv2(out, latent[i + 1], noise=noise2, input_is_stylespace=input_is_stylespace)
                skip, rgb_style = to_rgb(out, latent[i + 2], skip, input_is_stylespace=input_is_stylespace)
            
                style_vector.extend([out_style1, out_style2, rgb_style])

                i += 3

        image = skip

        if return_latents:
            return image, latent, style_vector

        else:
            return image, None     
        
class ConvLayer(keras.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))
        super().__init__(layers)
        

class ResBlock(keras.layers.Layer):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.blur_kernel=blur_kernel
        super().__init__()
    
    def build(self,input_shapes):
        self.conv1 = ConvLayer(self.in_channel, self.in_channel, 3)
        self.conv2 = ConvLayer(self.in_channel, self.out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            self.in_channel, self.out_channel, 1, downsample=True, activate=False, bias=False
        )
        super().build(input_shapes)
    
    def call(self,x):
        out=self.conv1(x)
        out = self.conv2(out)
        
        skip = self.skip(x)
        out=(out + skip) / math.sqrt(2)
        
        return out
    
class Discriminator(keras.layers.Layer):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        self.size=size
        self.log_size = int(math.log(self.size, 2))
        
        self.channel_multiplier=channel_multiplier
        self.blur_kernel=blur_kernel
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        
        self.stddev_group = 4
        self.stddev_feat = 1
    
        super().__init__()
        
    def build(self,input_shapes):
        convs=[ConvLayer(3, self.channels[self.size], 1)]
        in_channel = self.channels[self.size]
        
        for i in range(self.log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, self.blur_kernel))
            in_channel = out_channel
        
        self.convs = keras.Sequential(convs)

        self.final_conv = ConvLayer(in_channel + 1, self.channels[4], 3)
        self.final_linear = keras.Sequential(
            [EqualLinear(self.channels[4] * 4 * 4, self.channels[4], activation='fused_lrelu'),
            EqualLinear(self.channels[4], 1),]
        )
        super().build(input_shapes)
        
    def call(self,x):
        out = self.convs(x)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev=tf.reshape(
                    out,shape=[group, -1, self.stddev_feat, channel // self.stddev_feat, height, width]
               )
        stddev=tf.sqrt(tf.math.reduce_variance(stddev,axis=0)+ 1e-8)
        stddev=tf.squeeze(tf.reduce_mean(stddev,axis=[2,3,4], keepdims=True),axis=2)
        stddev=tf.tile(stddev,[group, 1, height, width])
        out=tf.concat([out, stddev],axis=1)

        out = self.final_conv(out)

        out=tf.reshape(out,shape=[batch,-1])
        out = self.final_linear(out)

        return out        

class generatorModel(keras.Model):
    def __init__(self,inputs,outputs,trainable_vars=None):
        super().__init__(inputs,outputs)
        self.trainable_vars=trainable_vars
    
    def train_step(self,inputs):
        x=inputs[0]["styles"]
        with tf.GradientTape() as tape:
            y_pred=self(x,training=True)
            y=tf.ones_like(y_pred)
            loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=y_pred))
            
        if self.trainable_vars is None:
            trainable_vars=self.trainable_variables
        else:
            trainable_vars=self.trainable_vars
        gradients=tape.gradient(loss,trainable_vars)
        self.optimizer.apply_gradients(zip(gradients,trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

class discriminatorModel(keras.Model):
    def __init__(self,inputs,outputs,trainable_vars=None):
        super().__init__(inputs,outputs)
        self.trainable_vars=trainable_vars
        
    def train_step(self,inputs):
        x=inputs[0]["x"]
        styles=inputs[0]["styles"]
        styles=inputs["styles"]
        with tf.GradientTape() as tape:
            real_labels,fake_labels=self(x,training=True)
            loss=(
                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_labels),logits=real_labels))+
                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_labels),logits=fake_labels))
            )
            
        if self.trainable_vars is None:
            trainable_vars=self.trainable_variables
        else:
            trainable_vars=self.trainable_vars
            
        gradients=tape.gradient(loss,trainable_vars)
        self.optimizer.apply_gradients(zip(gradients,trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

def styleGAN(model_config):
    x=keras.Input(shape=model_config.input_shape,dtype=tf.float32,name="x",batch_size=model_config.batch_size)
    styles=keras.Input(shape=(4,model_config.style_dim),dtype=tf.float32,name="styles",batch_size=model_config.batch_size)

    discriminator=Discriminator(size=model_config.discriminator_size)
    generator=Generator(
        size=model_config.generator_size,
        style_dim=model_config.style_dim,
        n_mlp=model_config.n_mlp
    )

    fakes,_=generator(styles)
    reals=x

    real_labels=discriminator(reals)
    fake_labels=discriminator(fakes)

    g_model=generatorModel(inputs=styles,outputs=fakes,trainable_vars=generator.trainable_variables)
    g_model.compile(
        optimizer="adam"
        )

    d_model=discriminatorModel(inputs=[x,styles],outputs=[real_labels,fake_labels],trainable_vars=discriminator.trainable_variables)
    d_model.compile(
        optimizer="adam"
        )
    return g_model,d_model