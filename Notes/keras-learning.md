## keras 安装和入门

安装 keras：

1. 可以使用 pip 安装

   ``` 
   pip install keras
   ```

2. 也可以 ancoda 安装：

   ``` 
   conda install keras
   ```

测试是否安装成功：

``` python
import keras
```

查看 keras 版本：

``` python
import keras
print(keras.__version__)
```



先看完该文：[一些基本概念 - Keras中文文档](https://keras-cn.readthedocs.io/en/latest/for_beginners/concepts/)

再看：[Keras TensorFlow教程：如何从零开发一个复杂深度学习模型 - SegmentFault 思否](https://segmentfault.com/a/1190000012645225)

其他阅读：

- [【私人笔记】深度学习框架keras踩坑记](https://mp.weixin.qq.com/s/suBo64ozWDSu-rQv118IVA)  [荐]





---



### 第一个例子：回归模型

首先我们在Keras中定义一个单层全连接网络，进行线性回归模型的训练：

``` 
# Regressor example
# Code: https://github.com/keloli/KerasPractise/edit/master/Regressor.py

import numpy as np
np.random.seed(1337)  
from keras.models import Sequential 
from keras.layers import Dense
import matplotlib.pyplot as plt

# 创建数据集
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # 将数据集随机化
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, )) # 假设我们真实模型为：Y=0.5X+2
# 绘制数据集plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]     # 把前160个数据放到训练集
X_test, Y_test = X[160:], Y[160:]       # 把后40个点放到测试集

# 定义一个model，
model = Sequential () # Keras有两种类型的模型，序贯模型（Sequential）和函数式模型
                      # 比较常用的是Sequential，它是单输入单输出的
model.add(Dense(output_dim=1, input_dim=1)) # 通过add()方法一层层添加模型
                                            # Dense是全连接层，第一层需要定义输入，
                                            # 第二层无需指定输入，一般第二层把第一层的输出作为输入

# 定义完模型就需要训练了，不过训练之前我们需要指定一些训练参数
# 通过compile()方法选择损失函数和优化器
# 这里我们用均方误差作为损失函数，随机梯度下降作为优化方法
model.compile(loss='mse', optimizer='sgd')

# 开始训练
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train) # Keras有很多开始训练的函数，这里用train_on_batch（）
    if step % 100 == 0:
        print('train cost: ', cost)

# 测试训练好的模型
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()    # 查看训练出的网络参数
                                        # 由于我们网络只有一层，且每次训练的输入只有一个，输出只有一个
                                        # 因此第一层训练出Y=WX+B这个模型，其中W,b为训练出的参数
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
```

最终的测试 cost 为：0.00313670327887，可视化结果如下图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190322153034.png)

### 第二个例子：手写数字识别

MNIST 数据集可以说是在业内被搞过次数最多的数据集了，毕竟各个框架的“hello world”都用它。这里我们也简单说一下在 Keras 下如何训练这个数据集：

``` 
# _*_ coding: utf-8 _*_
# Classifier mnist

import numpy as np
np.random.seed(1337)  
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# 下载数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data() # 下载不来，可以手动下载下来后缀npz的数据，下载地址：https://s3.amazonaws.com/img-datasets/mnist.npz

# 数据预处处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255. 
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 不使用model.add()，用以下方式也可以构建网络
model = Sequential([
    Dense(400, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# 定义优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy']) # metrics赋值为'accuracy'，会在训练过程中输出正确率

# 这次我们用fit()来训练网路
print('Training ------------')
model.fit(X_train, y_train, epochs=4, batch_size=32)

print('\nTesting ------------')
# 评价训练出的网络
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
```

简单训练后得到：test loss: 0.0970609934615，test accuracy: 0.9743

### 第三个例子：加经典网络的预训练模型（以VGG16为例）

(1) 当服务器不能联网时，需要把模型 `*.h5` 文件下载到用户目录下的 `~/.keras/model`，模型的预训练权重在载入模型时自动载入

(2) 通过以下代码加载VGG16：

``` 
# 使用VGG16模型
from keras.applications.vgg16 import VGG16
print('Start build VGG16 -------')

# 获取vgg16的卷积部分，如果要获取整个vgg16网络需要设置:include_top=True
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

# 创建自己的输入格式
# if K.image_data_format() == 'channels_first':
#   input_shape = (3, img_width, img_height)
# else:
#   input_shape = (img_width, img_height, 3)

input = Input(input_shape, name = 'image_input') # 注意，Keras有个层就是Input层

# 将vgg16模型原始输入转换成自己的输入
output_vgg16_conv = model_vgg16_conv(input)

# output_vgg16_conv是包含了vgg16的卷积层，下面我需要做二分类任务，所以需要添加自己的全连接层
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dense(1, activation='softmax', name='predictions')(x)

# 最终创建出自己的vgg16模型
my_model = Model(input=input, output=x)

# 下面的模型输出中，vgg16的层和参数不会显示出，但是这些参数在训练的时候会更改
print('\nThis is my vgg16 model for the task')
my_model.summary()
```

### 其他 Keras 使用细节

#### 指定占用的GPU以及多GPU并行

查看 GPU 使用情况语句（Linux）：

``` 
# 1秒钟刷新一次
watch -n 1 nvidia-smi
```

指定显卡：

``` 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

这里指定了使用编号为 2 的GPU，大家可以根据需要和实际情况来指定使用的 GPU。

GPU 并行：

``` 
from model import unet
G = 3 # 同时使用3个GPU
with tf.device("/cpu:0"):
        M = unet(input_rows, input_cols, 1)
model = keras.utils.training_utils.multi_gpu_model(M, gpus=G)
model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train,
              batch_size=batch_size*G, epochs=nb_epoch, verbose=0, shuffle=True,
              validation_data=(X_valid, y_valid))

model.save_weights('/path/to/save/model.h5')
```

#### 查看网络结构的命令

查看搭建的网络：`print (model.summary())`

效果如图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190322153515.png)

保存网络结构图：

``` 
# 你还可以用plot_model()来讲网络保存为图片
plot_model(my_model, to_file='my_vgg16_model.png')
```

#### 训练集与测试集图像的处理

```
from keras.preprocessing.image import ImageDataGenerator
print('Lodaing data -----------')
train_datagen=ImageDataGenerator()
test_datagen=ImageDataGenerator()
```

> *来源：[【Keras】Keras入门指南 - 简书](https://www.jianshu.com/p/e9c1e68a615e)*



---



## keras API 学习

### 1. concatenate()函数

keras 中 concatenate 源代码如下：

``` python
def concatenate(tensors, axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.
    # Returns
        A tensor.
    """
    if axis < 0:
        rank = ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0
 
    if py_all([is_sparse(x) for x in tensors]):
        return tf.sparse_concat(axis, tensors)
    else:
        return tf.concat([to_dense(x) for x in tensors], axis)
```

可以看出 keras 的 concatenate() 函数是披了外壳的 tf.concat()。不过用法没有 tf.concat() 那么复杂。对tf.concat() 解释可以看这篇博文《[tf.concat()详解](https://blog.csdn.net/leviopku/article/details/82380118) 》，如果只想了解 concatenate 的用法，可以不用移步。

axis=n 表示**从第n个维度进行拼接**，对于一个三维矩阵，axis 的取值可以为[-3, -2, -1, 0, 1, 2]。虽然 keras 用模除允许 axis 的取值可以在这个范围之外，但不建议那么用。

可以通过如下小段代码来理解：

``` python
import numpy as np
import cv2
import keras.backend as K
import tensorflow as tf

t1 = K.variable(np.array([
    [
        [1, 2],
        [2, 3]
     ], [
        [4, 4],
        [5, 3]
    ]
]))
t2 = K.variable(np.array([
    [
        [7, 4],
        [8, 4]
    ], [
        [2, 10],
        [15, 11]
    ]
]))
d0 = K.concatenate([t1, t2], axis=-2)
d1 = K.concatenate([t1, t2], axis=1)
d2 = K.concatenate([t1, t2], axis=-1)
d3 = K.concatenate([t1, t2], axis=2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(d0))
    print(sess.run(d1))
    print(sess.run(d2))
    print(sess.run(d3))
```

axis=-2，意思是从倒数第 2 个维度进行拼接，对于三维矩阵而言，这就等同于 axis=1。

axis=-1，意思是从倒数第 1 个维度进行拼接，对于三维矩阵而言，这就等同于 axis=2。

输出如下：

``` xml
d0:
[[[  1.   2.]
  [  2.   3.]
  [  7.   4.]
  [  8.   4.]]
 
 [[  4.   4.]
  [  5.   3.]
  [  2.  10.]
  [ 15.  11.]]]
 
d1:
[[[  1.   2.]
  [  2.   3.]
  [  7.   4.]
  [  8.   4.]]
 
 [[  4.   4.]
  [  5.   3.]
  [  2.  10.]
  [ 15.  11.]]]
 
d2:
[[[  1.   2.   7.   4.]
  [  2.   3.   8.   4.]]
 
 [[  4.   4.   2.  10.]
  [  5.   3.  15.  11.]]]
 
d3:
[[[  1.   2.   7.   4.]
  [  2.   3.   8.   4.]]
 
 [[  4.   4.   2.  10.]
  [  5.   3.  15.  11.]]]
```



### 2. 



---



## keras 的坑

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时间把你的想法转换为实验结果，是做好研究的关键。本人是keras的忠实粉丝，可能是因为它实在是太简单易用了，不用多少代码就可以将自己的想法完全实现，但是在使用的过程中还是遇到了不少坑，本文做了一个归纳，供大家参考。

Keras 兼容的 Python 版本: Python 2.7-3.6。

**详细教程请参阅Keras官方中文文档：http://keras-cn.readthedocs.io/en/latest/** 

**1、Keras输出的loss，val这些值如何保存到文本中去：**

Keras中的fit函数会返回一个History对象，它的History.history属性会把之前的那些值全保存在里面，如果有验证集的话，也包含了验证集的这些指标变化情况，具体写法：

``` python
hist=model.fit(train_set_x,train_set_y,batch_size=256,shuffle=True,nb_epoch=nb_epoch,validation_split=0.1)
with open('log_sgd_big_32.txt','w') as f:
    f.write(str(hist.history))
```

我觉得保存之前的loss，val这些值还是比较重要的，在之后的调参过程中有时候还是需要之前loss的结果作为参考的，特别是你自己添加了一些自己的loss的情况下，但是这样的写法会使整个文本的取名比较乱，所以其实可以考虑使用Aetros的插件，Aetros网址，这是一个基于Keras的一个管理工具，可以可视化你的网络结构，中间卷积结果的可视化，以及保存你以往跑的所有结果，还是很方便的，就是有些不稳定，有时候会崩。。。

**history对象包含两个重要属性：**

- epoch：训练的轮数
- history：它是一个字典，包含val_loss,val_acc,loss,acc四个key。

**2、关于训练集，验证集和测试集：**

其实一开始我也没搞清楚这个问题，拿着测试集当验证集用，其实验证集是从训练集中抽取出来用于调参的，而测试集是和训练集无交集的，用于测试所选参数用于该模型的效果的，这个还是不要弄错了。。。在Keras中，验证集的划分只要在fit函数里设置validation_split的值就好了，这个对应了取训练集中百分之几的数据出来当做验证集。但由于shuffle是在validation _split之后执行的，所以如果一开始训练集没有shuffle的话，有可能使验证集全是负样本。测试集的使用只要在evaluate函数里设置就好了。

print model.evaluate（test_set_x，test_set_y ,batch_size=256）

这里注意evaluate和fit函数的默认batch_size都是32，自己记得修改。

**总结：**

- 验证集是在fit的时候通过validation_split参数自己从训练集中划分出来的；
- 测试集需要专门的使用evaluate去进行评价。

**3、关于优化方法使用的问题之学习率调整**

开始总会纠结哪个优化方法好用，但是最好的办法就是试，无数次尝试后不难发现，Sgd的这种学习率非自适应的优化方法，调整学习率和初始化的方法会使它的结果有很大不同，但是由于收敛确实不快，总感觉不是很方便，我觉得之前一直使用Sgd的原因一方面是因为优化方法不多，其次是用Sgd都能有这么好的结果，说明你网络该有多好啊。其他的Adam，Adade，RMSprop结果都差不多，Nadam因为是adam的动量添加的版本，在收敛效果上会更出色。所以如果对结果不满意的话，就把这些方法换着来一遍吧。 

**（1）方法一**：通过LearningRateScheduler实现学习率调整

有很多初学者人会好奇怎么使sgd的学习率动态的变化，其实Keras里有个反馈函数叫LearningRateScheduler，具体使用如下：

``` python
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.fit(train_set_x, train_set_y, validation_split=0.1, nb_epoch=200, batch_size=256, callbacks=[lrate])
```

上面代码是使学习率指数下降，具体如下图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190416095822.png)

**（2）方式二：**最直接的调整学习率方式

当然也可以直接在sgd声明函数中修改参数来直接修改学习率，学习率变化如下图：

``` python
sgd = SGD(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=0.9, nesterov=True)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190416095848.png)

具体可以参考这篇文章Using Learning Rate Schedules for Deep Learning Models in Python with Keras

除此之外，还有一种学利率调整方式，即

**（3）方法三：**通过ReduceLROnPlateau调整学习率

``` python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

当评价指标不在提升时，减少学习率。当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在patience个epoch中看不到模型性能提升，则减少学习率

参数：

``` xml
monitor：被监测的量
factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
epsilon：阈值，用来确定是否进入检测值的“平原区”
cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
min_lr：学习率的下限
```

代码示例如下：

``` python
from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

model.fit(train_x, train_y, batch_size=32, epochs=5, validation_split=0.1, callbacks=[reduce_lr])
```

**4、如何用 Keras 处理超过内存的数据集？**

你可以使用 model.train_on_batch(x，y) 和 model.test_on_batch(x，y) 进行批量训练与测试。请参阅 模型文档。或者，你可以编写一个生成批处理训练数据的生成器，然后使用 model.fit_generator(data_generator，steps_per_epoch，epochs) 方法。

**5、Batchnormalization层的放置问题：**

BN层是真的吊，简直神器，除了会使网络搭建的时间和每个epoch的时间延长一点之外，但是关于这个问题我看到了无数的说法，对于卷积和池化层的放法，又说放中间的，也有说池化层后面的，对于dropout层，有说放在它后面的，也有说放在它前面的，对于这个问题我的说法还是试！虽然麻烦。。。但是DL本来不就是一个偏工程性的学科吗。。。还有一点是需要注意的，就是BN层的参数问题，我一开始也没有注意到，仔细看BN层的参数：

``` python
keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')
```

mode：整数，指定规范化的模式，取0或1

- 0：按特征规范化，输入的各个特征图将独立被规范化。规范化的轴由参数axis指定。注意，如果输入是形如（samples，channels，rows，cols）的4D图像张量，则应设置规范化的轴为1，即沿着通道轴规范化。输入格式是‘tf’同理。

- 1：按样本规范化，该模式默认输入为2D

我们大都使用的都是mode=0也就是按特征规范化，对于放置在卷积和池化之间或之后的4D张量，需要设置axis=1，而Dense层之后的BN层则直接使用默认值就好了。

**6、在验证集的误差不再下降时，如何中断训练？**

你可以使用 EarlyStopping 回调：

``` python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

**总结：关于callbacks参数的妙用**

（1）查询每隔epoch之后的loss和acc

（2）通过LearningRateScheduler实现衰减学习率或自定义衰减学习率

（3）通过EarlyStopping实现中断训练

（4）我们还可以自己定义回调函数，所为回调函数其实就是在训练完每一个epoch之后我们希望实现的操作。

**7.如何「冻结」网络层？**

「冻结」一个层意味着将其排除在训练之外，即其权重将永远不会更新。这在微调模型或使用固定的词向量进行文本输入中很有用。有两种方式实现：

**方式一：**在构造层的时候传递一个bool类型trainable参数，如下：

``` python
frozen_layer = Dense(32, trainable=False)
```

您可以将 trainable 参数（布尔值）传递给一个层的构造器，以将该层设置为不可训练的：

**方式二：**通过层对象的trainable属性去设置，如下：

```python
x = Input(shape=(32,))
layer = Dense(32)  #构造一个层
layer.trainable = False  #设置层的trainable属性
y = layer(x)
```

注意：可以在实例化之后将网络层的 trainable 属性设置为 True 或 False。为了使之生效，在修改 trainable 属性之后，需要在模型上调用 compile()。及重新编译模型。

**8.如何从 Sequential 模型中移除一个层？**

你可以通过调用模型的 .pop() 来删除 Sequential 模型中最后添加的层：

``` python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

——from：[深度学习框架keras踩坑记](https://mp.weixin.qq.com/s/suBo64ozWDSu-rQv118IVA)





---

*update：2019-04-16* 









