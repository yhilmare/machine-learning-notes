# FaceReplace（换脸）

制作这个项目最初的动因是三个月前的一天在网络上看到了关于将94版《射雕英雄传》中朱茵扮演的黄蓉的脸换成了杨幂，而且不得不说AI生成的杨幂的脸在视频中无论是表情，光照还是和原脸的契合程度都非常完美。

![](https://github.com/yhswjtuILMARE/Machine-Learning-Study-Notes/blob/master/pics/fr-3.jpg)

看完这个视频后除了惊叹于现在AI技术进展神速之外，也感叹现在的技术人员的技术应用能力真的很强，换做是我八辈子也想不到用神经网络来做换脸这种事。最初我直观上以为AI换脸是使用`GAN`来实现的，在网上搜索一番之后发现AI换脸是使用`deepfakes`工具完成的，这个工具在最初推出的时候非常流行，很多人用它来制作明星换脸视频，至于视频内容嘛，都是一些不可描述的东西。虽然`deepfakes`引起了不小的争议，但是技术总是无罪的。结合网上的资料和自己的分析，我发现最初版本的`deepfakes`并不是基于`GAN`而是一个相对简单的模型`AutoEncoder`。这令我非常开心，因为自己的当前的硬件绝对无法训练`GAN`这样的网络，但是跑跑`AutoEncoder`还是没问题的，于是就着手复现这个项目。

## 训练数据获取

我选取了斗鱼主播刘飞儿和我比较喜欢的明星神仙姐姐刘亦菲作为被换脸人和目标脸提供者。首先下载了刘飞儿的一个直播视频，使用一个小脚本将视频拆分成一帧一帧的图片形式：

![](https://github.com/yhswjtuILMARE/Machine-Learning-Study-Notes/blob/master/pics/fr-2.jpg)

当然，这些原始的视频图片中包含很多没有脸或者脸hen模糊的图像，为了模型的训练稳定，我将这些图片均从数据集中删掉了。接下来的工作是从这些图片中将人脸抠出来，这里使用`dlib`框架从数千张原始图片中抠出了人脸组成训练数据集：

![](https://github.com/yhswjtuILMARE/Machine-Learning-Study-Notes/blob/master/pics/fr-4.jpg)

## 模型结构

模型本身的结构非常简单，就是一个稍微变形的自编码器，其数学表达如下所示：

$$A^{'}=Encoder_{1}(A)$$
$$B^{'}=Encoder_{2}(B)$$
$$A^{''}=Decoder(A^{'})$$
$$B^{''}=Decoder(B^{'})$$

这个模型和普通的自编码器不一样的地方就在于：这个自编码器有两个不同的编码器，但是只拥有一个解码器，这样做的目的是让两个编码器分别编码A脸和B脸，目的是为了分别提取两张脸的全部特征，解码器是为了根据编码器提取到的特征还原出这张脸，两个编码器共用一个解码器促使该解码器能够针对不同的编码器编码出的特征都能够还原出人脸。通俗地解释这个模型就是让一个画家一直画A脸，直到他能“记住”A脸的所有细节，然后再让这个画家去画B脸，直到他能“记住”B脸的所有细节。最后无论他再画A脸还是B脸都会带有另外一张脸的特征和细节。

具体的网络结构如下所示：

```Python
Encoder: 64x64x3->8x8x512
x = input_
x = conv(128)(x)
x = conv(256)(x)
x = conv(512)(x)
x = conv(1024)(x)
x = Dense(ENCODER_DIM)(Flatten()(x))
x = Dense(4 * 4 * 1024)(x)
x = Reshape((4, 4, 1024))(x)
x = upscale(512)(x)

Decoder：8x8x512->64x64x3
x = input_
x = upscale(256)(x)
x = upscale(128)(x)
x = upscale(64)(x)
x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
```

## 训练Trick

如果直接将数据不加处理就直接训练，得到的结果是不理想的。一个训练小Trick是将数据集中的人脸做一定的变换和扭曲，这样能够训练自编码器更强的泛化能力，举例来说就是如此：

![](https://github.com/yhswjtuILMARE/Machine-Learning-Study-Notes/blob/master/pics/fr-7.jpg)

这样稍微扭曲一些的图片能够提高模型的泛化能力。

## 结果

由于实验室的服务器上没有GPU，因此训练了相当长的时间，最后结果只能说是差强人意：

![](https://github.com/yhswjtuILMARE/Machine-Learning-Study-Notes/blob/master/pics/fr-8.jpg)

还好，百度实图稍微给了我一些信心（笑哭）：

![](https://github.com/yhswjtuILMARE/Machine-Learning-Study-Notes/blob/master/pics/fr-9.jpg)

接下来的工作就是将AI生成的脸换到原图上去，这又是一个很大的工程，等到完成了之后再将细节写出来。