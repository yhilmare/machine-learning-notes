# 机器学习实践与笔记

机器学习是时下计算机科学的一个非常火热的分枝，笔者在进入研究生阶段的学习后也渐渐开始了解一些相关的算法。创建这个仓库的目的就是为了把笔者在机器学习领域遇到的问题，学到的知识，编写的代码分享给大家。

这个仓库中主要包含`java`和`python`两种类型的代码。包括svm，对数回归，随机森林和决策树等等。目前用java做机器学习研究学习的人不是很多，但是笔者作为一个JavaWeb开发者，一直想在一些web应用上集成机器学习的算法，因此创建了这个目录。java目录下目前还没有什么东西，笔者现在主要在python目录下更新，之后可能会逐渐添加一些java的代码。

除此之外，笔者还会发表一些学习笔记。这些笔记可能写的不是那么全面和正规，希望读者见谅，这些笔记主要以jupyter notebook的形式发表。这些笔记主要在`notebooks`目录中收录。

## 计算机视觉

在这里，可以看到与计算机视觉相关的小demo。比如前段时间的很火的`deepfakes`，笔者根据开源代码复现了核心代码使得程序能够根据一个人的脸生成另外一个人的脸，而这两张脸拥有相同的表情，光照：

![](https://github.com/yhswjtuILMARE/Machine-Learning-Study-Notes/blob/master/pics/fr-8.jpg)

在这里，你也可以看到根据数百张人脸合成一张平均脸的demo：

![](https://github.com/yhswjtuILMARE/Machine-Learning-Study-Notes/blob/master/pics/fa-1.jpg)

## 自然语言处理

在这里，可以看到使用循环神经网络生成的小说与诗词：

```
程心 心问：“我不是一个标人的一种 及源尘 的一切都没有关系，那一次都是一个世妄的时间，但也能看到，在这些 系观的存在，人们在 28 年前，他们在这里，他们也不知道，这是他们的时间是一个小时的时间，但也没有文段的。”

　　“我的！” 程心想起了一种声冷，“我的！” AA 的话学她说，“我的我也是 一切都有人的东西。这时，他们也没见过来。”关一帆走过程心的话。他说，“那是在我的人都不会，我们不知道那话的，他们不是这十的话。 他们在这个范像是 一种死票的 辨V，在她的 几睛， 她们也不是为了一个标座的。

　　在他的手指上了， “我想起来，她的几睛就是这十。”

　　“我不是我们的时间里，她在那个世界是 一个世界，她的几睛，她就在这个时代的一个人就是 一个世 界的世界。 他的目光中是一个人， 她不知道，我们在这种 话的话，他们的目光，但她的一切都不是 一个人。”

　　“我不知道，我们的生活，那个时代，他也不是一个人的那个世界的，她不是一个人，不管她是我。” 程心灯灯头。” 程心灯灯头，“你的目光中，那是一个 一个 方面的 草穷来时 代，在那一次，我的人都没有看，他们 都不再 是一个人的那种，
```
还可以看到对`word2vec`的实现以及相关的数学推导：

```
Nearst to "three" : four - <0.740>, five - <0.706>, two - <0.706>, zero - <0.703>, six - <0.695>, eight - <0.691>, seven - <0.661>, one - <0.633>
Nearst to "would" : can - <0.426>, to - <0.424>, transformation - <0.421>, says - <0.421>, adults - <0.416>, had - <0.398>, instrumental - <0.396>, like - <0.388>
Nearst to "known" : repeatedly - <0.413>, regarded - <0.394>, falls - <0.394>, joint - <0.375>, available - <0.369>, sir - <0.365>, UNK - <0.364>, select - <0.362>
Nearst to "world" : holding - <0.423>, died - <0.415>, holy - <0.394>, first - <0.393>, wayne - <0.391>, sea - <0.390>, mathbf - <0.382>, focus - <0.379>
Nearst to "history" : UNK - <0.424>, identical - <0.419>, state - <0.418>, orbital - <0.415>, next - <0.413>, boston - <0.411>, list - <0.410>, frequent - <0.408>
Nearst to "but" : however - <0.503>, that - <0.462>, ties - <0.462>, sky - <0.456>, exposed - <0.452>, generally - <0.441>, because - <0.430>, landscape - <0.414>
Nearst to "most" : supporters - <0.419>, earned - <0.406>, friendly - <0.405>, anarchist - <0.397>, absence - <0.394>, articles - <0.389>, apparent - <0.386>, difference - <0.385>
Nearst to "states" : co - <0.477>, symbolic - <0.428>, market - <0.412>, by - <0.408>, skills - <0.396>, legs - <0.395>, in - <0.392>, alphabet - <0.382>
Nearst to "only" : respectively - <0.434>, powerful - <0.404>, warming - <0.400>, transform - <0.392>, showing - <0.391>, theory - <0.389>, georgia - <0.383>, adventures - <0.382>
Nearst to "while" : and - <0.411>, enlightenment - <0.409>, started - <0.405>, conduct - <0.403>, convention - <0.396>, academy - <0.390>, lived - <0.390>, novels - <0.384>
Nearst to "other" : additional - <0.418>, fall - <0.415>, recovery - <0.398>, caught - <0.391>, gene - <0.390>, egyptian - <0.386>, different - <0.382>, some - <0.371>
Nearst to "which" : that - <0.490>, this - <0.435>, max - <0.424>, and - <0.410>, images - <0.408>, success - <0.407>, benefit - <0.404>, five - <0.403>
Nearst to "they" : why - <0.401>, assassination - <0.401>, he - <0.400>, occasions - <0.400>, southwest - <0.391>, inherited - <0.383>, democrats - <0.382>, norwegian - <0.378>
Nearst to "as" : detail - <0.459>, constitutional - <0.432>, error - <0.416>, cd - <0.412>, compiler - <0.407>, million - <0.405>, certain - <0.404>, creation - <0.397>
Nearst to "use" : deal - <0.438>, maintain - <0.406>, sphere - <0.392>, office - <0.390>, energy - <0.388>, effects - <0.385>, anarchist - <0.384>, every - <0.382>
Nearst to "has" : had - <0.468>, have - <0.455>, is - <0.439>, representative - <0.415>, was - <0.398>, kind - <0.395>, jordan - <0.385>, speech - <0.378>
```

## 结语

总之，笔者将在该项目中持续更新我关于机器学习的项目和理解。至于更深层次的文章请关注我的[ILMARE的博客](http://www.ilmareblog.com)，我会在上面发表一些文章来阐述一些机器学习的细节。
