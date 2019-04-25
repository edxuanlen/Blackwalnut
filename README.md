(PS: 在自己本机上 tensorboard 的使用可以参考官方网站https://www.tensorflow.org/guide/summaries_and_tensorboard?hl=zh-cn, 如果无法访问该网站可以自行查找博客)
# 介绍

**什么是车标识别？**

  在学习 TensorFlow 的过程中，我们在开源社区中找到了一个名为 MobileNet 的模型，该模型能对图像进行分类。Google 发布的 tensorflow-for-poets 项目正好支持这个模型，我们希望能结合 TensorFlow 与 MobileNet，将其应用于分类不同车标的图片。同时以此作为基础，体验 ML/TF 中不同超参数对训练的影响。

**我们要创造什么？**

  在这个 Codelab 中，你将要使用车标图像训练一个 TensorFlow 模型。你将学会：

- 收集并分类车标图像
- 使用 MobileNet\_v1\_128 模型
- 训练模型并测试效果
- 分别测试 Learning Rate 为 0.1、0.001、0.0001、0.00001 时的训练效果
- 测试将预训练模型替换为 MobileNet\_v1\_224 后的训练效果
- 选择最好的训练超参数，执行预测

**你将需要什么？**

- BlackWalnut Labs. AI Cloud 访问账号

#
# 收集并分类车标图像

  本 Codelab 采用图像分类的方式来进行车标识别，因此需要收集的数据集很简单，只需要拍摄各种的车标图片即可。本 Codelab 提供了已拍摄完毕的演示用数据集，可以搭配自己拍摄的数据集使用。

  在拍摄数据集时，需要注意以下几点。

- 拍摄的照片的分辨率最好为 1:1
- 拍摄的车标占拍摄图片的大多数区域
- 每种车标拍摄相同数量的图片，且每种不少于 200 张
- 不要逆光拍摄

  新建名为 Codelab 的目录，并在该目录下新建名为 data 的目录。新建 Terminal 界面，进入 Codelab 目录，将 Tools/datasets 目录下的图片复制到该目录下。

cp -r ../Tools/datasets/\* data

  在 Codelab/data 目录下上传自己拍摄的图片并解压。（以 image.zip 为例）

unzip image.zip

rm image.zip

  在图像分类模型的训练过程中，模型需要知道训练的标签以及每个标签所对应的图片，因此需要将上一步中拍摄的图片按照车标种类分类。

  本 Codelab 所采用的训练程序基于 tensorflow-for-poets 项目，该项目的训练程序只需要将相同的车标归类到一个目录下即可，即在训练时传入的标签名即为归类的目录名。

  本 Codelab 提供的数据集已经按照名称归类，在 Codelab/data 目录下分别新建名为 BUICK、CHANGAN、BMW、NISSAN、GM 的目录，切换到 Terminal 界面并进入 Codelab/data 目录，将对应名称段的图片分别移动到这些目录中。

- BUICK:    1521121626331.jpg ~ 1521121911843.jpg
- CHANGAN:  1521171197033.jpg ~ 1521171332105.jpg
- BMW:    1521187291619.jpg ~ 1521187912599.jpg
- NISSAN:   1521189204685.jpg ~ 1521189316223.jpg
- GM:     1521257197105.jpg ~ 1521257469684.jpg

mv 1521121\* BUICK

mv 1521171\* CHANGAN

mv 1521187\* BMW

mv 1521189\* NISSAN

mv 1521257\* GM



#
# 使用
**MobileNet\_v1\_128**
#  模型

  能实现图像分类的模型有很多，比如 MobileNet、Inception、GoogleNet 等等，每种模型都有各自的特点，下图为 Google 的 Codelab 提供的各图像分类模型对比图，本 Codelab 中选择的是 MobileNet 模型。

 ![](https://github.com/edxuanlen/blackwalnut/blob/master/image.png)
 
  MobileNet V1 模型目前支持 2 种分辨率的图像，分别为 128\*128 和 224\*224，不过提供的训练程序会自动将传入的图片转换成指定的分辨率。

  在 Terminal 界面下声明环境变量，即训练时所要用到的模型以及图像参数。在当前步骤中，我们选择使用 MobileNet 模型，训练图像分辨率为 128\*128。

IMAGE\_SIZE=128

ARCHITECTURE=&quot;MobileNet\_1.0\_${IMAGE\_SIZE}&quot;

#
# 训练模型并评估结果

  车标识别的训练基于 Google 提供的 tensorflow-for-poets 项目，切换到 Terminal 界面，在 Codelab 目录下新建名为 scripts 的目录，进入 Codelab 目录，将 Tools/project 目录下提供的程序复制到该目录下。

cp -r ../Tools/project/\* .

  在 Terminal 界面下进入 Codelab 目录，参照下面的做法执行训练，该项目在训练时会自动把数据集分类成三种数据集，一种是训练集，一种是测试集，一种是验证集，在默认情况下，这三种数据集的比例分别为 80%、10%、10%。

- bottleneck\_dir 表示训练集转换后的文件存放的目录
- how\_many\_training\_steps 表示总训练的步数
- model\_dir 表示模型文件存放的目录，如果目录下不存在模型则会自动下载
- summaries\_dir 表示训练的中间结果存放的目录
- output\_graph 和 output\_labels 表示训练结果输出的目录
- architecture 就是我们前面设置的训练参数；image\_dir表示数据集存放的目录。
- testing\_percentage 测试集的比例，默认值为 10
- validation\_percentage 验证集的比例，默认为 10

python3 -m scripts.retrain \

--bottleneck\_dir=tmp/bottlenecks \

--how\_many\_training\_steps=5000 \

--model\_dir=model \

--summaries\_dir=tmp/training\_summaries/&quot;${ARCHITECTURE}&quot; \

--output\_graph=output/retrained\_graph\_128\_0\_01.pb \

--output\_labels=output/retrained\_labels\_128\_0\_01.txt \

--architecture=&quot;${ARCHITECTURE}&quot; \

--testing\_percentage=10 \

--validation\_percentage=10 \

--image\_dir=data \

--learning\_rate=0.001

  在执行训练后，程序会自动下载模型文件并进行训练，并且训练结束后在 output\_graph 目录下生成 retrained\_graph\_128\_0\_01.pb 文件，在 output\_labels 目录下生成 retrained\_labels\_128\_0\_01.txt 文件。

  有两种评估训练结果的方式，一种是通过 Tensorboard 查看训练过程中各个参数的值，一种是执行评估程序直接查看识别准确率。

  切换到 jupyter 界面，进入 Codelab/tmp/training\_summaries 目录，选择在当前目录下进入 Tensorboard 界面。

  Tensorboard 中的 accuracy\_1 的曲线图即为识别的准确率随着训练步数的增加的变化情况。

  如果想直接查看最终的准确率，先编辑 Codelab/scripts/evaluate.py 文件，根据下面的提示找到并修改程序。

# 找到这段代码
```python
with load\_graph(graph\_file\_name).as\_default() as graph:
    ground\_truth\_input = tf.placeholder(
        tf.float32, [None, 5], name=&#39;GroundTruthInput&#39;)
```
# 修改为这段代码with 
```python
load\_graph(graph\_file\_name).as\_default() as graph:
    ground\_truth\_input = tf.placeholder(
        tf.float32, [None, (&#39;替换为训练用标签的种类数&#39;)], name=&#39;GroundTruthInput&#39;)
```
# --------------------------------------------------------------------

# 找到这段代码

image\_dir = &#39;tf\_files/flower\_photos&#39;

# 修改为这段代码

image\_dir = &#39;data&#39;

# --------------------------------------------------------------------

# 找到这段代码
```python
with tf.Session(graph=graph) as sess:
    for filename, ground\_truth in zip(filenames, ground\_truths):
        image = Image.open(filename).resize((224,224),Image.ANTIALIAS)
```
# 修改为这段代码
```python
with tf.Session(graph=graph) as sess:
    for filename, ground\_truth in zip(filenames, ground\_truths):
        image = Image.open(filename).resize((128,128),Image.ANTIALIAS)
```
  切换到 Terminal 界面，进入 Codelab 目录，执行评估程序。

python3 -m scripts.evaluate output/retrained\_graph\_128\_0\_01.pb

  输出值中 Accuracy 即为训练的准确率。

#
# 调整 Learning Rate

  决定最终识别率的因素有三种，一种是训练的模型，一种是数据集的完整性和干净性，另一种是训练的超参数的合理性。

  超参数即不会在训练过程中随着训练的进行而发生变化的训练参数，在本 Codelab中，选择 learning\_rate 为超参数并将其调整为 0.1、0.001、0.0001、0.00001 再次训练。切换到 Terminal 界面，进入 Codelab 目录，选择不同的 Learning Rate 开始训练。

  Learning Rate 为 0.1 时：

python3 -m scripts.retrain \

    --bottleneck\_dir=tmp/bottlenecks \

    --how\_many\_training\_steps=5000 \

    --model\_dir=model \

    --summaries\_dir=tmp/training\_summaries/&quot;${ARCHITECTURE}&quot; \

    --output\_graph=output/retrained\_graph\_128\_0\_1.pb \

    --output\_labels=output/retrained\_labels\_128\_0\_1.txt \

    --architecture=&quot;${ARCHITECTURE}&quot; \

    --testing\_percentage=10 \

    --validation\_percentage=10 \

    --image\_dir=data \

    --learning\_rate=0.1

  Learning Rate 为 0.001 时：

python3 -m scripts.retrain \

    --bottleneck\_dir=tmp/bottlenecks \

    --how\_many\_training\_steps=5000 \

    --model\_dir=model \

    --summaries\_dir=tmp/training\_summaries/&quot;${ARCHITECTURE}&quot; \

    --output\_graph=output/retrained\_graph\_128\_0\_001.pb \

    --output\_labels=output/retrained\_labels\_128\_0\_001.txt \

    --architecture=&quot;${ARCHITECTURE}&quot; \

    --testing\_percentage=10 \

    --validation\_percentage=10 \

    --image\_dir=data \

    --learning\_rate=0.001

  Learning Rate 为 0.0001 时：

python3 -m scripts.retrain \

    --bottleneck\_dir=tmp/bottlenecks \

    --how\_many\_training\_steps=5000 \

    --model\_dir=model \

    --summaries\_dir=tmp/training\_summaries/&quot;${ARCHITECTURE}&quot; \

    --output\_graph=output/retrained\_graph\_128\_0\_0001.pb \

    --output\_labels=output/retrained\_labels\_128\_0\_0001.txt \

    --architecture=&quot;${ARCHITECTURE}&quot; \

    --testing\_percentage=10 \

    --validation\_percentage=10 \

    --image\_dir=data \

    --learning\_rate=0.0001

  Learning Rate 为 0.00001 时：

python3 -m scripts.retrain \

    --bottleneck\_dir=tmp/bottlenecks \

    --how\_many\_training\_steps=5000 \

    --model\_dir=model \

    --summaries\_dir=tmp/training\_summaries/&quot;${ARCHITECTURE}&quot; \

    --output\_graph=output/retrained\_graph\_128\_0\_00001.pb \

    --output\_labels=output/retrained\_labels\_128\_0\_00001.txt \

    --architecture=&quot;${ARCHITECTURE}&quot; \

    --testing\_percentage=10 \

    --validation\_percentage=10 \

    --image\_dir=data \

    --learning\_rate=0.00001

  切换到 Terminal 界面，进入 Codelab 目录，执行评估程序。

python3 -m scripts.evaluate output/retrained\_graph\_128\_0\_1.pb

python3 -m scripts.evaluate output/retrained\_graph\_128\_0\_001.pb

python3 -m scripts.evaluate output/retrained\_graph\_128\_0\_0001.pb

python3 -m scripts.evaluate output/retrained\_graph\_128\_0\_00001.pb

  输出值中 Accuracy 即为训练的准确率。

#
# 调整训练分辨率

  训练时所使用的图像分辨率也是训练的超参数之一，不同的分辨率同样会影响模型的训练结果。在 Learning Rate 为 0.01 的情况下，测试训练图像分辨率为 224\*224 时的结果。

  测试图像分辨率为 224\*224\*：

IMAGE\_SIZE=224

ARCHITECTURE=&quot;MobileNet\_1.0\_${IMAGE\_SIZE}&quot;

python3 -m scripts.retrain \

--bottleneck\_dir=tmp/bottlenecks \

--how\_many\_training\_steps=5000 \

--model\_dir=model \

--summaries\_dir=tmp/training\_summaries/&quot;${ARCHITECTURE}&quot; \

--output\_graph=output/retrained\_graph\_224\_0\_01.pb \

--output\_labels=output/retrained\_labels\_224\_0\_01.txt \

--architecture=&quot;${ARCHITECTURE}&quot; \

--testing\_percentage=10 \

--validation\_percentage=10 \

--image\_dir=data \

--learning\_rate=0.001

  编辑 Codelab/scripts/evaluate.py 文件，根据下面的提示找到并修改程序。

# 找到这段代码with tf.Session(graph=graph) as sess:

    for filename, ground\_truth in zip(filenames, ground\_truths):

        image = Image.open(filename).resize((128,128),Image.ANTIALIAS)

# 修改为这段代码with tf.Session(graph=graph) as sess:

    for filename, ground\_truth in zip(filenames, ground\_truths):

        image = Image.open(filename).resize((224,224),Image.ANTIALIAS)

Terminal 界面，进入 Codelab 目录，执行评估程序。

python3 -m scripts.evaluate output/retrained\_graph\_224\_0\_01.pb

  输出值中 Accuracy 即为训练的准确率。

#
# 执行预测

  在训练完成之后，可能结果会显示准确率 100% ，貌似不太可信啊，我们可以使用另外的交通标志图片进行测试。

  切换到 jupyter 项目树界面，在 Codelab 目录下新建名为 test 的目录。从网上寻找对应的车标图片，放在 test 目录下，这里寻找的照片不需要特别注意分辨率，测试程序中内置了统一分辨率的功能。

  参考下面的程序执行在线测试，测试结果会显示在输出中，每次执行会选择目录中的一张图片预测。      

- graph 表示训练后生成的 PB 文件所在的路径
- image 表示要预测的图片所在的路径

python3 -m scripts.label\_image \

--graph=output/retrained\_graph\_224\_0\_01.pb  \

--labels=output/retrained\_labels\_224\_0\_01.txt  \

--input\_height=224 \

--input\_width=224 \

--image=test/image.jpg
