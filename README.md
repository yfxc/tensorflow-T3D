Tensorflow-T3D
====================
Tensorflow implementation for 'Temporal 3D ConvNets'(t3d)
****

|Author|yfxc|
|---|---
|E-mail|1512165940@qq.com

****
##Introduction
'T3D' model is can be applied to action recognition.Paper Url:https://arxiv.org/abs/1711.08200
Here is the tensorflow implementation with tf.slim to make code leaner.
**Different from the authors' original implementation.I convert traditional 3D convolution(eg. tf.nn.conv3d(...)) to P3D('pseudo-3d') convolution which can greatly recude number of parameters.**(P3D details:http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf )

## Preparing your own dataset.
Suppose you are about to use UCF dataset.Firstly converting videos to images is necessary.
To do this,you could run codes like follows:(Suppose UCF-101 dataset is in the same directory as the code-files.)
- **./process_video2image.sh UCF101** 
- And next step,you should get the 'train.list' and 'test.list' which you would afterwards fetch from for training data and testing
data individually:(number ‘5’ indicates that one-fifth of all data is testing data.)
- **./process_gettxt.sh UCF101 5**

**Note that:Due to the fact that *Relative Path* of the video clips exist in 'train.list' and 'test.list',
So you must make sure that 'DataGenerator.py' and UCF-101 are in the same directory! or modify the codes by yourself.**
## Train or Eval model     
After getting your own data and setting preference parameters in 'settings.py',You can run **python train.py --txt='./train.list'**(input parameter 'txt' has default value:'./train.list',in this way you can just run **python train.py**) to train model.
You can also train and test model in 'tf-p3d-train_eval.ipynb' with jupyter notebook.
## Others
- Changing the properties for data augmentation in 'DataAugmenter.py'
## Warning
- DO NOT use tf.contrib.layers.batch_norm() or slim.batch_norm() which may lead to wrong answers when testing.Using **tf.layers.batch_normalization(training=...)** instead.
