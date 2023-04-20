# 声纹识别

![GitHub last commit](https://img.shields.io/github/last-commit/konas122/Voiceprint-recognition)
![GitHub license](https://img.shields.io/github/license/konas122/Voiceprint-recognition?style=flat-square)

## python第三方库

```
python==3.8
tensorboardX==2.6
tensorboard==2.11.2
numpy==1.23.5
librosa==0.9.2
scikit-learn==1.2.2
matplotlib==3.6.3
torch==1.13.1
torchaudio==0.13.1            
```

​	
## 训练
​	运行 `train.py` 进行训练。

​	该网络是在 `resnet18` 或 `vgg19` 的基础上再添加 LSTM 和线性层，从而实现声纹识别。
​	该项目同时也保留了单用 CNN 的方法( `net_cnn.py` )来实现声纹识别，其实效果也不差。


​	
## 训练数据
​	这是我所用的数据集：https://pan.baidu.com/s/1_KrjPB27AHPrBa_1AeMQSQ?pwd=0mag	提取码：0mag	

​	当然，也可以用自己的数据集。只需在 `train.py` 的相同目录下创建 `data` 文件夹，并在 `data` 下创建子文件夹 `train`，然后将自己的训练数据放到 `train` 中。目前，这代码仅支持 `.wav` 格式的训练音频。

​	

### Acknowledge

We study many useful projects in our codeing process, which includes:

[Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification.](https://arxiv.org/abs/2005.07143v3)

[clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

[lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py).

[TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

Thanks for these authors to open source their code!

未完待续...
