# 声纹识别

![GitHub last commit](https://img.shields.io/github/last-commit/konas122/Voiceprint-recognition)
![GitHub license](https://img.shields.io/github/license/konas122/Voiceprint-recognition?style=flat-square)

## requirements

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

## File Structure
```
.
├── audio.py
├── data
│   ├── dev
│   ├── test
│   └── train
├── eval.py
├── fine_tuning.py
├── img
├── loader.py
├── logs
│   └── acc
│       ├── test_acc
│       │   
│       └── train_acc         
├── loss.py
├── models
│   ├── tdnn.py
│   ├── tdnn_module.py
│   └── tdnn_pretrain.py
├── param.model
├── test.py
├── tools.py
└── train.py
```


## Usage
若要对模型进行微调，先下载本人训练好的模型[param.model](https://github.com/konas122/tdnn-on-directml/releases/download/v1.0/param.model)，并将该模型放在 `fine_tuning.py` 的同一目录下，然后运行 `fine_tuning.py`。

若想从零开始训练出一个模型，则运行 `train.py` 进行训练。

若要对模型进行评估，则运行 `test.py`。




## Dataset
这是我所用的数据集：https://pan.baidu.com/s/1_KrjPB27AHPrBa_1AeMQSQ?pwd=0mag	提取码：0mag	

当然，也可以用自己的数据集。只需在 `train.py` 的相同目录下创建 `data` 文件夹，并在 `data` 下创建子文件夹 `train`，然后将自己的训练数据放到 `train` 中。目前，这代码仅支持 `.wav` 格式的训练音频。


## Reference

Original ECAPA-TDNN paper
```
@inproceedings{desplanques2020ecapa,
  title={{ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification}},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech 2020},
  pages={3830--3834},
  year={2020}
}
```


## Acknowledge

We study many useful projects in our codeing process, which includes:

[Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification.](https://arxiv.org/abs/2005.07143v3)

[clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

[lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py).

[TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

Thanks for these authors to open source their code!
