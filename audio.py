import torch
import random
import librosa
import numpy as np
import librosa.display
from scipy.signal import medfilt
import matplotlib.pyplot as plt
# import torchaudio.transforms as T


path = '.\\voices'
name = 'a001.wav'
audio_filename = ".\\data\\test\\G2231\\T0055G2231S0076.wav"


def noise_augmentation(samples, min_db=40, max_db=80):
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    db = np.random.randint(low=min_db, high=max_db)
    db *= 1e-6
    noise = db * np.random.normal(0, 1, len(samples))  # 高斯分布
    # print(db)
    samples = samples + noise
    samples = samples.astype(data_type)
    return samples


def add_noise(x, snr, method='vectorized', axis=0):
    # Signal power
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)
    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))
    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)
    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')

    Psdb = 10 * np.log10(Ps)        # Signal power, in dB
    Pn = Psdb - snr         # Noise level necessary
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)      # Noise vector (or matrix)
    return x + n


def load_spectrogram(filename):
    wav, fs = librosa.load(filename, sr=16000)
    mag = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=512, n_mels=80,
                                         win_length=400, hop_length=160)
    mag = librosa.power_to_db(mag, ref=1.0, amin=1e-10, top_db=None)
    librosa.display.specshow(mag, sr=16000, x_axis='time', y_axis='mel')  # 画mel谱图
    plt.show()

    return mag


def audio_to_wav(filename, sr=16000, noise=False):
    wav, fs = librosa.load(filename, sr=sr)

    # wav1 = load_spectrogram(wav)
    # t = T.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
    #                      f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80)
    # wav2 = torch.from_numpy(wav)
    # wav2 = t(wav2)

    extended_wav = np.append(wav, wav)
    if len(extended_wav) < 41000:
        extended_wav = np.append(extended_wav, wav)
    if noise:
        extended_wav = add_noise(extended_wav, fs)
    return extended_wav, fs


def loadWAV(filename, noise=False):
    y, sr = audio_to_wav(filename=filename, noise=noise)
    assert len(y) >= 41000, f'Error: file {filename}\n'
    num = random.randint(0, len(y) - 41000)
    y = y[num:num + 41000]
    y = torch.from_numpy(y).float()
    return y


def load_pure_wav(filename, frame_threshold=10, noise=False):
    y, sr = audio_to_wav(filename=filename, noise=noise)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, win_length=1024, hop_length=512, n_fft=1024)
    Mfcc1 = medfilt(mfcc[0, :], 9)      # 对mfcc进行中值滤波
    pic = Mfcc1
    start = 0
    end = 0
    points = []
    min_data = min(pic) * 0.9
    for i in range((pic.shape[0])):
        if pic[i] < min_data and start == 0:
            start = i
        if pic[i] < min_data and start != 0:
            end = i
        elif pic[i] > min_data and start != 0:
            hh = [start, end]
            points.append(hh)
            start = 0
    if pic[-1] < min_data and start != 0:       # 解决 文件的最后为静音
        hh = [start, end]
        points.append(hh)
    distances = []
    for i in range(len(points)):
        two_ends = points[i]
        distance = two_ends[1] - two_ends[0]
        if distance > frame_threshold:
            distances.append(points[i])

    # out, _ = soundfile.read(filename)
    # out = out.astype(np.float32)
    if len(distances) == 0:     # 无静音段
        return y
    else:
        silence_data = []
        for i in range(len(distances)):
            if i == 0:
                start, end = distances[i]
                if start == 1:
                    internal_clean = y[0:0]
                else:
                    start = (start - 1) * 512   # 求取开始帧的开头
                    # end = (end - 1) * 512 + 1024
                    internal_clean = y[0:start - 1]
            else:
                _, end = distances[i - 1]
                start, _ = distances[i]
                start = (start - 1) * 512
                end = (end - 1) * 512 + 1024    # 求取结束帧的结尾
                internal_clean = y[end + 1:start]
            # hhh = np.array(internal_clean)
            silence_data.extend(internal_clean)
        ll = len(distances)     # 结尾音频处理
        _, end = distances[ll - 1]
        end = (end - 1) * 512 + 1024
        end_part_clean = y[end:len(y)]
        silence_data.extend(end_part_clean)
        y = silence_data
        y = torch.from_numpy(np.array(y)).float()
        return y


if __name__ == '__main__':
    a = load_pure_wav(audio_filename, noise=True)
    print(a.shape, a.dtype)
    _ = load_spectrogram(audio_filename)
    # a = np.array([[[-11, -10, -9, -8],
    #               [-7, -6, -5, -4],
    #                [-3, -2, -1, 0]],
    #              [[1, 2, 3, 4],
    #               [5, 6, 7, 8],
    #               [9, 10, 11, 12]]])
