import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import numpy as np
import librosa
import scipy.io.wavfile

def plot_data(data, fn, figsize=(12, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        if len(data) == 1:
            ax = axes
        else:
            ax = axes[i]
        g = ax.imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
        plt.colorbar(g, ax=ax)
    plt.savefig(fn)



def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def recover_wav(mel, wav_path, mel_mean_std, ismel=False,
        n_fft = 2048,win_length=800, hop_length=200):
    if ismel:
        mean, std = np.load(mel_mean_std)
    else:
        mean, std = np.load(mel_mean_std.replace('mel','spec'))
    
    mean = mean[:,None]
    std = std[:,None]
    mel = 1.2 * mel * std + mean
    mel = np.exp(mel)

    if ismel:
        filters = librosa.filters.mel(sr=16000, n_fft=2048, n_mels=80)
        inv_filters = np.linalg.pinv(filters)
        spec = np.dot(inv_filters, mel)
    else:
        spec = mel

    def _griffin_lim(stftm_matrix, shape, max_iter=50):
        y = np.random.random(shape)
        for i in range(max_iter):
            stft_matrix = librosa.core.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
            y = librosa.core.istft(stft_matrix, win_length=win_length, hop_length=hop_length)
        return y

    shape = spec.shape[1] * hop_length -  hop_length + 1

    y = _griffin_lim(spec, shape)
    scipy.io.wavfile.write(wav_path, 16000, y)
    return y