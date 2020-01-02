import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


import os
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader

from reader import TextMelIDLoader, TextMelIDCollate, id2ph, id2sp
from hparams import create_hparams
from model import Parrot, lcm
from train import load_model
import scipy.io.wavfile


########### Configuration ###########
hparams = create_hparams()

#generation list
hlist = '/home/jxzhang/Documents/DataSets/VCTK/list/hold_english.list'
tlist = '/home/jxzhang/Documents/DataSets/VCTK/list/eval_english.list'

# use seen (tlist) or unseen list (hlist)
test_list = tlist
checkpoint_path='outdir/checkpoint_0'
# TTS or VC task?
input_text=False
# number of utterances for generation
NUM=10
ISMEL=(not hparams.predict_spectrogram)
#####################################

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


model = load_model(hparams)

model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()

test_set = TextMelIDLoader(test_list, hparams.mel_mean_std, shuffle=True)
sample_list = test_set.file_path_list
collate_fn = TextMelIDCollate(lcm(hparams.n_frames_per_step_encoder,
                        hparams.n_frames_per_step_decoder))

test_loader = DataLoader(test_set, num_workers=1, shuffle=False,
                              sampler=None,
                              batch_size=1, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)



task = 'tts' if input_text else 'vc'
path_save = os.path.join(checkpoint_path.replace('checkpoint', 'test'), task)
path_save += '_seen' if test_list == tlist else '_unseen'
if not os.path.exists(path_save):
    os.makedirs(path_save)

print(path_save)

def recover_wav(mel, wav_path, ismel=False, 
        n_fft=2048, win_length=800,hop_length=200):
    
    if ismel:
        mean, std = np.load(hparams.mel_mean_std)
    else:
        mean, std = np.load(hparams.mel_mean_std.replace('mel','spec'))
    
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


text_input, text_targets, mel, spec, speaker_id, text_ranks = test_set[0]
reference_mel = mel.cuda().unsqueeze(0) 
ref_sp = id2sp[speaker_id.item()]

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

with torch.no_grad():

    errs = 0
    totalphs = 0

    for i, batch in enumerate(test_loader):
        if i == NUM:
            break
        
        sample_id = sample_list[i].split('/')[-1][9:17]
        print(('%d index %s, decoding ...'%(i,sample_id)))

        x, y = model.parse_batch(batch)
        predicted_mel, post_output, predicted_gate, alignments, \
            text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments, \
            speaker_id = model.inference(x, input_text, reference_mel, hparams.beam_width)

        post_output = post_output.data.cpu().numpy()[0]
        alignments = alignments.data.cpu().numpy()[0].T
        audio_seq2seq_alignments = audio_seq2seq_alignments.data.cpu().numpy()[0].T

        text_hidden = text_hidden.data.cpu().numpy()[0].T #-> [hidden_dim, max_text_len]
        audio_seq2seq_hidden = audio_seq2seq_hidden.data.cpu().numpy()[0].T
        audio_seq2seq_phids = audio_seq2seq_phids.data.cpu().numpy()[0] # [T + 1]
        speaker_id = speaker_id.data.cpu().numpy()[0] # scalar

        task = 'TTS' if input_text else 'VC'

        recover_wav(post_output, 
                    os.path.join(path_save, 'Wav_%s_ref_%s_%s.wav'%(sample_id, ref_sp, task)), 
                    ismel=ISMEL)
        
        post_output_path = os.path.join(path_save, 'Mel_%s_ref_%s_%s.npy'%(sample_id, ref_sp, task))
        np.save(post_output_path, post_output)
                
        plot_data([alignments, audio_seq2seq_alignments], 
            os.path.join(path_save, 'Ali_%s_ref_%s_%s.pdf'%(sample_id, ref_sp, task)))
        
        plot_data([np.hstack([text_hidden, audio_seq2seq_hidden])], 
            os.path.join(path_save, 'Hid_%s_ref_%s_%s.pdf'%(sample_id, ref_sp, task)))
         
        audio_seq2seq_phids = [id2ph[id] for id in audio_seq2seq_phids[:-1]]
        target_text = y[0].data.cpu().numpy()[0]
        target_text = [id2ph[id] for id in target_text[:]]

        print('Sounds like %s, Decoded text is '%(id2sp[speaker_id]))

        print(audio_seq2seq_phids)
        print(target_text)
       
        err = levenshteinDistance(audio_seq2seq_phids, target_text)
        print(err, len(target_text))

        errs += err
        totalphs += len(target_text)

print(float(errs)/float(totalphs))

        
        