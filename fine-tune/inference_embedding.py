import os
import numpy as np
import torch
import argparse

from hparams import create_hparams
from model import lcm
from train import load_model
from torch.utils.data import DataLoader
from reader import TextMelIDLoader, TextMelIDCollate, id2sp
from inference_utils import plot_data

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint_path', type=str,
                        help='directory to save checkpoints')
parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
args = parser.parse_args()

checkpoint_path=args.checkpoint_path

hparams = create_hparams(args.hparams)

model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False)
_ = model.eval()


def gen_embedding(speaker):

    training_list = hparams.training_list

    train_set_A = TextMelIDLoader(training_list, hparams.mel_mean_std, hparams.speaker_A,
            hparams.speaker_B, 
            shuffle=False,pids=[speaker])
            
    collate_fn = TextMelIDCollate(lcm(hparams.n_frames_per_step_encoder,
                            hparams.n_frames_per_step_decoder))

    train_loader_A = DataLoader(train_set_A, num_workers=1, shuffle=False,
                                sampler=None,
                                batch_size=1, pin_memory=False,
                                drop_last=True, collate_fn=collate_fn)

    with torch.no_grad():

        speaker_embeddings = []

        for i,batch in enumerate(train_loader_A):
            #print i
            x, y = model.parse_batch(batch)
            text_input_padded, mel_padded, text_lengths, mel_lengths, speaker_id = x
            speaker_id, speaker_embedding = model.speaker_encoder.inference(mel_padded)

            speaker_embedding = speaker_embedding.data.cpu().numpy()
            speaker_embeddings.append(speaker_embedding)

        speaker_embeddings = np.vstack(speaker_embeddings)
        
    print(speaker_embeddings.shape)
    if not os.path.exists('outdir/embeddings'):
        os.makedirs('outdir/embeddings')
    
    np.save('outdir/embeddings/%s.npy'%speaker, speaker_embeddings)
    plot_data([speaker_embeddings], 
        'outdir/embeddings/%s.pdf'%speaker)


print('Generating embedding of %s ...'%hparams.speaker_A)
gen_embedding(hparams.speaker_A)

print('Generating embedding of %s ...'%hparams.speaker_B)
gen_embedding(hparams.speaker_B)
