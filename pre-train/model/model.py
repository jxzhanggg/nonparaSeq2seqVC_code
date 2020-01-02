import torch
from torch import nn
from torch.autograd import Variable
from math import sqrt
from .utils import to_gpu
from .decoder import Decoder
from .layers import SpeakerClassifier, SpeakerEncoder, AudioSeq2seq, TextEncoder,  PostNet, MergeNet


class Parrot(nn.Module):
    def __init__(self, hparams):
        super(Parrot, self).__init__()

        #print hparams
        # plus <sos> 
        self.embedding = nn.Embedding(
            hparams.n_symbols + 1, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std

        self.sos = hparams.n_symbols

        self.embedding.weight.data.uniform_(-val, val)

        self.text_encoder = TextEncoder(hparams)

        self.audio_seq2seq = AudioSeq2seq(hparams)

        self.merge_net = MergeNet(hparams)

        self.speaker_encoder = SpeakerEncoder(hparams)

        self.speaker_classifier = SpeakerClassifier(hparams)

        self.decoder = Decoder(hparams)
        
        self.postnet = PostNet(hparams)

        self.spemb_input = hparams.spemb_input

    def grouped_parameters(self,):

        params_group1 = [p for p in self.embedding.parameters()]
        params_group1.extend([p for p in self.text_encoder.parameters()])
        params_group1.extend([p for p in self.audio_seq2seq.parameters()])
        params_group1.extend([p for p in self.speaker_encoder.parameters()])
        params_group1.extend([p for p in self.merge_net.parameters()])
        params_group1.extend([p for p in self.decoder.parameters()])
        params_group1.extend([p for p in self.postnet.parameters()])

        return params_group1, [p for p in self.speaker_classifier.parameters()]

    def parse_batch(self, batch):
        text_input_padded, mel_padded, spc_padded, speaker_id, \
                    text_lengths, mel_lengths, gate_padded = batch
        
        text_input_padded = to_gpu(text_input_padded).long()
        mel_padded = to_gpu(mel_padded).float()
        spc_padded = to_gpu(spc_padded).float()
        speaker_id = to_gpu(speaker_id).long()
        text_lengths = to_gpu(text_lengths).long()
        mel_lengths = to_gpu(mel_lengths).long()
        gate_padded = to_gpu(gate_padded).float()

        return ((text_input_padded, mel_padded, text_lengths, mel_lengths),
                (text_input_padded, mel_padded, spc_padded,  speaker_id, gate_padded))


    def forward(self, inputs, input_text):
        '''
        text_input_padded [batch_size, max_text_len]
        mel_padded [batch_size, mel_bins, max_mel_len]
        text_lengths [batch_size]
        mel_lengths [batch_size]

        #
        predicted_mel [batch_size, mel_bins, T]
        predicted_gate [batch_size, T/r]
        alignment input_text==True [batch_size, T/r, max_text_len] or input_text==False [batch_size, T/r, T/r]
        text_hidden [B, max_text_len, hidden_dim]
        mel_hidden [B, T/r, hidden_dim]
        spearker_logit_from_mel [B, n_speakers]
        speaker_logit_from_mel_hidden [B, T/r, n_speakers]
        text_logit_from_mel_hidden [B, T/r, n_symbols]

        '''

        text_input_padded, mel_padded, text_lengths, mel_lengths = inputs

        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2) # -> [B, text_embedding_dim, max_text_len]
        text_hidden = self.text_encoder(text_input_embedded, text_lengths) # -> [B, max_text_len, hidden_dim]

        B = text_input_padded.size(0)
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding)

        # -> [B, n_speakers], [B, speaker_embedding_dim] 
        speaker_logit_from_mel, speaker_embedding = self.speaker_encoder(mel_padded, mel_lengths) 

        if self.spemb_input:
            T = mel_padded.size(2)
            audio_input = torch.cat([mel_padded, 
                speaker_embedding.detach().unsqueeze(2).expand(-1, -1, T)], 1)
        else:
            audio_input = mel_padded
        
        audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments = self.audio_seq2seq(
                audio_input, mel_lengths, text_input_embedded, start_embedding) 
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]
        
        
        speaker_logit_from_mel_hidden = self.speaker_classifier(audio_seq2seq_hidden) # -> [B, text_len, n_speakers]

        if input_text:
            hidden = self.merge_net(text_hidden, text_lengths)
        else:
            hidden = self.merge_net(audio_seq2seq_hidden, text_lengths)
        
        L = hidden.size(1)
        hidden = torch.cat([hidden, speaker_embedding.detach().unsqueeze(1).expand(-1, L, -1)], -1)

        predicted_mel, predicted_gate, alignments = self.decoder(hidden, mel_padded, text_lengths)

        post_output = self.postnet(predicted_mel)

        outputs = [predicted_mel, post_output, predicted_gate, alignments,
                  text_hidden, audio_seq2seq_hidden, audio_seq2seq_logit, audio_seq2seq_alignments, 
                  speaker_logit_from_mel, speaker_logit_from_mel_hidden,
                  text_lengths, mel_lengths]

        return outputs

    
    def inference(self, inputs, input_text, mel_reference, beam_width):
        '''
        decode the audio sequence from input
        inputs x
        input_text True or False
        mel_reference [1, mel_bins, T]
        '''
        text_input_padded, mel_padded, text_lengths, mel_lengths = inputs
        text_input_embedded = self.embedding(text_input_padded.long()).transpose(1, 2)
        text_hidden = self.text_encoder.inference(text_input_embedded)

        B = text_input_padded.size(0) # B should be 1
        start_embedding = Variable(text_input_padded.data.new(B,).fill_(self.sos))
        start_embedding = self.embedding(start_embedding) # [1, embedding_dim]

        #-> [B, text_len+1, hidden_dim] [B, text_len+1, n_symbols] [B, text_len+1, T/r]
        speaker_id, speaker_embedding = self.speaker_encoder.inference(mel_reference)

        if self.spemb_input:
            T = mel_padded.size(2)
            audio_input = torch.cat([mel_padded, 
                speaker_embedding.detach().unsqueeze(2).expand(-1, -1, T)], 1)
        else:
            audio_input = mel_padded
        
        audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments = self.audio_seq2seq.inference_beam(
                audio_input, start_embedding, self.embedding, beam_width=beam_width) 
        audio_seq2seq_hidden= audio_seq2seq_hidden[:,:-1, :] # -> [B, text_len, hidden_dim]

        # -> [B, n_speakers], [B, speaker_embedding_dim] 

        if input_text:
            hidden = self.merge_net.inference(text_hidden)
        else:
            hidden = self.merge_net.inference(audio_seq2seq_hidden)

        L = hidden.size(1)
        hidden = torch.cat([hidden, speaker_embedding.detach().unsqueeze(1).expand(-1, L, -1)], -1)
          
        predicted_mel, predicted_gate, alignments = self.decoder.inference(hidden)

        post_output = self.postnet(predicted_mel)

        return (predicted_mel, post_output, predicted_gate, alignments,
            text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments,
            speaker_id)


