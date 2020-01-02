import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from .basic_layers import sort_batch, ConvNorm, LinearNorm, Attention, tile
from .utils import get_mask_from_lengths
from .beam import Beam, GNMTGlobalScorer
        
class SpeakerClassifier(nn.Module):
    '''
    - n layer CNN + PROJECTION
    '''
    def __init__(self, hparams):
        super(SpeakerClassifier, self).__init__()
        
        convolutions = []
        for i in range(hparams.SC_n_convolutions):
            #parse dim
            if i == 0:
                in_dim = hparams.encoder_embedding_dim
                out_dim = hparams.SC_hidden_dim
            elif i == (hparams.SC_n_convolutions-1):
                in_dim = hparams.SC_hidden_dim
                out_dim = hparams.SC_hidden_dim
            
            conv_layer = nn.Sequential(
                ConvNorm(in_dim,
                         out_dim,
                         kernel_size=hparams.SC_kernel_size, stride=1,
                         padding=int((hparams.SC_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='leaky_relu',
                         param=0.2),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.projection = LinearNorm(hparams.SC_hidden_dim, hparams.n_speakers)

    def forward(self, x):
        # x [B, T, dim]

        # -> [B, DIM, T]
        hidden = x.transpose(1, 2)
        for conv in self.convolutions:
            hidden = conv(hidden)
        
        # -> [B, T, dim]
        hidden = hidden.transpose(1, 2)
        logits = self.projection(hidden)

        return logits

class SpeakerEncoder(nn.Module):
    '''
    -  Simple 2 layer bidirectional LSTM with global mean_pooling

    '''
    def __init__(self, hparams):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(hparams.n_mel_channels, int(hparams.speaker_encoder_hidden_dim / 2), 
                            num_layers=2, batch_first=True,  bidirectional=True, dropout=hparams.speaker_encoder_dropout)
        self.projection1 = LinearNorm(hparams.speaker_encoder_hidden_dim, 
                                      hparams.speaker_embedding_dim, 
                                      w_init_gain='tanh')
        self.projection2 = LinearNorm(hparams.speaker_embedding_dim, hparams.n_speakers) 
    
    def forward(self, x, input_lengths):
        '''
         x  [batch_size, mel_bins, T]

         return 
         logits [batch_size, n_speakers]
         embeddings [batch_size, embedding_dim]
        '''
        x = x.transpose(1,2)
        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
 
        outputs = torch.sum(outputs,dim=1) / sorted_lengths.unsqueeze(1).float() # mean pooling -> [batch_size, dim]

        outputs = F.tanh(self.projection1(outputs))
        outputs = outputs[initial_index]
        # L2 normalizing #
        embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        logits = self.projection2(outputs)

        return logits, embeddings
    
    def inference(self, x): 
        
        x = x.transpose(1,2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs = torch.sum(outputs,dim=1) / float(outputs.size(1)) # mean pooling -> [batch_size, dim]
        outputs = F.tanh(self.projection1(outputs))
        embeddings = outputs / torch.norm(outputs, dim=1, keepdim=True)
        logits = self.projection2(outputs)

        pid = torch.argmax(logits, dim=1)

        return pid, embeddings 


class MergeNet(nn.Module):
    '''
    one layer bi-lstm
    '''
    def __init__(self, hparams):
        super(MergeNet, self).__init__()
        self.lstm = nn.LSTM(hparams.encoder_embedding_dim, int(hparams.encoder_embedding_dim/2), 
                            num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, x, input_lengths):
        '''
        x [B, T, dim]
        '''
        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        outputs = outputs[initial_index]

        return outputs

    def inference(self,x):

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class AudioEncoder(nn.Module):
    '''
    - Simple 2 layer bidirectional LSTM

    '''
    def __init__(self, hparams):
        super(AudioEncoder, self).__init__()

        if hparams.spemb_input:
            input_dim = hparams.n_mel_channels + hparams.speaker_embedding_dim
        else:
            input_dim = hparams.n_mel_channels
        
        self.lstm1 = nn.LSTM(input_dim, int(hparams.audio_encoder_hidden_dim / 2), 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hparams.audio_encoder_hidden_dim*hparams.n_frames_per_step_encoder,
                             int(hparams.audio_encoder_hidden_dim / 2), 
                            num_layers=1, batch_first=True, bidirectional=True)

        self.concat_hidden_dim = hparams.audio_encoder_hidden_dim*hparams.n_frames_per_step_encoder
        self.n_frames_per_step = hparams.n_frames_per_step_encoder
    
    def forward(self, x, input_lengths):
        '''
        x  [batch_size, mel_bins, T]

        return [batch_size, T, channels]
        '''
        x = x.transpose(1, 2)

        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths.cpu().numpy(), batch_first=True)

        self.lstm1.flatten_parameters()
        outputs, _ = self.lstm1(x_packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=x.size(1)) # use total_length make sure the recovered sequence length not changed

        outputs = outputs.reshape(x.size(0), -1, self.concat_hidden_dim)

        output_lengths = torch.ceil(sorted_lengths.float() / self.n_frames_per_step).long()
        outputs = nn.utils.rnn.pack_padded_sequence(
            outputs, output_lengths.cpu().numpy() , batch_first=True)

        self.lstm2.flatten_parameters()
        outputs, _ = self.lstm2(outputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs[initial_index], output_lengths[initial_index]
    
    def inference(self, x):
        
        x = x.transpose(1, 2)

        self.lstm1.flatten_parameters()
        outputs, _ = self.lstm1(x)
        outputs = outputs.reshape(1, -1, self.concat_hidden_dim)
        self.lstm2.flatten_parameters()
        outputs, _ = self.lstm2(outputs)

        return outputs


class AudioSeq2seq(nn.Module):
    '''
    - Simple 2 layer bidirectional LSTM

    '''
    def __init__(self, hparams):
        super(AudioSeq2seq, self).__init__()

        self.encoder = AudioEncoder(hparams)

        self.decoder_rnn_dim = hparams.audio_encoder_hidden_dim

        self.attention_layer = Attention(self.decoder_rnn_dim, hparams.audio_encoder_hidden_dim,
            hparams.AE_attention_dim, hparams.AE_attention_location_n_filters,
            hparams.AE_attention_location_kernel_size)

        self.decoder_rnn =  nn.LSTMCell(hparams.symbols_embedding_dim + hparams.audio_encoder_hidden_dim, 
            self.decoder_rnn_dim)
   
        def _proj(activation):
            if activation is not None:
                return nn.Sequential(LinearNorm(self.decoder_rnn_dim+hparams.audio_encoder_hidden_dim, 
                                     hparams.encoder_embedding_dim,
                                     w_init_gain=hparams.hidden_activation),
                                     activation)
            else:
                return LinearNorm(self.decoder_rnn_dim+hparams.audio_encoder_hidden_dim, 
                                  hparams.encoder_embedding_dim,
                                  w_init_gain=hparams.hidden_activation)
        
        if hparams.hidden_activation == 'relu':
            self.project_to_hidden = _proj(nn.ReLU())
        elif hparams.hidden_activation == 'tanh':
            self.project_to_hidden = _proj(nn.Tanh())
        elif hparams.hidden_activation == 'linear':
            self.project_to_hidden = _proj(None)
        else:
            print('Must be relu, tanh or linear.')
            assert False
        
        self.project_to_n_symbols= LinearNorm(hparams.encoder_embedding_dim,
            hparams.n_symbols + 1) # plus the <eos>
        self.eos = hparams.n_symbols
        self.activation = hparams.hidden_activation
        self.max_len = 100

    def initialize_decoder_states(self, memory, mask):
   
        B = memory.size(0)
        
        MAX_TIME = memory.size(1)
        
        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        
        self.attention_weigths = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weigths_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
    
    def map_states(self, fn):
        '''
        mapping the decoder states using fn
        '''
        self.decoder_hidden = fn(self.decoder_hidden, 0)
        self.decoder_cell = fn(self.decoder_cell, 0)
        self.attention_weigths = fn(self.attention_weigths, 0)
        self.attention_weigths_cum = fn(self.attention_weigths_cum, 0)
        self.attention_context = fn(self.attention_context, 0)


    def parse_decoder_outputs(self, hidden, logit, alignments):

        # -> [B, T_out + 1, max_time]
        alignments = torch.stack(alignments).transpose(0,1)
        # [T_out + 1, B, n_symbols] -> [B, T_out + 1,  n_symbols] 
        logit = torch.stack(logit).transpose(0, 1).contiguous()
        hidden = torch.stack(hidden).transpose(0, 1).contiguous()

        return hidden, logit, alignments

    def decode(self, decoder_input):

        cell_input = torch.cat((decoder_input, self.attention_context),-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input, 
            (self.decoder_hidden, 
            self.decoder_cell))

        attention_weigths_cat = torch.cat(
            (self.attention_weigths.unsqueeze(1),
            self.attention_weigths_cum.unsqueeze(1)),dim=1)
        
        self.attention_context, self.attention_weigths = self.attention_layer(
            self.decoder_hidden, 
            self.memory, 
            self.processed_memory, 
            attention_weigths_cat, 
            self.mask)

        self.attention_weigths_cum += self.attention_weigths
        
        hidden_and_context = torch.cat((self.decoder_hidden, self.attention_context), -1)

        hidden = self.project_to_hidden(hidden_and_context)
        
        # dropout to increasing g
        logit = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))

        return hidden, logit, self.attention_weigths
    
    def forward(self, mel, mel_lengths, decoder_inputs, start_embedding):
        '''
        decoder_inputs: [B, channel, T] 

        start_embedding [B, channel]

        return 
        hidden_outputs [B, T+1, channel]
        logits_outputs [B, T+1, n_symbols]
        alignments [B, T+1, max_time]

        '''

        memory, memory_lengths = self.encoder(mel, mel_lengths)

        decoder_inputs = decoder_inputs.permute(2, 0, 1) # -> [T, B, channel]
        decoder_inputs = torch.cat((start_embedding.unsqueeze(0), decoder_inputs), dim=0)

        self.initialize_decoder_states(memory, 
            mask=~get_mask_from_lengths(memory_lengths))

        hidden_outputs, logit_outputs, alignments = [], [], []
        while len(hidden_outputs) < decoder_inputs.size(0):

            decoder_input = decoder_inputs[len(hidden_outputs)]
            hidden, logit, attention_weights = self.decode(decoder_input)

            hidden_outputs += [hidden]
            logit_outputs += [logit]
            alignments += [attention_weights]
        
        hidden_outputs, logit_outputs, alignments = \
            self.parse_decoder_outputs(
                hidden_outputs, logit_outputs, alignments)

        return hidden_outputs, logit_outputs, alignments
        
    '''
    use beam search ? 
    '''
    def inference_greed(self, x, start_embedding, embedding_table):
        '''
        decoding the phone sequence using greed algorithm
        x [1, mel_bins, T]
        start_embedding [1,embedding_dim]
        embedding_table nn.Embedding class

        return
        hidden_outputs [1, ]
        '''
        MAX_LEN = 100

        decoder_input = start_embedding
        memory = self.encoder.inference(x)

        self.initialize_decoder_states(memory, mask=None)

        hidden_outputs, alignments, phone_ids = [], [], []
        while True:
            hidden, logit, attention_weights = self.decode(decoder_input)

            hidden_outputs += [hidden]
            alignments += [attention_weights]
            phone_id = torch.argmax(logit,dim=1)
            phone_ids += [phone_id]

            # if reaches the <eos>
            if phone_id.squeeze().item() == self.eos:
                break
            if len(hidden_outputs) == self.max_len:
                break
                print('Warning! The decoded text reaches the maximum lengths.')

            # embedding the phone_id
            decoder_input = embedding_table(phone_id) # -> [1, embedding_dim]

        hidden_outputs, phone_ids, alignments = \
            self.parse_decoder_outputs(hidden_outputs, phone_ids, alignments)

        return hidden_outputs, phone_ids, alignments

    def inference_beam(self, x, start_embedding, embedding_table,
            beam_width=20,):
 
        memory = self.encoder.inference(x).expand(beam_width, -1,-1)

        MAX_LEN = 100
        n_best = 5

        self.initialize_decoder_states(memory, mask=None)
        decoder_input = tile(start_embedding, beam_width)


        beam = Beam(beam_width, 0, self.eos, self.eos, 
            n_best=n_best, cuda=True, global_scorer=GNMTGlobalScorer())
        
        hidden_outputs, alignments, phone_ids = [], [], []

        for step in range(MAX_LEN):
            if beam.done():
                break
            
            hidden, logit, attention_weights = self.decode(decoder_input)
            logit = F.log_softmax(logit, dim=1)

            beam.advance(logit, attention_weights, hidden)
            select_indices = beam.get_current_origin()

            self.map_states(lambda state, dim: state.index_select(dim, select_indices))

            decoder_input = embedding_table(beam.get_current_state())

        scores, ks = beam.sort_finished(minimum=n_best)
        hyps, attn, hiddens = [], [], []

        for i, (times, k) in enumerate(ks[:n_best]):
            hyp, att, hid = beam.get_hyp(times, k)
            hyps.append(hyp)
            attn.append(att)
            hiddens.append(hid)

        return hiddens[0].unsqueeze(0), hyps[0].unsqueeze(0), attn[0].unsqueeze(0)
        


class TextEncoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(TextEncoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        self.dropout = hparams.text_encoder_dropout 

        
        def _proj(activation):
            if activation is not None:
                return nn.Sequential(LinearNorm(hparams.encoder_embedding_dim, 
                                     hparams.encoder_embedding_dim,
                                     w_init_gain=hparams.hidden_activation),
                                     activation)
            else:
                return LinearNorm(hparams.encoder_embedding_dim, 
                                  hparams.encoder_embedding_dim,
                                  w_init_gain=hparams.hidden_activation)
        
        if hparams.hidden_activation == 'relu':
            self.projection = _proj(nn.ReLU())
        elif hparams.hidden_activation == 'tanh':
            self.projection = _proj(nn.Tanh())
        elif hparams.hidden_activation == 'linear':
            self.projection = _proj(None)
        else:
            print('Must be relu, tanh or linear.')
            assert False
        
        #self.projection = nn.LinearNorm(hparams.encoder_embedding_dim, 
        #    hparams.encoder_embedding_dim,
        #    w_init_gain='relu') # fusing bi-directional info

    def forward(self, x, input_lengths):
        '''
        x: [batch_size, channel, T]

        return [batch_size, T, channel]
        '''
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), self.dropout, self.training)

        # -> [batch_size, T, channel]
        x = x.transpose(1, 2)

        x_sorted, sorted_lengths, initial_index = sort_batch(x, input_lengths)

        # pytorch tensor are not reversible, hence the conversion
        #input_lengths = input_lengths.cpu().numpy()
        sorted_lengths = sorted_lengths.cpu().numpy()

        x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, sorted_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        outputs = self.projection(outputs)

        return outputs[initial_index]

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), self.dropout, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs = self.projection(outputs)

        return outputs


class PostNet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_dim,
                             hparams.postnet_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_dim))
            )


        if hparams.predict_spectrogram:
            out_dim = hparams.n_spc_channels
            self.projection = LinearNorm(hparams.n_mel_channels, hparams.n_spc_channels, bias=False)
        else:
            out_dim = hparams.n_mel_channels
        
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_dim, out_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(out_dim))
            )
        self.dropout = hparams.postnet_dropout
        self.predict_spectrogram = hparams.predict_spectrogram

    def forward(self, input):
        # input [B, mel_bins, T]

        x = input
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), self.dropout, self.training)
        x = F.dropout(self.convolutions[-1](x), self.dropout, self.training)

        if self.predict_spectrogram:
            o = x + self.projection(input.transpose(1,2)).transpose(1,2)
        else:
            o = x + input

        return o
