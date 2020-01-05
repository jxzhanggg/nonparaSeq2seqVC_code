import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from .basic_layers import ConvNorm, LinearNorm, ForwardAttentionV2, Prenet
from .utils import get_mask_from_lengths


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step_decoder
        self.hidden_cat_dim = hparams.encoder_embedding_dim + hparams.speaker_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.feed_back_last = hparams.feed_back_last

        if hparams.feed_back_last:
            prenet_input_dim = hparams.n_mel_channels
        else:
            prenet_input_dim = hparams.n_mel_channels * hparams.n_frames_per_step_decoder
        
        self.prenet = Prenet(
            prenet_input_dim ,
            hparams.prenet_dim)

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim[-1] + self.hidden_cat_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = ForwardAttentionV2(
            hparams.attention_rnn_dim, 
            self.hidden_cat_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            self.hidden_cat_dim + hparams.attention_rnn_dim,
            hparams.decoder_rnn_dim)
        
        self.linear_projection = LinearNorm(
            self.hidden_cat_dim + hparams.decoder_rnn_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step_decoder)

        self.gate_layer = LinearNorm(
            self.hidden_cat_dim + hparams.decoder_rnn_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        if self.feed_back_last:
            input_dim = self.n_mel_channels
        else:
            input_dim = self.n_mel_channels * self.n_frames_per_step
        
        decoder_input = Variable(memory.data.new(
            B,  input_dim).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.hidden_cat_dim).zero_())
        
        self.log_alpha = Variable(memory.data.new(B, MAX_TIME).fill_(-float(1e20)))
        self.log_alpha[:, 0].fill_(0.)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.reshape(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        if self.feed_back_last:
            decoder_inputs = decoder_inputs[:,:,-self.n_mel_channels:]
        
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B, MAX_TIME) -> (B, T_out, MAX_TIME)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        if alignments.size(0) == 1:
            gate_outputs = torch.stack(gate_outputs).unsqueeze(0)
        else:
            gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def attend(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        self.attention_context, self.attention_weights, self.log_alpha = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask, self.log_alpha)
        
        self.attention_weights_cum += self.attention_weights

        decoder_rnn_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)

        return decoder_rnn_input, self.attention_context, self.attention_weights

    def decode(self, decoder_input):

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))

        return self.decoder_hidden

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs  [B, encoder_max_time, hidden_dim]
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs  [B, mel_bin, T]
        memory_lengths: Encoder output lengths for attention masking. [B]
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder   [B, mel_bin, T]
        gate_outputs: gate outputs from the decoder [B, T/r]
        alignments: sequence of attention weights from the decoder [B, T/r, encoder_max_time]
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs) # [T/r + 1, B, prenet_dim ]

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]

            decoder_rnn_input, context, attention_weights = self.attend(decoder_input)
            
            decoder_rnn_output = self.decode(decoder_rnn_input)

            decoder_hidden_attention_context = torch.cat(
                (decoder_rnn_output, context), dim=1)
            
            mel_output = self.linear_projection(decoder_hidden_attention_context)
            gate_output = self.gate_layer(decoder_hidden_attention_context)

            mel_outputs += [mel_output.squeeze(1)] #? perhaps don't need squeeze
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)

            decoder_input_final, context, alignment = self.attend(decoder_input)

            #mel_output, gate_output, alignment = self.decode(decoder_input)
            decoder_rnn_output = self.decode(decoder_input_final)
            decoder_hidden_attention_context = torch.cat(
                (decoder_rnn_output, context), dim=1)
            
            mel_output = self.linear_projection(decoder_hidden_attention_context)
            gate_output = self.gate_layer(decoder_hidden_attention_context)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]


            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            if self.feed_back_last:
                decoder_input = mel_output[:,-self.n_mel_channels:]
            else:
                decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments