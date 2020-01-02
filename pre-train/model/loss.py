import torch
from torch import nn
from torch.nn import functional as F
from .utils import get_mask_from_lengths

class ParrotLoss(nn.Module):
    def __init__(self, hparams):
        super(ParrotLoss, self).__init__()
        self.hidden_dim = hparams.encoder_embedding_dim
        self.ce_loss = hparams.ce_loss

        self.L1Loss = nn.L1Loss(reduction='none')
        self.MSELoss = nn.MSELoss(reduction='none')
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.n_frames_per_step = hparams.n_frames_per_step_decoder
        self.eos = hparams.n_symbols
        self.predict_spectrogram = hparams.predict_spectrogram

        self.contr_w = hparams.contrastive_loss_w
        self.consi_w = hparams.consistent_loss_w
        self.spenc_w = hparams.speaker_encoder_loss_w
        self.texcl_w = hparams.text_classifier_loss_w
        self.spadv_w = hparams.speaker_adversial_loss_w
        self.spcla_w = hparams.speaker_classifier_loss_w

    def parse_targets(self, targets, text_lengths):
        '''
        mel_target [batch_size, mel_bins, T]
        speaker_target [batch_size]
        text_target [batch_size, text_len]
        gate_target [batch_size, T]
        hard_alignments [batch_size, text_len, T]
        '''
        text_target, mel_target, spc_target, speaker_target, gate_target = targets

        B = gate_target.size(0)
        gate_target = gate_target.reshape(B, -1, self.n_frames_per_step)
        gate_target = gate_target[:, :, 0]

        padded = torch.tensor(text_target.data.new(B,1).zero_())
        text_target = torch.cat((text_target, padded), dim=-1)
        
        # adding the ending token for target
        for bid in range(B):
            text_target[bid, text_lengths[bid].item()] = self.eos

        return text_target, mel_target, spc_target, speaker_target, gate_target
    
    def forward(self, model_outputs, targets, input_text, eps=1e-5):

        '''
        predicted_mel [batch_size, mel_bins, T]
        predicted_gate [batch_size, T/r]
        alignment input_text==True [batch_size, T/r, max_text_len] or input_text==False [batch_size, T/r, T/r]
        text_hidden [B, max_text_len, hidden_dim]
        mel_hidden [B, max_text_len, hidden_dim]
        speaker_logit_from_mel [B, n_speakers]
        speaker_logit_from_mel_hidden [B, max_text_len, n_speakers]
        text_logit_from_mel_hidden [B, max_text_len+1, n_symbols+1]
        '''
        predicted_mel, post_output, predicted_gate, alignments,\
            text_hidden, mel_hidden, text_logit_from_mel_hidden, \
            audio_seq2seq_alignments, \
            speaker_logit_from_mel, speaker_logit_from_mel_hidden, \
             text_lengths, mel_lengths = model_outputs

        text_target, mel_target, spc_target, speaker_target, gate_target  = self.parse_targets(targets, text_lengths)

        ## get masks ##
        mel_mask = get_mask_from_lengths(mel_lengths, mel_target.size(2)).unsqueeze(1).expand(-1, mel_target.size(1), -1).float()
        spc_mask = get_mask_from_lengths(mel_lengths, mel_target.size(2)).unsqueeze(1).expand(-1, spc_target.size(1), -1).float()

        mel_step_lengths = torch.ceil(mel_lengths.float() / self.n_frames_per_step).long()
        gate_mask = get_mask_from_lengths(mel_step_lengths, 
                                    mel_target.size(2)/self.n_frames_per_step).float() # [B, T/r]
        text_mask = get_mask_from_lengths(text_lengths).float()
        text_mask_plus_one = get_mask_from_lengths(text_lengths + 1).float()

        # reconstruction loss #
        recon_loss = torch.sum(self.L1Loss(predicted_mel, mel_target) * mel_mask) / torch.sum(mel_mask)

        if self.predict_spectrogram:
            recon_loss_post = (self.L1Loss(post_output, spc_target) * spc_mask).sum() / spc_mask.sum()
        else:
            recon_loss_post = (self.L1Loss(post_output, mel_target) * mel_mask).sum() / torch.sum(mel_mask)
        
        gate_loss = torch.sum(self.BCEWithLogitsLoss(predicted_gate, gate_target) * gate_mask) / torch.sum(gate_mask)


        if self.contr_w == 0.:
            contrast_loss = torch.tensor(0.).cuda()
        else:
            # contrastive mask #
            contrast_mask1 =  get_mask_from_lengths(text_lengths).unsqueeze(2).expand(-1, -1, mel_hidden.size(1)) # [B, text_len] -> [B, text_len, T/r]
            contrast_mask2 = get_mask_from_lengths(text_lengths).unsqueeze(1).expand(-1, text_hidden.size(1), -1) # [B, T/r] -> [B, text_len, T/r]
            contrast_mask = (contrast_mask1 & contrast_mask2).float()
            text_hidden_normed = text_hidden / (torch.norm(text_hidden, dim=2, keepdim=True) + eps)
            mel_hidden_normed = mel_hidden / (torch.norm(mel_hidden, dim=2, keepdim=True) + eps)

            # (x - y) ** 2 = x ** 2 + y ** 2 - 2xy
            distance_matrix_xx = torch.sum(text_hidden_normed ** 2, dim=2, keepdim=True) #[batch_size, text_len, 1]
            distance_matrix_yy = torch.sum(mel_hidden_normed ** 2, dim=2)
            distance_matrix_yy = distance_matrix_yy.unsqueeze(1) #[batch_size, 1, text_len]

            #[batch_size, text_len, text_len]
            distance_matrix_xy = torch.bmm(text_hidden_normed, torch.transpose(mel_hidden_normed, 1, 2)) 
            distance_matrix = distance_matrix_xx + distance_matrix_yy - 2 * distance_matrix_xy
            
            TTEXT = distance_matrix.size(1)
            hard_alignments = torch.eye(TTEXT).cuda()
            contrast_loss = hard_alignments * distance_matrix + \
                (1. - hard_alignments) * torch.max(1. - distance_matrix, torch.zeros_like(distance_matrix))

            contrast_loss = torch.sum(contrast_loss * contrast_mask) / torch.sum(contrast_mask)
        
        if self.consi_w == 0.:
            consist_loss = torch.tensor(0.).cuda()
        else:
            consist_loss = self.MSELoss(text_hidden, mel_hidden)
            mask = text_mask.unsqueeze(2).expand(-1, -1, text_hidden.size(2))
            consist_loss = torch.sum(consist_loss * mask)/torch.sum(mask)

        n_speakers = speaker_logit_from_mel_hidden.size(2)
        TTEXT = speaker_logit_from_mel_hidden.size(1)
        n_symbols_plus_one = text_logit_from_mel_hidden.size(2)

        # speaker classification loss #
        speaker_encoder_loss = nn.CrossEntropyLoss()(speaker_logit_from_mel, speaker_target)
        _, predicted_speaker = torch.max(speaker_logit_from_mel,dim=1)
        speaker_encoder_acc = ((predicted_speaker == speaker_target).float()).sum() / float(speaker_target.size(0))

        speaker_logit_flatten = speaker_logit_from_mel_hidden.reshape(-1, n_speakers) # -> [B* TTEXT, n_speakers]
        _, predicted_speaker = torch.max(speaker_logit_flatten, dim=1)
        speaker_target_flatten = speaker_target.unsqueeze(1).expand(-1, TTEXT).reshape(-1)
        speaker_classification_acc = ((predicted_speaker == speaker_target_flatten).float() * text_mask.reshape(-1)).sum() / text_mask.sum()
        loss = self.CrossEntropyLoss(speaker_logit_flatten, speaker_target_flatten)

        speaker_classification_loss = torch.sum(loss * text_mask.reshape(-1)) / torch.sum(text_mask)

        # text classification loss #
        text_logit_flatten = text_logit_from_mel_hidden.reshape(-1, n_symbols_plus_one)
        text_target_flatten = text_target.reshape(-1)
        _, predicted_text =  torch.max(text_logit_flatten, dim=1)
        text_classification_acc = ((predicted_text == text_target_flatten).float()*text_mask_plus_one.reshape(-1)).sum()/text_mask_plus_one.sum()
        loss = self.CrossEntropyLoss(text_logit_flatten, text_target_flatten)
        text_classification_loss = torch.sum(loss * text_mask_plus_one.reshape(-1)) / torch.sum(text_mask_plus_one)

        # speaker adversival loss #
        flatten_target = 1. / n_speakers * torch.ones_like(speaker_logit_flatten)
        loss = self.MSELoss(F.softmax(speaker_logit_flatten, dim=1), flatten_target)
        mask = text_mask.unsqueeze(2).expand(-1,-1, n_speakers).reshape(-1, n_speakers)

        if self.ce_loss:
            speaker_adversial_loss = - speaker_classification_loss
        else:
            speaker_adversial_loss = torch.sum(loss * mask) / torch.sum(mask)
        
        loss_list = [recon_loss, recon_loss_post,  gate_loss,
                contrast_loss, consist_loss, speaker_encoder_loss, speaker_classification_loss,
                text_classification_loss, speaker_adversial_loss]
            
        acc_list = [speaker_encoder_acc, speaker_classification_acc, text_classification_acc]
        
        
        combined_loss1 = recon_loss + recon_loss_post + gate_loss + self.contr_w * contrast_loss + self.consi_w * consist_loss + \
            self.spenc_w * speaker_encoder_loss +  self.texcl_w * text_classification_loss + \
            self.spadv_w * speaker_adversial_loss

        combined_loss2 = self.spcla_w * speaker_classification_loss
        


        return loss_list, acc_list, combined_loss1, combined_loss2

