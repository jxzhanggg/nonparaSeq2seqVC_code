import os
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_alignment
from plotting_utils import plot_gate_outputs_to_numpy


class ParrotLogger(SummaryWriter):
    def __init__(self, logdir, ali_path='ali'):
        super(ParrotLogger, self).__init__(logdir)
        ali_path = os.path.join(logdir, ali_path)
        if not os.path.exists(ali_path):
            os.makedirs(ali_path)
        self.ali_path = ali_path

    def log_training(self, reduced_loss, reduced_losses, reduced_acces, grad_norm, learning_rate, duration,
                     iteration):
        
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("training.loss.recon", reduced_losses[0], iteration)
        self.add_scalar("training.loss.recon_post", reduced_losses[1], iteration)
        self.add_scalar("training.loss.stop",  reduced_losses[2], iteration)
        self.add_scalar("training.loss.contr", reduced_losses[3], iteration)
        self.add_scalar("training.loss.spenc", reduced_losses[4], iteration)
        self.add_scalar("training.loss.spcla", reduced_losses[5], iteration)
        self.add_scalar("training.loss.texcl", reduced_losses[6], iteration)
        self.add_scalar("training.loss.spadv", reduced_losses[7], iteration)

        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

        
        self.add_scalar('training.acc.spenc', reduced_acces[0], iteration)
        self.add_scalar('training.acc.spcla', reduced_acces[1], iteration)
        self.add_scalar('training.acc.texcl', reduced_acces[2], iteration)
    
    def log_validation(self, reduced_loss, reduced_losses, reduced_acces, model, y, y_pred, iteration, task):

        self.add_scalar('validation.loss.%s'%task, reduced_loss, iteration)
        self.add_scalar("validation.loss.%s.recon"%task, reduced_losses[0], iteration)
        self.add_scalar("validation.loss.%s.recon_post"%task, reduced_losses[1], iteration)
        self.add_scalar("validation.loss.%s.stop"%task,  reduced_losses[2], iteration)
        self.add_scalar("validation.loss.%s.contr"%task, reduced_losses[3], iteration)
        self.add_scalar("validation.loss.%s.spenc"%task, reduced_losses[4], iteration)
        self.add_scalar("validation.loss.%s.spcla"%task, reduced_losses[5], iteration)
        self.add_scalar("validation.loss.%s.texcl"%task, reduced_losses[6], iteration)
        self.add_scalar("validation.loss.%s.spadv"%task, reduced_losses[7], iteration)

        self.add_scalar('validation.acc.%s.spenc'%task, reduced_acces[0], iteration)
        self.add_scalar('validation.acc.%s.spcla'%task, reduced_acces[1], iteration)
        self.add_scalar('validatoin.acc.%s.texcl'%task, reduced_acces[2], iteration)
        
        predicted_mel, post_output, predicted_stop, alignments, \
            text_hidden, mel_hidden,  text_logit_from_mel_hidden, \
            audio_seq2seq_alignments, \
            speaker_logit_from_mel_hidden, \
            text_lengths, mel_lengths = y_pred

        text_target, mel_target, spc_target, speaker_target,  stop_target  = y

        stop_target = stop_target.reshape(stop_target.size(0), -1, int(stop_target.size(1)/predicted_stop.size(1)))
        stop_target = stop_target[:,:,0]

        # plot distribution of parameters
        #for tag, value in model.named_parameters():
        #    tag = tag.replace('.', '/')
        #    self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, stop target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        alignments = alignments.data.cpu().numpy()
        audio_seq2seq_alignments = audio_seq2seq_alignments.data.cpu().numpy()

        self.add_image(
            "%s.alignment"%task,
            plot_alignment_to_numpy(alignments[idx].T),
            iteration, dataformats='HWC')
        
        # plot more alignments
        plot_alignment(alignments[:4], self.ali_path+'/step-%d-%s.pdf'%(iteration, task))

        self.add_image(
            "%s.audio_seq2seq_alignment"%task,
            plot_alignment_to_numpy(audio_seq2seq_alignments[idx].T),
            iteration, dataformats='HWC')

        self.add_image(
            "%s.mel_target"%task,
            plot_spectrogram_to_numpy(mel_target[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        
        self.add_image(
            "%s.mel_predicted"%task,
            plot_spectrogram_to_numpy(predicted_mel[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        
        self.add_image(
            "%s.spc_target"%task,
            plot_spectrogram_to_numpy(spc_target[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        
        self.add_image(
            "%s.post_predicted"%task,
            plot_spectrogram_to_numpy(post_output[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')

        self.add_image(
            "%s.stop"%task,
            plot_gate_outputs_to_numpy(
                stop_target[idx].data.cpu().numpy(),
                F.sigmoid(predicted_stop[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
