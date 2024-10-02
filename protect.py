import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import vits.utils as utils
from vits.mel_processing import (
    mel_spectrogram_torch,
    spec_to_mel_torch,
    spectrogram_torch
)
import vits.commons as commons
from vits.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from vits.losses import (
    generator_loss,
    feature_loss,
    kl_loss
)
from toolbox import build_models, get_spec


def main():
    parser = argparse.ArgumentParser(description="The detailed setting for protecting...")

    parser.add_argument("--device", type=str, default="cuda", help="The training device which should be GPU or CPU.")
    parser.add_argument("--model_name", type=str, default="VITS", help="The surrogate model.")
    parser.add_argument("--dataset_name", type=str, default="OneSpeaker", help="The selected dataset to be protected.")
    parser.add_argument("--config_path", type=str, default="configs/onespeaker_vits.json", help="The configuration path for building model.")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/pretrained_ljs.pth", help="The checkpoint path of the pre-trained model.")
    parser.add_argument("--epsilon", type=float, default=8/255, help="The protective radius of the embedded perturbation by l_p norm.")
    parser.add_argument("--iterations", type=int, default=200, help="Running iterations.")
    parser.add_argument("--mode", type=str, default="POP", choices=["POP", "EM", "RSP", "ESP"],
                        help="The corresponding four protection modes in this paper.")


    args = parser.parse_args()
    device = args.device
    model_name = args.model_name
    dataset_name = args.dataset_name
    mode = args.mode

    config_path = args.config_path
    hps = utils.get_hparams_from_file(config_path=config_path)
    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed(hps.train.seed)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset,
                              num_workers=4,
                              shuffle=False,
                              collate_fn=collate_fn,
                              batch_size=hps.train.batch_size,
                              pin_memory=True,
                              drop_last=False)
    
    checkpoint_path = args.pretrained_path
    net_g, net_d = build_models(hps, checkpoint_path=checkpoint_path)
    net_g, net_d = net_g.to(device), net_d.to(device)

    for param in net_g.parameters():
        param.requires_grad = False
    for param in net_d.parameters():
        param.requires_grad = False
    
    noises = len(train_loader) * [None]
    epsilon = float(args.epsilon)
    alpha = epsilon / 10
    max_epoch = int(args.iterations)

    start_time = time.time()
    for batch_index, batch in enumerate(train_loader):
        noises[batch_index], loss = minimize_error(hps, [net_g, net_d], epsilon, alpha,
                                                   max_epoch, batch, mode, model_name)

        torch.cuda.empty_cache()
        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))
        print(f'[{formatted_time}] Batch {batch_index}, the loss is {loss:.6f}')

    if os.path.exists("checkpoints/noises") is False:
        os.mkdir("checkpoints/noises")

    save_path = f'checkpoints/noises/{model_name}_{mode}_{dataset_name}.noise'
    torch.save(noises, save_path)
    print(f"Saving the noise file to {save_path}.")


def minimize_error(hps, nets, epsilon, alpha, max_epoch, batch_data, mode, model_name):
    net_g, net_d = nets
    device = next(net_g.parameters()).device
    text, text_len, spec, spec_len, wav, wav_len, speakers = batch_data
    text, text_len = text.to(device), text_len.to(device)
    wav, wav_len = wav.to(device), wav_len.to(device)
    speakers = speakers.to(device)
    noise = torch.zeros(wav.shape).to(device)

    p_wav = Variable(wav.data + noise, requires_grad=True)
    p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)

    lr_noise = 5e-2
    opt_noise = optim.SGD([p_wav], lr=lr_noise, weight_decay=0.95)

    net_g.train()
    loss = 0.0
    for iteration in tqdm(range(max_epoch)):
        opt_noise.zero_grad()
        p_spec, spec_len = get_spec(hps.data, p_wav, wav_len)

        is_fixed = True if mode != "RSP" else False
        is_clip = True if mode != "ESP" else False

        wav_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(text, text_len, p_spec, spec_len, speakers,
                                                        is_fixed=is_fixed, is_clip=is_clip)

        if ids_slice is not None:
            p_wav_slice = commons.slice_segments(p_wav, ids_slice * hps.data.hop_length, hps.train.segment_size)
        else:
            p_wav_slice = p_wav

        loss_mel = compute_reconstruction_loss(hps, p_wav_slice, wav_hat)

        if mode == "POP":
            loss = loss_mel
        elif mode == "RSP":
            loss = loss_mel
        elif mode == "EM":
            wav_d_hat_r, wav_d_hat_g, fmap_r, fmap_g = net_d(p_wav_slice, wav_hat)
            loss_dur = torch.sum(l_length.float())
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(wav_d_hat_g)

            loss = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        elif mode == "ESP":
            loss = loss_mel
        else:
            raise "The protective mode is setting wrong!"

        p_wav.retain_grad = True
        loss.backward()
        opt_noise.step()
        grad = p_wav.grad

        noise = alpha * torch.sign(grad) * -1.
        p_wav = Variable(p_wav.data + noise, requires_grad=True)
        noise = torch.clamp(p_wav.data - wav.data, min=-epsilon, max=epsilon)
        p_wav = Variable(wav.data + noise, requires_grad=True)
        p_wav = Variable(torch.clamp(p_wav, min=-1., max=1.), requires_grad=True)

    return noise, loss


def compute_reconstruction_loss(hps, wav, wav_hat):
    wav_mel = mel_spectrogram_torch(
        wav.squeeze(1),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    wav_hat_mel = mel_spectrogram_torch(
        wav_hat.squeeze(1),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
    )
    loss_mel_wav = F.l1_loss(wav_mel, wav_hat_mel) * hps.train.c_mel

    return loss_mel_wav


if __name__ == "__main__":
    main()