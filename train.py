import os
import time
import torch
import argparse

from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


import vits.commons as commons
import vits.utils as utils
from vits.mel_processing import (
    mel_spectrogram_torch,
    spec_to_mel_torch,
    spectrogram_torch
)
from vits.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from vits.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from toolbox import build_models, get_spec
from evaluate import evaluation


def main():
    parser = argparse.ArgumentParser(description="The detailed setting for training...")

    parser.add_argument("--device", type=str, default="cuda", help="The training device which should be GPU or CPU.")
    parser.add_argument("--model_name", type=str, default="VITS", help="The surrogate model.")
    parser.add_argument("--dataset_name", type=str, default="OneSpeaker", help="The selected dataset to be protected.")
    parser.add_argument("--config_path", type=str, default="configs/onespeaker_vits.json", help="The configuration path for building model.")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/pretrained_ljs.pth", help="The checkpoint path of the pre-trained model.")
    parser.add_argument("--is_fixed", type=str, default="True", help="Training at the fixed patch or not.")

    args = parser.parse_args()
    device = args.device
    model_name = args.model_name
    dataset_name = args.dataset_name
    is_fixed = True if args.is_fixed == "True" else False
    assert torch.cuda.is_available(), "CPU training is not allowed."

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
    
    print(f"The dataset length is {len(train_dataset)}.")

    checkpoint_path = args.pretrained_path
    if checkpoint_path == "":
        raise "The pre-trained checkpoint is not be None!"

    net_g, net_d = build_models(hps, checkpoint_path=checkpoint_path)
    net_g, net_d = net_g.to(device), net_d.to(device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    
    epoch_str = 1
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    start_time = time.time()
    net_g.train(), net_d.train()
    for epoch in range(1, hps.train.epochs + 1):
        loss_disc_all, loss_gen_all = train(hps, [net_g, net_d], [optim_g, optim_d], train_loader, scaler, is_fixed)

        scheduler_g.step()
        scheduler_d.step()

        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))
        print(f"[{formatted_time}] Epoch {epoch}: D loss {loss_disc_all:.6f}, G loss {loss_gen_all:.6f}")

    if os.path.exists("checkpoints") is False:
        os.mkdir("checkpoints")

    save_path = f"./checkpoints/{model_name}_finetuning_{dataset_name}_{epoch}.pth"
    torch.save(net_g.state_dict(), save_path)
    print(f"Saving the checkpoint to {save_path}.")
    evaluation(net_g, config_path, model_name, dataset_name, "finetuning", device)


def train(hps, nets, optims, train_loader, scaler, is_fixed):
    net_g, net_d = nets
    optim_g, optim_d = optims

    device = next(net_g.parameters()).device
    loss_disc_all, loss_gen_all = 0, 0
    # for batch in tqdm(train_loader):
    for batch in train_loader:
        text, text_len, spec, spec_len, wav, wav_len, speakers = batch
        text, text_len = text.to(device), text_len.to(device)
        spec, spec_len = spec.to(device), spec_len.to(device)
        wav, wav_len = wav.to(device), wav_len.to(device)
        speakers = speakers.to(device)

        wav_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g.forward(text, text_len, spec, spec_len, speakers, is_fixed=is_fixed)

        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        wav_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
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

        wav = commons.slice_segments(wav, ids_slice * hps.data.hop_length, hps.train.segment_size)

        wav_d_hat_r, wav_d_hat_g, _, _ = net_d(wav, wav_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(wav_d_hat_r, wav_d_hat_g)
        loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        wav_d_hat_r, wav_d_hat_g, fmap_r, fmap_g = net_d(wav, wav_hat.detach())
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(wav_mel, wav_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(wav_d_hat_g)

        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl 

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

    return loss_disc_all, loss_gen_all


if __name__ == "__main__":
    main()
