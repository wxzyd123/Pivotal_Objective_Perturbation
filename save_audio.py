import os
import torch
import argparse
from tqdm import tqdm
import soundfile as sf
from torch.utils.data import DataLoader

import vits.utils as utils
from vits.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)


def main():
    parser  = argparse.ArgumentParser(description="The audio saving script.")

    parser.add_argument("--config_path", type=str, default="configs/onespeaker_vits.json", help="The configuration path for building model.")
    parser.add_argument("--noise_path", type=str, default="checkpoints/noises/VITS_POP_OneSpeaker.noise", help="The generated noise path.")
    parser.add_argument("--store_path", type=str, default="data/protected_audio", help="The store folder path of protected audio.")

    args = parser.parse_args()
    config_path = args.config_path
    hps = utils.get_hparams_from_file(config_path=config_path)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset,
                              num_workers=4,
                              shuffle=False,
                              collate_fn=collate_fn,
                              batch_size=hps.train.batch_size,
                              pin_memory=True,
                              drop_last=False)
    
    store_path = args.store_path
    if os.path.exists(store_path) is False:
        os.mkdir(store_path)

    noise_path = args.noise_path
    mode = noise_path.split("/")[2].split("_")[1]
    assert mode in ["POP", "EM", "RSP", "ESP"], print("The protective mode is wrong!")
    noises = torch.load(noise_path, map_location="cpu")

    count = 0
    batch_size = hps.train.batch_size
    for batch_index, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        noise = noises[batch_index]
        text, _, spec, spec_len, o_wav, wav_len, sid = batch
        p_wavs = torch.clamp(o_wav + noise, min=-1., max=1.)
        for p_index, p_wav in enumerate(p_wavs):
            current_p_wav = p_wav[:, :wav_len[p_index]]
            current_sid = sid[p_index]
            for data_index in range(0, batch_size):
                text, _, i_wav, inner_sid = train_dataset[data_index + batch_index * batch_size]
                if i_wav.shape == current_p_wav.shape and inner_sid == current_sid:
                    rate = hps.data.sampling_rate
                
                    output_file_name = os.path.join(store_path, f"{inner_sid.item()}_{count}_{mode}.wav")
                    audio = current_p_wav.numpy().squeeze()
                    sf.write(output_file_name, audio, samplerate=rate)
        
                    count += 1
                    break
    
    print(f"The process audio num is {count} of {len(train_dataset)}")
    assert count == len(train_dataset)
    
if __name__ == "__main__":
    main()