import torch
import noisereduce as nr

from vits.mel_processing import spectrogram_torch
from vits.models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from vits.text.symbols import symbols


def nr_traditional(waves, sr=24000):
    reduced_waves = torch.tensor(waves).to(waves.device)

    for i, wave in enumerate(waves):
        pro_wave = nr.reduce_noise(y=wave.cpu().numpy(), sr=sr)
        tensor_wave = torch.tensor(pro_wave).to(waves.device)
        reduced_waves[i] = tensor_wave

    return reduced_waves


def get_spec(hps, waves, waves_len):
    spec_np = []
    spec_lengths = torch.LongTensor(len(waves))

    device = waves.device
    for index, wave in enumerate(waves):
        audio_norm = wave[:, :waves_len[index]]
        spec = spectrogram_torch(audio_norm,
                                 hps.filter_length, hps.sampling_rate,
                                 hps.hop_length, hps.win_length,
                                 center=False)
        spec = torch.squeeze(spec, 0)
        spec_np.append(spec)
        spec_lengths[index] = spec.size(1)

    max_spec_len = max(spec_lengths)
    spec_padded = torch.FloatTensor(len(waves), spec_np[0].size(0), max_spec_len)
    spec_padded.zero_()

    for i, spec in enumerate(waves):
        spec_padded[i][:, :spec_lengths[i]] = spec_np[i]

    return spec_padded.to(device), spec_lengths.to(device)


def build_models(hps, checkpoint_path=None):
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        try:
            checkpoint_dict = checkpoint['model']
        except:
            checkpoint_dict = checkpoint
        for layer_name, layer_params in net_g.state_dict().items():
            if layer_name in checkpoint_dict:
                checkpoint_dict_param = checkpoint_dict[layer_name]
                if checkpoint_dict_param.shape == layer_params.shape:
                    net_g.state_dict()[layer_name].copy_(checkpoint_dict_param)
                    # print(f"[·] Load the {layer_name} successfully!")
                else:
                    print(
                        f"[>] Layer {layer_name}, the layer size is {layer_params.shape}, the checkpoint size is {checkpoint_dict_param.shape}")
            else:
                print(f"[!] The layer {layer_name} is not found!")

    return net_g, net_d