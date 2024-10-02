import os

from tqdm import tqdm
import torch
import whisper
import jiwer
import soundfile as sf
from pymcd.mcd import Calculate_MCD

import vits.commons as commons
import vits.utils as utils
from vits.text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def evaluation(net_g, config_path, model_name, dataset_name, mode, device):
    config_path = config_path
    hps = utils.get_hparams_from_file(config_path=config_path)
    _ = net_g.eval()

    test_file = hps.data.test_files
    with open(test_file, 'r') as f:
        lines = f.readlines()
    
    if os.path.exists("evaluation") is False:
        os.mkdir("evaluation")
    if os.path.exists(f"evaluation/{model_name}") is False:
        os.mkdir(f"evaluation/{model_name}")
    if os.path.exists(f"evaluation/{model_name}/data") is False:
        os.mkdir(f"evaluation/{model_name}/data")
    if os.path.exists(f"evaluation/{model_name}/data/{dataset_name}") is False:
        os.mkdir(f"evaluation/{model_name}/data/{dataset_name}")

    output_path = f'evaluation/{model_name}/data/{dataset_name}/{mode}'
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # 1. Generate the evaluation dataset
    for index, line in tqdm(enumerate(lines), total=len(lines)):
        audio_path, sid, text = line.split('|')
        output_audio_name = sid + "_" + audio_path.split('/')[1] + "_" + str(index) + '.wav'

        stn_tst = get_text(text, hps)
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.tensor([int(sid)]).long().to(device)

        wav_gen = net_g.infer(x_tst, x_tst_lengths, sid,
                        noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        rate = hps.data.sampling_rate
        output_file_name = os.path.join(output_path, output_audio_name)

        sf.write(output_file_name, wav_gen, samplerate=rate)

    # 2. Generate the evaluation lists
    syn_path = output_path
    gt_audio_path = test_file
    assert os.path.exists(syn_path), "Synthesis path is not exists!"
    
    if os.path.exists(f"evaluation/{model_name}/evallists") is False:
        os.mkdir(f"evaluation/{model_name}/evallists")

    eval_list = f'./evaluation/{model_name}/evallists/{model_name}_{mode}_{dataset_name}_text.txt'
    with open(gt_audio_path, 'r') as f:
        gt_audio = f.readlines()

    syn_audio_list = os.listdir(syn_path)
    assert len(syn_audio_list) == len(gt_audio)

    with open(eval_list, 'w') as f:
        for index, gt in tqdm(enumerate(gt_audio), total=len(gt_audio)):
            gt_path = gt.split('|')[0]
            text = gt.replace("\n", "").split('|')[2]
            speaker_id = gt_path.split('/')[1]

            for syn_audio_path in syn_audio_list:
                syn_audio_name = syn_audio_path[:-4]
                inner_sid = syn_audio_name.split('_')[1]
                inner_index = syn_audio_name.split('_')[2]

                if inner_index == str(index):
                    assert inner_sid == speaker_id
                    gt_write_in = gt_path + '|' + text + "\n"
                    syn_write_in = os.path.join(syn_path, syn_audio_path) + '|' + text + "\n"
                    write_in = gt_write_in + syn_write_in
                    f.write(write_in)
                    break

    # 3. Evaluate the generated dataset
    # 3.1 MCD
    with open(eval_list, 'r') as f:
        audio_list = f.readlines()

    gt_audio_list = []
    syn_audio_list = []
    for index, audio_path in enumerate(audio_list):
        if index % 2 == 0:
            gt_audio_list.append(audio_path)
        else:
            syn_audio_list.append(audio_path)

    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    assert len(gt_audio_list) == len(syn_audio_list)

    mcd_value = 0.0
    for gt_path, syn_path in tqdm(zip(gt_audio_list, syn_audio_list), total=len(gt_audio_list)):
        gt_path, syn_path = gt_path.split('|')[0].replace('\n', ''), syn_path.split('|')[0].replace('\n', '')

        # MCD calculation
        mcd = mcd_toolbox.calculate_mcd(gt_path, syn_path)
        mcd_value += mcd

    mcd_value = mcd_value / len(gt_audio_list)
    print(f"Mode {mode}, MCD: ", {mcd_value})

    # 3.2 WER
    model = whisper.load_model("medium.en", device=device).to(device)

    with open(eval_list, 'r') as f:
        lines = f.readlines()

    WER_gt, WER_syn = 0.0, 0.0
    for index, line in tqdm(enumerate(lines), total=len(lines)):
        if index % 2 == 0:
            continue
        audio_path, gt_text = line.split('|')
        result = model.transcribe(audio_path, language="en")
        gen_text = result['text']
        wer = jiwer.wer(gt_text, gen_text)

        if index % 2 == 0:
            WER_gt += wer
        else:
            WER_syn += wer

    WER_gt /= (len(lines) // 2)
    WER_syn /= (len(lines) // 2)
    print(f"Mode {mode}: GT WER is {WER_gt:.6f}, Syn WER is {WER_syn:.6f}")