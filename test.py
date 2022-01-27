import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
# from models.asr.transformer import Transformer, Encoder, Decoder
from utils import constant
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer, calculate_cer_en_zh,calc_f1,bleu,distinct
from utils.functions import save_model, load_model
from utils.lstm_utils import LM


def evaluate(model, test_loader, lm=None):
    """
    Evaluation
    args:
        model: Model object
        test_loader: DataLoader object
    """
    model.eval()
    F1data = []
    predictions = []
    total_word, total_char, total_cer, total_wer = 0, 0, 0, 0
    total_en_cer, total_zh_cer, total_en_char, total_zh_char = 0, 0, 0, 0
    device = torch.device("cpu")
    with torch.no_grad():
        test_pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
        for i, (data) in enumerate(test_pbar):
            src, tgt, src_percentages, src_lengths, tgt_lengths,targets2 = data
            tgt = tgt.to(device)
            src = src.to(device)
            src_percentages = src_percentages.to(device)
            src_lengths = src_lengths.to(device)
            tgt_lengths = tgt_lengths.to(device)
            targets2 = targets2.to(device)

            # if constant.USE_CUDA:
            #     src = src.cuda()
            #     tgt = tgt.cuda()

            batch_ids_hyps, batch_strs_hyps, batch_strs_gold = model.evaluate2( src,
                                                                               src_lengths,
                                                                               tgt)

            for x in range(len(batch_strs_gold)):
                hyp = batch_strs_hyps[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(constant.PAD_CHAR, "")
                gold = batch_strs_gold[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(constant.PAD_CHAR, "")

                wer = calculate_wer(hyp, gold)
                cer = calculate_cer(hyp.strip(), gold.strip())
                F1data.append((hyp.strip(), gold.strip()))
                predictions.append(hyp.strip())
                en_cer, zh_cer, num_en_char, num_zh_char = calculate_cer_en_zh(hyp, gold)
                total_en_cer += en_cer
                total_zh_cer += zh_cer
                total_en_char += num_en_char
                total_zh_char += num_zh_char

                total_wer += wer
                total_cer += cer
                total_word += len(gold.split(" "))
                total_char += len(gold)

            test_pbar.set_description("TEST CER:{:.2f}% WER:{:.2f}% CER_EN:{:.2f}% CER_ZH:{:.2f}%".format(
                total_cer*100/total_char, total_wer*100/total_word, total_en_cer*100/max(1, total_en_char), total_zh_cer*100/max(1, total_zh_char)))
        f1 = calc_f1(F1data)
        bleu_1, bleu_2 = bleu(F1data)
        unigrams_distinct, bigrams_distinct = distinct(predictions)
        print("TEST CER:{:.2f}% WER:{:.2f}% f1:{:.2f}% bleu_1:{:.2f}% bleu_2:{:.2f}% unigrams_distinct:{:.2f}% bigrams_distinct:{:.2f}%".format(
                total_cer * 100 / total_char, total_wer * 100 / total_word, f1, bleu_1, bleu_2, unigrams_distinct,
                bigrams_distinct))


if __name__ == '__main__':
    args = constant.args

    start_iter = 0

    # Load the model
    load_path = constant.args.continue_from
    p = "/home/zmw/big_space/zhangmeiwei_space/asr_res_model/multitask/thch30/transformer_model/3_5/best_model.th"
    model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(p)


    audio_conf = dict(sample_rate=loaded_args.sample_rate,
                      window_size=loaded_args.window_size,
                      window_stride=loaded_args.window_stride,
                      window=loaded_args.window,
                      noise_dir=loaded_args.noise_dir,
                      noise_prob=loaded_args.noise_prob,
                      noise_levels=(loaded_args.noise_min, loaded_args.noise_max))

    test_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath_list=constant.args.test_manifest_list, label2id=label2id,
                                   normalize=True, augment=False)
    test_sampler = BucketingSampler(test_data, batch_size=constant.args.batch_size)
    test_loader = AudioDataLoader(test_data, num_workers=args.num_workers, batch_sampler=test_sampler)

    lm = None
    if constant.args.lm_rescoring:
        lm = LM(constant.args.lm_path)

    print(model)

    evaluate(model, test_loader, lm=lm)
