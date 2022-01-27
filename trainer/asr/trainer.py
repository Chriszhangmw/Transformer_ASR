import time
from utils.encoding import DataParallelModel, DataParallelCriterion
import numpy as np
from tqdm import tqdm
from utils import constant
from utils.functions import save_model
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer,calc_f1,bleu,distinct
from torch.autograd import Variable
import torch
# from warpctc_pytorch import CTCLoss
import torch.nn.functional as F
import logging
from torch.nn import CTCLoss
import sys
# ctcloss = CTCLoss(size_average=True)
from time import *
import torch.distributed as dist
def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor

class Trainer():
    """
    Trainer class
    """
    def __init__(self):
        logging.info("Trainer is initialized")

    def train(self, model, train_loader, train_sampler,valid_loader,
              valid_sampler, opt, loss_type, start_epoch, num_epochs, label2id, id2label, last_metrics=None):
        history = []
        best_valid_loss = 1000000000 if last_metrics is None else last_metrics['valid_loss']

        for epoch in range(start_epoch, num_epochs):
            sys.stdout.flush()
            total_loss, total_cer, total_wer, total_char, total_word = 0, 0, 0, 0, 0
            start_iter = 0
            print("TRAIN")
            model.train()
            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            for i, (data) in enumerate(pbar, start=start_iter):
                if data:
                    src, tgt, src_percentages, src_lengths, tgt_lengths,targets2 = data
                    # if constant.USE_CUDA:
                    src = src.cuda()
                    tgt = tgt.cuda()
                    src_lengths = src_lengths.cuda()
                    opt.zero_grad()
                    # b1 = time()
                    pred,gold, hyp_seq, gold_seq,en1,en2 = model(src,src_lengths,tgt, verbose=False)
                    # b2 = time()
                    # print('model time:',b2-b1)
                    try: # handle case for CTC
                        strs_gold, strs_hyps = [], []
                        for ut_gold in gold_seq:
                            str_gold = ""
                            for x in ut_gold:
                                if int(x) == constant.PAD_TOKEN:
                                    break
                                str_gold = str_gold + id2label[int(x)]
                            strs_gold.append(str_gold)
                        for ut_hyp in hyp_seq:
                            str_hyp = ""
                            for x in ut_hyp:
                                if int(x) == constant.PAD_TOKEN:
                                    break
                                str_hyp = str_hyp + id2label[int(x)]
                            strs_hyps.append(str_hyp)
                    except Exception as e:
                        print(e)
                        print("NaN predictions")
                        continue
                    # b3 = time()
                    loss = calculate_metrics(pred,gold, targets2,tgt_lengths,en1,en2)
                    # loss = all_reduce_tensor(loss, world_size=2)
                    # print('Loss :',loss)
                    # b4 = time()
                    # print('loss time:', b4 - b3)

                    if loss.item() == float('Inf'):
                        print("Found infinity loss, masking")
                        loss = torch.where(loss != loss, torch.zeros_like(loss), loss) # NaN masking
                        continue

                    for j in range(len(strs_hyps)):
                        strs_hyps[j] = strs_hyps[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                        strs_gold[j] = strs_gold[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                        cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_gold[j].replace(' ', ''))
                        wer = calculate_wer(strs_hyps[j], strs_gold[j])
                        total_cer += cer
                        total_wer += wer
                        total_char += len(strs_gold[j].replace(' ', ''))
                        total_word += len(strs_gold[j].split(" "))
                    b5 = time()
                    loss.backward()
                    b6 = time()
                    print('loss2 time:', b6 - b5)
                    if constant.args.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), constant.args.max_norm)
                    opt.step()
                    total_loss += loss.item()

                    pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}% LR:{:.7f}".format(
                        (epoch+1), total_loss/(i+1), total_cer*100/total_char,total_wer*100/total_word, opt._rate))
                print("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}% LR:{:.7f}".format(
                    (epoch+1), total_loss/(len(train_loader)), total_cer*100/total_char, total_wer*100/total_word,opt._rate))

            # evaluate
            print("VALID")
            total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word = 0, 0, 0, 0, 0
            model.eval()
            F1data = []
            predictions = []
            valid_pbar = tqdm(iter(valid_loader), leave=True, total=len(valid_loader))
            for i, (data) in enumerate(valid_pbar):
                src, tgt, src_percentages, src_lengths, tgt_lengths,targets2 = data
                if constant.USE_CUDA:
                    src = src.cuda()
                    tgt = tgt.cuda()

                pred,gold, hyp_seq, gold_seq,en1,en2 = model(src, src_lengths, tgt, verbose=False)

                loss = calculate_metrics(pred, gold, targets2,tgt_lengths,en1,en2)

                if loss.item() == float('Inf'):
                    # logging.info("Found infinity loss, masking")
                    loss = torch.where(loss != loss, torch.zeros_like(loss), loss)  # NaN masking
                    continue

                try:  # handle case for CTC
                    strs_gold, strs_hyps = [], []
                    for ut_gold in gold_seq:
                        str_gold = ""
                        for x in ut_gold:
                            if int(x) == constant.PAD_TOKEN:
                                break
                            str_gold = str_gold + id2label[int(x)]
                        strs_gold.append(str_gold)
                    for ut_hyp in hyp_seq:
                        str_hyp = ""
                        for x in ut_hyp:
                            if int(x) == constant.PAD_TOKEN:
                                break
                            str_hyp = str_hyp + id2label[int(x)]
                        strs_hyps.append(str_hyp)
                except Exception as e:
                    print(e)
                    continue

                for j in range(len(strs_hyps)):
                    strs_hyps[j] = strs_hyps[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                    strs_gold[j] = strs_gold[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                    cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_gold[j].replace(' ', ''))
                    F1data.append((strs_hyps[j].replace(' ', ''), strs_gold[j].replace(' ', '')))
                    predictions.append(strs_gold[j].replace(' ', ''))
                    wer = calculate_wer(strs_hyps[j], strs_gold[j])
                    total_valid_cer += cer
                    total_valid_wer += wer
                    total_valid_char += len(strs_gold[j].replace(' ', ''))
                    total_valid_word += len(strs_gold[j].split(" "))

                total_valid_loss += loss.item()
                valid_pbar.set_description("VALID LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(total_valid_loss / (i + 1),
                                          total_valid_cer * 100 / total_valid_char, total_valid_wer *100/total_valid_word))
            f1 = calc_f1(F1data)
            bleu_1, bleu_2 = bleu(F1data)
            unigrams_distinct, bigrams_distinct = distinct(predictions)
            print(
                "VALID LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}% f1:{:.2f}% bleu_1:{:.2f}% bleu_2:{:.2f}% unigrams_distinct:{:.2f}% bigrams_distinct:{:.2f}%".format(
                    total_valid_loss / (len(valid_loader)),
                    total_valid_cer * 100 / total_valid_char, total_valid_wer * 100 / total_valid_word, f1, bleu_1,
                    bleu_2, unigrams_distinct, bigrams_distinct))

            metrics = {}
            metrics["train_loss"] = total_loss / len(train_loader)
            metrics["valid_loss"] = total_valid_loss / (len(valid_loader))
            metrics["train_cer"] = total_cer
            metrics["train_wer"] = total_wer
            metrics["valid_cer"] = total_valid_cer
            metrics["valid_wer"] = total_valid_wer
            metrics["history"] = history
            history.append(metrics)

            if best_valid_loss > total_valid_loss/len(valid_loader):
                best_valid_loss = total_valid_loss/len(valid_loader)
                save_model(model, (epoch+1), opt, metrics,
                        label2id, id2label, best_model=True)

            if constant.args.shuffle:
                print("SHUFFLE")
                train_sampler.shuffle(epoch)