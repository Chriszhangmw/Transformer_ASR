import json
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from trainer.asr.trainer import Trainer

from utils import constant
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.functions import save_model, load_model, init_transformer_model, init_optimizer
import logging
import datetime
import sys
import os
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

if __name__ == '__main__':
    args = constant.args
    print("="*50)
    print("THE EXPERIMENT LOG IS SAVED IN: " + "log/" + args.name)
    print("TRAINING MANIFEST: ", args.train_manifest_list)
    print("VALID MANIFEST: ", args.valid_manifest_list)
    print("TEST MANIFEST: ", args.test_manifest_list)
    print("="*50)

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    print(audio_conf)

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    # add PAD_CHAR, SOS_CHAR, EOS_CHAR
    labels = constant.PAD_CHAR + constant.SOS_CHAR + constant.EOS_CHAR + labels
    label2id, id2label = {}, {}
    count = 0
    for i in range(len(labels)):
        if labels[i] not in label2id:
            label2id[labels[i]] = count
            id2label[count] = labels[i]
            count += 1
        else:
            print("multiple label: ", labels[i])

    # label2id = dict([(labels[i], i) for i in range(len(labels))])
    # id2label = dict([(i, labels[i]) for i in range(len(labels))])

    train_data = SpectrogramDataset(audio_conf, args.train_manifest_list,
                                    label2id, normalize=True, augment=args.augment)
    train_sampler = BucketingSampler(train_data, batch_size=args.batch_size)


    train_loader = AudioDataLoader(
        train_data, num_workers=args.num_workers, batch_sampler=train_sampler)
    #,pin_memory=True

    # valid_loader_list, test_loader_list = [], []
    # for i in range(len(args.valid_manifest_list)):
    valid_data = SpectrogramDataset(audio_conf, manifest_filepath_list=args.valid_manifest_list, label2id=label2id,
                                    normalize=True, augment=False)
    valid_sampler = BucketingSampler(valid_data, batch_size=args.batch_size)

    valid_loader = AudioDataLoader(
        valid_data, num_workers=args.num_workers, batch_sampler=valid_sampler)

    # test_data = SpectrogramDataset(audio_conf, manifest_filepath_list=args.test_manifest_list, label2id=label2id,
    #                             normalize=True, augment=False)
    # test_loader = AudioDataLoader(test_data, num_workers=args.num_workers)
    start_epoch = 0
    metrics = None
    loaded_args = None
    model = init_transformer_model(constant.args, label2id, id2label)
    opt = init_optimizer(constant.args, model, "noam")
    loss_type = args.loss

    device = torch.device("cuda:3")
    torch.cuda.set_device(device)
    model.to(device)

    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # batch_szie = 15
    # gpu0_bsz = 3
    # acc_grad = 2
    # model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()

    # torch.distributed.init_process_group(backend="nccl")
    # '''python - m  torch.distributed.launch  train.py'''
    # model = model.cuda()
    # model = nn.parallel.DistributedDataParallel(model)

    # model = nn.DataParallel(model)
    # model = model.cuda()


    num_epochs = args.epochs
    trainer = Trainer()
    trainer.train(model, train_loader, train_sampler, valid_loader, valid_sampler,opt, loss_type, start_epoch, num_epochs, label2id, id2label, metrics)
