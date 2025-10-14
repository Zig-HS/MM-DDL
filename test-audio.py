# python imports
import argparse

import os
import time
import datetime
import glob
import sys
import yaml
import numpy as np
import json
from tqdm import tqdm
from pprint import pprint


# torch imports
import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config, load_config_without_merge
from libs.utils import (fix_random_seed,make_optimizer,make_scheduler, ModelEma,train_one_epoch,save_checkpoint)
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed, MetricCollector

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 自定义 Representer，强制某些字段使用流式格式
def flow_style_dict(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)


def main(args):

    """ 读取配置文件 """
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config_without_merge(args.config)
    else:
        raise ValueError("Config file does not exist.")
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    cfg["dataset"]["devices"] = cfg["devices"]
    cfg["dataset"]["num_workers"] = cfg["loader"]["num_workers"]
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk


    """ 检查权重文件 """
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    """ 设置随机数 """
    _ = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    """ 加载数据集,设置Dataloder """
    val_dataset = make_dataset(cfg['dataset_name'], is_training=False, **cfg['dataset'])
    val_loader = make_data_loader(dataset=val_dataset, is_training=False, generator=None, batch_size=1, num_workers=cfg['loader']['num_workers'], shuffle=True)


    """ 加载模型,设置DP多卡训练和EMA """
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])


    """ 恢复模型训练 """
    print("=> loading checkpoint '{}'".format(ckpt_file))
    device = torch.device(cfg['devices'][0])  # 将字符串转为 torch.device 对象
    checkpoint = torch.load(
        ckpt_file,
        map_location=device,
    )
    # load ema model instead
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    """ 开始模型验证 """
    os.makedirs(f"./results/{args.name}/", exist_ok=True)
    model.eval()
    start = time.time()
    count = 0
    json_data = {}
    for iter_idx, audio_list in enumerate(tqdm(val_loader, desc="Evaluating")):
        count += 1
        # if count > 10:
        #     break
        with torch.no_grad():
            outputs = model(audio_list)
        for i, data_dict in enumerate(audio_list): # video_list是列表
            # 合并视频和音频的segments和scores
            segments = outputs[i]['segments'].cpu().numpy()
            scores = outputs[i]['scores'].cpu().numpy()

            max_score = 0 if len(scores) == 0 else max(scores)
            with open(f"./results/{args.name}/prediction.txt", "a") as f:
                f.write(f"{outputs[i]['video_id']},{max_score:.4f}\n")
    
            # 构建segments列表
            segments_list = []
            for seg, score in zip(segments, scores):
                segments_list.append([
                    round(float(score), 4),     # 保留4位小数
                    round(float(seg[0]), 3),   # 保留3位小数
                    round(float(seg[1]), 3)    # 保留3位小数
                ])
                # segments_list.append([float(score), float(seg[0]), float(seg[1])])
            json_data[outputs[i]['video_id']] = segments_list

    # 写入JSON文件
    with open(f"./results/{args.name}/prediction.json", "w") as f:
        json.dump(json_data, f, indent=4)

    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR', help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR', help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1, help='checkpoint epoch')
    parser.add_argument('-name', type=str, default='test', help='name of test')
    parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true', help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10 iterations)')
    command_line_args = ["configs_test/ijcai25audio-wavLM.yaml", "ckpt/ijcai25audio-wavLM", "-epoch", "3", "-name", 'audio-wavLM-epoch3-nms0' ]
   
    args = parser.parse_args(command_line_args)
    main(args)
