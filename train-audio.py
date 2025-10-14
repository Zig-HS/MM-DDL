# python imports
import argparse
from pprint import pprint
import os
import time
import datetime
import sys
import yaml

# torch imports
import torch
import torch.nn as nn
from torchinfo import summary

# our code
from libs.core import load_config, load_config_without_merge
from libs.utils import (fix_random_seed,make_optimizer,make_scheduler, ModelEma,train_one_epoch,save_checkpoint)
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch

# for visualization
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 自定义 Representer，强制某些字段使用流式格式
def flow_style_dict(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)


def main(args):

    """1. setup parameters / folders"""

    """ 读取配置文件 """
    args.start_epoch = 0
    if os.path.isfile(args.config):
        # cfg = load_config(args.config)
        cfg = load_config_without_merge(args.config)
    else:
        raise ValueError("Config file does not exist.")
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    cfg["dataset"]["devices"] = cfg["devices"]
    cfg["dataset"]["num_workers"] = cfg["loader"]["num_workers"]


    """ 设置实验保存路径 """
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)


    """ 初始化tensorboard,设置rng_generator """
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    # re-scale learning rate / # workers based on number of GPUs


    """ 加载数据集,设置Dataloder """
    train_dataset = make_dataset(cfg['dataset_name'], True, **cfg['dataset'])
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    # data loaders
    train_loader = make_data_loader(
        dataset=train_dataset, 
        is_training=True, 
        generator=rng_generator, 
        batch_size=cfg['loader']['batch_size'], 
        num_workers=cfg['loader']['num_workers'], 
        shuffle=True,
    )

    """ 加载模型,设置DP多卡训练和EMA """
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """ 恢复模型训练 """
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            device = torch.device(cfg['devices'][0])
            checkpoint = torch.load(args.resume, map_location = device)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return


    """ 设置完成,保存配置文件 """
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()
    pprint(cfg)


    """ 开始模型训练与验证 """
    print("\nStart training model {:s} ...".format(cfg['model_name']))
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        # save ckpt once in a while
        if (((epoch + 1) == max_epochs) or((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )
    # wrap up
    tb_writer.close()
    print("All done!")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR', help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int, help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str, help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to a checkpoint (default: none)')

    command_line_args = ["configs_train/ijcai25audio-wavLM.yaml", "--output", "new", "--ckpt-freq", "1",]
    args = parser.parse_args(command_line_args)
    main(args)
