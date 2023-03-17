import argparse
import random
import numpy as np
import torch
import os.path as osp
from configs import CFG_default
from utils import setup_logger
from trainer import FreeMatchTrainer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_configs(args, cfg):
    
    print('*' * 20)
    print('Arguments to be overwritten')
    print('*' * 20)
    
    argkeys = list(args.__dict__.keys())
    argkeys.sort()
    for key in argkeys:
        if args.__dict__[key] is not None:
            print(key)
    
    print('*' * 20)
    print('Config File Arguments..')
    print('*' * 20)
    print(cfg)

def overwrite_config(cfg, args):
    
    if args.run_name:
        cfg.RUN_NAME = args.run_name
    
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    if args.log_dir:
        cfg.LOG_DIR = args.log_dir
        
    if args.tb_dir:
        cfg.TB_DIR = args.tb_dir
    
    if args.resume_checkpoint:
        cfg.RESUME = args.resume_checkpoint
    
    if args.cont_train:
        cfg.CONT_TRAIN = True
    
    if args.train_batch_size:
        cfg.DATASET.TRAIN_BATCH_SIZE = args.train_batch_size
    
    if args.test_batch_size:
        cfg.TEST_BATCH_SIZE = args.test_batch_size
    
    if args.seed:
        cfg.SEED = args.SEED

def setup_config(args):
    
    cfg = CFG_default
    cfg.merge_from_file(args.config_file)
    overwrite_config(cfg, args)
    cfg.freeze()
    return cfg

def main(args):
    
    cfg = setup_config(args) 
    if cfg.SEED >= 0:
        print('Training with seed initialziation..')
        set_random_seed(cfg.SEED)
    
    setup_logger(osp.join(cfg.LOG_DIR, cfg.RUN_NAME))

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    print_configs(args, cfg)
    trainer = FreeMatchTrainer(cfg)
    trainer.train()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config-file', default=None, type=str, help='Path to the config file of the experiment')
    
    parser.add_argument('--run-name', type=str, default=None, help='Run name of the experiment')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory to save the logs')
    parser.add_argument('--tb-dir', type=str, default=None, help='Directory to save tensorboard logs')
    parser.add_argument('--resume-checkpoint', type=str, default=None, help='Resume path of the checkpoint')
    parser.add_argument('--cont-train', action='store_true', help='Flag to continue training')
    
    parser.add_argument('--train-batch-size', type=int, default=None, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=None, help='Testing batch size')
    parser.add_argument('--seed', type=int, default=None, help='Seed')
    
    args = parser.parse_args()
    
    main(args=args)
