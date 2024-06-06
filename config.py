import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.1)
    parser.add_argument("--batch_size", help="batch size", default=600, type=int)
    parser.add_argument("--lr", help="learning rate", default=1, type=float)
    parser.add_argument("--nworkers", help="# workers", default=100, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=200, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=41, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=28, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='no', type=str,
                        choices=['no', 'mean_attack', 'backdoor', 'partial_trim', 'full_trim', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda'])
    parser.add_argument("--aggregation", help="aggregation rule", default='simple_mean', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])
    parser.add_argument("--home_path", help="home path", default='', type=str)
    return parser.parse_args()
