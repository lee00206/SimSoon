import os
import numpy as np
import pandas as pd
import random
import torch
import pprint
import warnings
from trainer import Trainer
from tokenization import tokenize_enumerated_smiles, tokenize_smiles_labels
from config import get_Config

warnings.filterwarnings('ignore')

def set_seed(args):
    np.seterr(all="ignore")
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize_all(args):
    # tokenize pretrain dataset
    if os.path.exists('data/embedding/pubchem_part.pth'):
        pass
    else:
        tokenize_enumerated_smiles(args)

    # tokenize downstream datasets
    datasets = ['tox21', 'bbbp', 'clintox', 'hiv', 'bace', 'sider', 'esol', 'freesolv', 'lipophilicity']
    for data in datasets:
        path = 'data/prediction/' + data + '.pth'
        if os.path.exists(path):
            pass
        elif data == 'hiv' or data == 'bace':
            tokenize_smiles_labels(args, data, split='scaffold')
        elif data == 'tox21':
            tokenize_smiles_labels(args, data, split='random', num_classes=12)
        elif data == 'sider':
            tokenize_smiles_labels(args, data, split='random', num_classes=27)
        elif data == 'clintox':
            tokenize_smiles_labels(args, data, split='random', num_classes=2)
        else:  # esol, freesolv, lipophilicity, bbbp
            tokenize_smiles_labels(args, data, split='random')


def main(args):
    print('<---------------- Training params ---------------->')
    pprint.pprint(args)

    # Random seed
    set_seed(args)

    # tokenize
    tokenize_all(args)

    # train
    if args.task == 'pretraining':
        trainer = Trainer(args, data='pubchem_part')
        trainer.train()

    elif args.task == 'downstream':
        trainer = Trainer(args, data=args.data)
        trainer.pre_train()
        trainer.test()
    elif args.task == 'inference':
        trainer = Trainer(args, data=args.data)
        trainer.test()


if __name__ == '__main__':
    args = get_Config()
    main(args)








