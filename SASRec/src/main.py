# -*- coding: utf-8 -*-


import os
import torch
import argparse
import numpy as np
from trainers import SASRecTrainer
from models import SASRecModel
from datasets import SASRecDataset, SASRecDataset_TTA
from utils import EarlyStopping, get_user_seqs, check_path, set_seed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    with open(args.log_file, 'a') as f:
        for arg in vars(args):
            info = f"{arg:<30} : {getattr(args, arg):>35}"
            print(info)
            f.write(info + '\n')


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', default=1, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--no_cuda", action="store_true")

    # model args
    parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stop patience")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--star_test", default=0, type=int)

    # TTA args
    parser.add_argument("--a", type=float, default=1, help="uniform distribution for TNoise")
    parser.add_argument("--b", type=float, default=0.5, help="uniform distribution for TNoise")
    parser.add_argument("--sigma", type=float, default=0.6, help="mask ratio for TMask")
    parser.add_argument("--m", type=int, default=10, help="number of augmented sequences (TMask)")
    parser.add_argument("--TTA", action='store_true')
    parser.add_argument("--TTA_type", default='TMask-R', type=str)
    parser.add_argument("--input_num", type=int, default=7, help="augmentation times for TNoise")
    parser.add_argument("--output_num", type=int, default=5, help="augmentation times for TNoise")

    args = parser.parse_args()
    assert args.TTA_type in ['TMask-R', 'TMask-B', 'TNoise']
    set_seed(args.seed)
    check_path(args.output_dir)

    assert args.data_name in ['Yelp', 'Beauty', 'Sports', 'Home']

    args.a_1 = args.a
    args.b_1 = args.b
    if args.data_name == 'Yelp':
        args.a_1 = 0.5
        args.b_1 = 0.1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file, args)
    args.item_size = max_item + 2

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    show_args_info(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # training data for node classification
    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    test_dataset_tta = SASRecDataset_TTA(args, user_seq)
    test_sampler_tta = SequentialSampler(test_dataset_tta)
    test_dataloader_tta = DataLoader(test_dataset_tta, sampler=test_sampler_tta, batch_size=args.batch_size)

    model = SASRecModel(args=args)
    trainer = SASRecTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.TTA:
        trainer.args.train_matrix = test_rating_matrix
        pretrained = os.path.join("./pretrained_" + args.model_name, args.model_name + '-' + args.data_name + '.pt')
        trainer.load(pretrained)
        print(f'Load model from {pretrained} for TTA test!')
        print('\nPerformance of pretrained original model:')
        trainer.test(0)
        print('\nPerformance with TTA:')
        if args.TTA_type == 'TMask-R' or args.TTA_type == 'TMask-B':
            trainer.test_aug(test_dataloader_tta)
        else:
            trainer.test_aug(test_dataloader)
        exit()

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            if epoch > args.star_test:
                scores, _ = trainer.valid(epoch)
                early_stopping(np.array(scores[-1:]), trainer.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        trainer.args.train_matrix = test_rating_matrix
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)

    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')


main()
