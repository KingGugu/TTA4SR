# -*- coding: utf-8 -*-


import copy
import torch
import random
from utils import neg_sample
from torch.utils.data import Dataset


class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length
        if self.args.TTA_type == 'TMask-R':
            self.aug = TMask_R(self.args.sigma)
        if self.args.TTA_type == 'TMask-B':
            self.aug = TMask_B(self.args.sigma)

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )
        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        if self.data_type == "train":

            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)


class SASRecDataset_TTA(Dataset):

    def __init__(self, args, user_seq):
        self.args = args
        self.user_seq = user_seq
        self.max_len = args.max_seq_length
        if self.args.TTA_type == 'TMask-R':
            self.aug = TMask_R(self.args.sigma)
        if self.args.TTA_type == 'TMask-B':
            self.aug = TMask_B(self.args.sigma)

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )
        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        input_ids = items[:-1]
        target_pos = items[1:]
        answer = [items[-1]]
        original_seq = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
        augmented_seqs = []
        for i in range(self.args.m):
            augmented_input_ids = self.aug(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids
            augmented_input_ids = augmented_input_ids[-self.max_len:]
            assert len(augmented_input_ids) == self.max_len
            cur_tensors = (torch.tensor(augmented_input_ids, dtype=torch.long))
            augmented_seqs.append(cur_tensors)
        return original_seq, augmented_seqs

    def __len__(self):
        return len(self.user_seq)


class TMask_B(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sequence):
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.sigma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class TMask_R(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.sigma * len(copied_sequence))
        remove_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        copied_sequence = [copied_sequence[i] for i in range(len(copied_sequence)) if i not in remove_idx]
        return copied_sequence
