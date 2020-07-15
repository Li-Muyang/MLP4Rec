# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : dataloader.py

import operator
from functools import reduce
import pandas as pd
import numpy as np
import torch
from sampler import Sampler
from .interaction import Interaction

class AbstractDataLoader(object):
    def __init__(self, config, dataset, batch_size):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = Sampler(config, dataset)

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError('Method [next] should be implemented.')

class GeneralDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, batch_size, pairwise=False, shuffle=False, real_time_neg_sampling=True, neg_sample_to=None, neg_sample_by=None):
        super(GeneralDataLoader, self).__init__(config, dataset, batch_size)

        self.pairwise = pairwise
        self.shuffle = shuffle
        self.real_time_neg_sampling = real_time_neg_sampling
        self.pr = 0

        self.neg_sample_to = neg_sample_to
        self.neg_sample_by = neg_sample_by

        if neg_sample_by is not None and neg_sample_to is not None:
            raise ValueError('neg_sample_to and neg_sample_by cannot be given value the same time')

        if not real_time_neg_sampling:
            self._pre_neg_sampling()

        if self.shuffle:
            self.dataset.shuffle()

    def __next__(self):
        if self.pr >= len(self.dataset):
            self.pr = 0
            raise StopIteration()
        cur_data = self.dataset[self.pr : self.pr+self.batch_size-1]
        self.pr += self.batch_size
        # TODO real time negative sampling
        if self.real_time_neg_sampling:
            if not self.pairwise:
                raise ValueError('real time neg sampling only support pairwise dataloader')
            pass
        cur_data = cur_data.to_dict(orient='list')
        for k in cur_data:
            ftype = self.dataset.field2type[k]
            # TODO seq support
            if ftype == 'token':
                cur_data[k] = torch.LongTensor(cur_data[k])
            elif ftype == 'float':
                cur_data[k] = torch.FloatTensor(cur_data[k])
            elif ftype == 'token_seq':
                raise NotImplementedError()
            elif ftype == 'float_seq':
                raise NotImplementedError()
            else:
                raise ValueError('Illegal ftype [{}]'.format(ftype))
        return Interaction(cur_data)

    def _pre_neg_sampling(self):
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        if self.neg_sample_by is not None:
            uids = self.dataset.inter_feat[self.config['USER_ID_FIELD']].to_list()
            # iids = self.dataset.inter_feat[self.config['ITEM_ID_FIELD']].to_list()
            # if self.neg_sample_by == 1:
            neg_iids = []
            for uid in uids:
                neg_iids.append(self.sampler.sample_by_user_id(uid, self.neg_sample_by))
            if self.pairwise:
                if self.neg_sample_by != 1:
                    raise ValueError('Pairwise dataloader can only neg sample by 1')
                neg_prefix = self.config['NEG_PREFIX']
                neg_item_id = neg_prefix + iid_field
                neg_iids = [_[0] for _ in neg_iids]
                self.dataset.inter_feat.insert(len(self.dataset.inter_feat.columns), neg_item_id, neg_iids)
                self.dataset.field2type[neg_item_id] = 'token'
                self.dataset.field2source[neg_item_id] = 'item_id'
                # TODO item_feat join
                if self.dataset.item_feat is not None:
                    pass
            else:   # Point-Wise
                neg_iids = list(map(list, zip(*neg_iids)))
                neg_iids = reduce(operator.add, neg_iids)
                neg_iids = self.dataset.inter_feat[iid_field].to_list() + neg_iids

                pos_inter_num = len(self.dataset.inter_feat)

                new_df = pd.concat([self.dataset.inter_feat] * (1 + self.neg_sample_by), ignore_index=True)
                new_df[iid_field] = neg_iids

                label_field = self.config['LABEL_FIELD']
                labels = pos_inter_num * [1] + self.neg_sample_by * pos_inter_num * [0]
                new_df[label_field] = labels

                self.dataset.inter_feat = new_df
        # TODO
        elif self.neg_sample_to is not None:
            if self.pairwise:
                raise ValueError('pairwise dataloader cannot neg sample to')
            user_num_in_one_batch = self.batch_size // self.neg_sample_to
            self.batch_size = (user_num_in_one_batch + 1) * self.neg_sample_to

            label_field = self.config['LABEL_FIELD']
            self.dataset.field2type[label_field] = 'float'
            self.dataset.field2source[label_field] = 'inter'
            new_inter = {
                uid_field: [],
                iid_field: [],
                label_field: []
            }

            uids = self.dataset.inter_feat[uid_field].to_list()
            iids = self.dataset.inter_feat[iid_field].to_list()
            uid2itemlist = {}
            for i in range(len(uids)):
                uid = uids[i]
                iid = iids[i]
                if uid not in uid2itemlist:
                    uid2itemlist[uid] = []
                uid2itemlist[uid].append(iid)
            for uid in uid2itemlist:
                pos_num = len(uid2itemlist[uid])
                if pos_num >= self.neg_sample_to:
                    uid2itemlist[uid] = uid2itemlist[uid][:self.neg_sample_to-1]
                    pos_num = self.neg_sample_to - 1
                neg_item_id = self.sampler.sample_by_user_id(uid, self.neg_sample_to - pos_num)
                for iid in uid2itemlist[uid]:
                    new_inter[uid_field].append(uid)
                    new_inter[iid_field].append(iid)
                    new_inter[label_field].append(1)
                for iid in neg_item_id:
                    new_inter[uid_field].append(uid)
                    new_inter[iid_field].append(iid)
                    new_inter[label_field].append(0)
            self.dataset.inter_feat = pd.DataFrame(new_inter)