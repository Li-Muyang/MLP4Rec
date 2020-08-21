# @Time   : 2020/6/28
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/21, 2020/8/5, 2020/8/16
# @Author : Yupeng Hou, Xingyu Pan, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, panxy@ruc.edu.cn, chenyushuo@ruc.edu.cn

import os
import json
import copy
from collections import Counter
from logging import getLogger
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.sparse import coo_matrix
from ..utils import FeatureSource, FeatureType, ModelType


class Dataset(object):
    def __init__(self, config, saved_dataset=None):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()

        if saved_dataset is None:
            self._from_scratch(config)
        else:
            self._restore_saved_dataset(saved_dataset)

    def _from_scratch(self, config):
        self.dataset_path = config['data_path']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        if config['seq_len'] is not None:
            self.field2seqlen = config['seq_len']
        else:
            self.field2seqlen = {}

        self.inter_feat = None
        self.user_feat = None
        self.item_feat = None

        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        self.inter_feat, self.user_feat, self.item_feat = self._load_data(self.dataset_name, self.dataset_path)

        self.filter_by_inter_num(max_user_inter_num=config['max_user_inter_num'],
                                 min_user_inter_num=config['min_user_inter_num'],
                                 max_item_inter_num=config['max_item_inter_num'],
                                 min_item_inter_num=config['min_item_inter_num'])

        self.filter_by_field_value(lowest_val=config['lowest_val'], highest_val=config['highest_val'],
                                   equal_val=config['equal_val'], not_equal_val=config['not_equal_val'],
                                   drop=config['drop_filter_field'])

        self._set_label_by_threshold(self.config['threshold'])

        self._remap_ID_all()

        if self.config['fill_nan']:
            self._fill_nan()

        if self.config['normalize_field'] or self.config['normalize_all']:
            self._normalize(self.config['normalize_field'])

    def _restore_saved_dataset(self, saved_dataset):
        if (saved_dataset is None) or (not os.path.isdir(saved_dataset)):
            raise ValueError('filepath [{}] need to be a dir'.format(saved_dataset))

        with open(os.path.join(saved_dataset, 'basic-info.json')) as file:
            basic_info = json.load(file)

        for k in basic_info:
            setattr(self, k, basic_info[k])

        feats = ['inter', 'user', 'item']
        for name in feats:
            cur_file_name = os.path.join(saved_dataset, '{}.csv'.format(name))
            if os.path.isfile(cur_file_name):
                df = pd.read_csv(cur_file_name)
                setattr(self, '{}_feat'.format(name), df)

        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

    def _load_data(self, token, dataset_path):
        user_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'user'))
        if os.path.isfile(user_feat_path):
            user_feat = self._load_feat(user_feat_path, FeatureSource.USER)
        else:
            # TODO logging user feat not exist
            user_feat = None

        item_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'item'))
        if os.path.isfile(item_feat_path):
            item_feat = self._load_feat(item_feat_path, FeatureSource.ITEM)
        else:
            # TODO logging item feat not exist
            item_feat = None

        inter_feat_path = os.path.join(dataset_path, '{}.{}'.format(token, 'inter'))
        if not os.path.isfile(inter_feat_path):
            raise ValueError('File {} not exist'.format(inter_feat_path))

        inter_feat = self._load_feat(inter_feat_path, FeatureSource.INTERACTION)

        if user_feat is not None and self.uid_field is None:
            raise ValueError('uid_field must be exist if user_feat exist')

        if item_feat is not None and self.iid_field is None:
            raise ValueError('iid_field must be exist if item_feat exist')

        if self.uid_field in self.field2source:
            self.field2source[self.uid_field] = FeatureSource.USER_ID

        if self.iid_field in self.field2source:
            self.field2source[self.iid_field] = FeatureSource.ITEM_ID

        return inter_feat, user_feat, item_feat

    def _load_feat(self, filepath, source):
        str2ftype = {
            'token': FeatureType.TOKEN,
            'float': FeatureType.FLOAT,
            'token_seq': FeatureType.TOKEN_SEQ,
            'float_seq': FeatureType.FLOAT_SEQ
        }

        if self.config['load_col'] is None:
            load_col = None
        elif source.value not in self.config['load_col']:
            return None
        else:
            load_col = set(self.config['load_col'][source.value])
            if source in {FeatureSource.USER, FeatureSource.INTERACTION} and self.uid_field is not None:
                load_col.add(self.uid_field)
            if source in {FeatureSource.ITEM, FeatureSource.INTERACTION} and self.iid_field is not None:
                load_col.add(self.iid_field)
            if source == FeatureSource.INTERACTION and self.time_field is not None:
                load_col.add(self.time_field)

        if self.config['unload_col'] is not None and source.value in self.config['unload_col']:
            unload_col = set(self.config['unload_col'][source.value])
        else:
            unload_col = None

        if load_col is not None and unload_col is not None:
            raise ValueError('load_col [{}] and unload_col [{}] can not be setted the same time'.format(
                load_col, unload_col))

        df = pd.read_csv(filepath, delimiter=self.config['field_separator'])
        field_names = []
        columns = []
        remain_field = set()
        for field_type in df.columns:
            field, ftype = field_type.split(':')
            field_names.append(field)
            if ftype not in str2ftype:
                raise ValueError('Type {} from field {} is not supported'.format(ftype, field))
            ftype = str2ftype[ftype]
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            # TODO user_id & item_id bridge check
            # TODO user_id & item_id not be set in config
            # TODO inter __iter__ loading
            self.field2source[field] = source
            self.field2type[field] = ftype
            if not ftype.value.endswith('seq'):
                self.field2seqlen[field] = 1
            columns.append(field)
            remain_field.add(field)

        if len(columns) == 0:
            print('source', source)
            return None
        df.columns = field_names
        df = df[columns]

        seq_separator = self.config['seq_separator']
        def _token(df, field): pass
        def _float(df, field): pass
        def _token_seq(df, field): df[field] = [_.split(seq_separator) for _ in df[field].values]
        def _float_seq(df, field): df[field] = [list(map(float, _.split(seq_separator))) for _ in df[field].values]
        ftype2func = {
            FeatureType.TOKEN: _token,
            FeatureType.FLOAT: _float,
            FeatureType.TOKEN_SEQ: _token_seq,
            FeatureType.FLOAT_SEQ: _float_seq,
        }
        for field in remain_field:
            ftype = self.field2type[field]
            ftype2func[ftype](df, field)
            if field not in self.field2seqlen:
                self.field2seqlen[field] = max(map(len, df[field].values))
        return df

    def _fill_nan(self):
        most_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent', copy=False)
        aveg = SimpleImputer(missing_values=np.nan, strategy='mean', copy=False)

        for feat in [self.inter_feat, self.user_feat, self.item_feat]:
            if feat is not None:
                for field in self.field2type:
                    if field not in feat:
                        continue
                    ftype = self.field2type[field]
                    if ftype == FeatureType.TOKEN:
                        feat.loc[:,field] = most_freq.fit_transform(feat.loc[:,field].values.reshape(-1, 1))
                    elif ftype == FeatureType.FLOAT:
                        feat.loc[:,field] = aveg.fit_transform(feat.loc[:,field].values.reshape(-1, 1))
                    elif ftype.endswith('seq'):
                        self.logger.warning('feature [{}] (type: {}) probably has nan, while has not been filled.'.format(field, ftype))

    def _normalize(self, fields=None):
        if fields is None:
            fields = list(self.field2type)
        else:
            for field in fields:
                if field not in self.field2type:
                    raise ValueError('Field [{}] doesn\'t exist'.format(field))
                elif self.field2type[field] != FeatureType.FLOAT:
                    self.logger.warn('{} is not a FLOAT feat, which will not be normalized.'.format(field))
        for feat in [self.inter_feat, self.user_feat, self.item_feat]:
            if feat is None:
                continue
            for field in feat:
                if field in fields and self.field2type[field] == FeatureType.FLOAT:
                    lst = feat[field].values
                    mx, mn = max(lst), min(lst)
                    if mx == mn:
                        raise ValueError('All the same value in [{}] from [{}_feat]'.format(field, source))
                    feat[field] = (lst - mn) / (mx - mn)

    def filter_by_inter_num(self, max_user_inter_num=None, min_user_inter_num=None,
                            max_item_inter_num=None, min_item_inter_num=None):
        ban_users = self._get_illegal_ids_by_inter_num(source='user', max_num=max_user_inter_num,
                                                       min_num=min_user_inter_num)
        ban_items = self._get_illegal_ids_by_inter_num(source='item', max_num=max_item_inter_num,
                                                       min_num=min_item_inter_num)

        if len(ban_users) == 0 and len(ban_items) == 0:
            return

        if self.user_feat is not None:
            selected_user = ~self.user_feat[self.uid_field].isin(ban_users)
            self.user_feat = self.user_feat[selected_user].reset_index(drop=True)

        if self.item_feat is not None:
            selected_item = ~self.item_feat[self.iid_field].isin(ban_users)
            self.item_feat = self.item_feat[selected_item].reset_index(drop=True)

        selected_inter = pd.Series(True, index=self.inter_feat.index)
        if self.uid_field:
            selected_inter &= ~self.inter_feat[self.uid_field].isin(ban_users)
        if self.iid_field:
            selected_inter &= ~self.inter_feat[self.iid_field].isin(ban_items)
        self.inter_feat = self.inter_feat[selected_inter].reset_index(drop=True)

    def _get_illegal_ids_by_inter_num(self, source, max_num=None, min_num=None):
        if source not in {'user', 'item'}:
            raise ValueError('source [{}] should be user or item'.format(source))
        if max_num is None and min_num is None:
            return set()

        max_num = max_num or np.inf
        min_num = min_num or -1

        field_name = self.uid_field if source == 'user' else self.iid_field
        if field_name is None:
            return set()

        ids = self.inter_feat[field_name].values
        inter_num = Counter(ids)
        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}
        return ids

    def filter_by_field_value(self, lowest_val=None, highest_val=None,
                              equal_val=None, not_equal_val=None, drop=False):
        self._filter_by_field_value(lowest_val, lambda x, y: x >= y, drop)
        self._filter_by_field_value(highest_val, lambda x, y: x <= y, drop)
        self._filter_by_field_value(equal_val, lambda x, y: x == y, drop)
        self._filter_by_field_value(not_equal_val, lambda x, y: x != y, drop)

        if self.user_feat is not None:
            remained_uids = set(self.user_feat[self.uid_field].values)
        elif self.uid_field is not None:
            remained_uids = set(self.inter_feat[self.uid_field].values)

        if self.item_feat is not None:
            remained_iids = set(self.item_feat[self.iid_field].values)
        elif self.iid_field is not None:
            remained_iids = set(self.inter_feat[self.iid_field].values)

        remained_inter = pd.Series(True, index=self.inter_feat.index)
        if self.uid_field is not None:
            remained_inter &= self.inter_feat[self.uid_field].isin(remained_uids)
        if self.iid_field is not None:
            remained_inter &= self.inter_feat[self.iid_field].isin(remained_iids)
        self.inter_feat = self.inter_feat[remained_inter]

        for source in {'user', 'item', 'inter'}:
            feat = getattr(self, '{}_feat'.format(source))
            if feat is not None:
                feat.reset_index(drop=True, inplace=True)

    def _filter_by_field_value(self, val, cmp, drop=False):
        if val is None:
            return
        all_feats = []
        for source in ['inter', 'user', 'item']:
            cur_feat = getattr(self, '{}_feat'.format(source))
            if cur_feat is not None:
                all_feats.append([source, cur_feat])
        for field in val:
            if field not in self.field2type:
                raise ValueError('field [{}] not defined in dataset'.format(field))
            for source, cur_feat in all_feats:
                if field in cur_feat:
                    new_feat = cur_feat[cmp(cur_feat[field].values, val[field])]
                    setattr(self, '{}_feat'.format(source), new_feat)
            if drop:
                self._del_col(field)

    def _del_col(self, field):
        for source in ['inter', 'user', 'item']:
            cur_feat = getattr(self, '{}_feat'.format(source))
            if cur_feat is not None and field in cur_feat:
                setattr(self, '{}_feat'.format(source), cur_feat.drop(columns=field))
        for dct in [self.field2id_token, self.field2seqlen, self.field2source, self.field2type]:
            if field in dct:
                del dct[field]

    def _set_label_by_threshold(self, threshold):
        if threshold is None:
            return

        if len(threshold) != 1:
            raise ValueError('threshold length should be 1')

        self.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        for field, value in threshold.items():
            if field in self.inter_feat:
                self.inter_feat[self.label_field] = (self.inter_feat[field] >= value).astype(int)
            else:
                raise ValueError('field [{}] not in inter_feat'.format(field))
            self._del_col(field)

    def _remap_ID_all(self):
        for field in self.field2type:
            ftype = self.field2type[field]
            fsource = self.field2source[field]
            if ftype == FeatureType.TOKEN:
                self._remap_ID(fsource, field)
            elif ftype == FeatureType.TOKEN_SEQ:
                self._remap_ID_seq(fsource, field)

    def _remap_ID(self, source, field):
        feat_name = '{}_feat'.format(source.value.split('_')[0])
        feat = getattr(self, feat_name)
        if feat is None:
            feat = pd.DataFrame(columns=[field])
        if source in [FeatureSource.USER_ID, FeatureSource.ITEM_ID]:
            df = pd.concat([self.inter_feat[field], feat[field]])
            new_ids, mp = pd.factorize(df)
            split_point = [len(self.inter_feat[field])]
            self.inter_feat[field], feat[field] = np.split(new_ids + 1, split_point)
            self.field2id_token[field] = [None] + list(mp)
        elif source in [FeatureSource.USER, FeatureSource.ITEM, FeatureSource.INTERACTION]:
            new_ids, mp = pd.factorize(feat[field])
            feat[field] = new_ids + 1
            self.field2id_token[field] = [None] + list(mp)

    def _remap_ID_seq(self, source, field):
        if source in [FeatureSource.USER, FeatureSource.ITEM, FeatureSource.INTERACTION]:
            feat_name = '{}_feat'.format(source.value)
            df = getattr(self, feat_name)
            split_point = np.cumsum(df[field].agg(len))[:-1]
            new_ids, mp = pd.factorize(df[field].agg(np.concatenate))
            new_ids = np.split(new_ids + 1, split_point)
            df[field] = new_ids
            self.field2id_token[field] = [None] + list(mp)

    def num(self, field):
        if field not in self.field2type:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        if self.field2type[field] not in {FeatureType.TOKEN, FeatureType.TOKEN_SEQ}:
            return self.field2seqlen[field]
        else:
            return len(self.field2id_token[field])

    def fields(self, ftype=None):
        ftype = set(ftype) if ftype is not None else set(FeatureType)
        ret = []
        for field in self.field2type:
            tp = self.field2type[field]
            if tp in ftype:
                ret.append(field)
        return ret

    def set_field_property(self, field, field2type, field2source, field2seqlen):
        self.field2type[field] = field2type
        self.field2source[field] = field2source
        self.field2seqlen[field] = field2seqlen

    def copy_field_property(self, dest_field, source_field):
        self.field2type[dest_field] = self.field2type[source_field]
        self.field2source[dest_field] = self.field2source[source_field]
        self.field2seqlen[dest_field] = self.field2seqlen[source_field]

    @property
    def user_num(self):
        return self.num(self.uid_field)

    @property
    def item_num(self):
        return self.num(self.iid_field)

    @property
    def inter_num(self):
        return len(self.inter_feat)

    @property
    def avg_actions_of_users(self):
        return np.mean(self.inter_feat.groupby(self.uid_field).size())

    @property
    def avg_actions_of_items(self):
        return np.mean(self.inter_feat.groupby(self.iid_field).size())

    @property
    def sparsity(self):
        return 1 - self.inter_num / self.user_num / self.item_num

    @property
    def uid2items(self):
        self._check_field('uid_field', 'iid_field')
        uid2items = dict()
        columns = [self.uid_field, self.iid_field]
        for uid, iid in self.inter_feat[columns].values:
            if uid not in uid2items:
                uid2items[uid] = []
            uid2items[uid].append(iid)
        return pd.DataFrame(list(uid2items.items()), columns=columns)

    @property
    def uid2index(self):
        self._check_field('uid_field')
        self.sort(by=self.uid_field, ascending=True)
        uid_list = []
        start, end = dict(), dict()
        for i, uid in enumerate(self.inter_feat[self.uid_field].values):
            if uid not in start:
                uid_list.append(uid)
                start[uid] = i
            end[uid] = i
        index = [(uid, slice(start[uid], end[uid] + 1)) for uid in uid_list]
        uid2items_num = [end[uid] - start[uid] + 1 for uid in uid_list]
        return np.array(index), np.array(uid2items_num)

    def prepare_data_augmentation(self, max_item_list_len=None):
        if hasattr(self, 'uid_list'):
            return self.uid_list, self.item_list_index, self.target_index, self.item_list_length

        self._check_field('uid_field', 'time_field')
        if max_item_list_len is None:
            max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].values):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_list_length = np.array(item_list_length)
        return self.uid_list, self.item_list_index, self.target_index, self.item_list_length

    def _check_field(self, *field_names):
        for field_name in field_names:
            if getattr(self, field_name, None) is None:
                raise ValueError('{} isn\'t set'.format(field_name))

    def join(self, df):
        if self.user_feat is not None and self.uid_field in df:
            df = pd.merge(df, self.user_feat, on=self.uid_field, how='left', suffixes=('_inter', '_user'))
        if self.item_feat is not None and self.iid_field in df:
            df = pd.merge(df, self.item_feat, on=self.iid_field, how='left', suffixes=('_inter', '_item'))
        return df

    def __getitem__(self, index, join=True):
        df = self.inter_feat[index]
        return self.join(df) if join else df

    def __len__(self):
        return len(self.inter_feat)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = []
        if self.uid_field:
            info.extend(['The number of users: {}'.format(self.user_num),
                         'Average actions of users: {}'.format(self.avg_actions_of_users)])
        if self.iid_field:
            info.extend(['The number of items: {}'.format(self.item_num),
                         'Average actions of items: {}'.format(self.avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            info.append('The sparsity of the dataset: {}%'.format(self.sparsity * 100))
        info.append('Remain Fields: {}'.format(list(self.field2type)))
        return '\n'.join(info)

    # def __iter__(self):
    #     return self

    # TODO next func
    # def next(self):
    #     pass

    # TODO copy
    def copy(self, new_inter_feat):
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def _calcu_split_ids(self, tot, ratios):
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def split_by_ratio(self, ratios, group_by=None):
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        if group_by is None:
            tot_cnt = self.__len__()
            split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [range(start, end) for start, end in zip([0] + split_ids, split_ids + [tot_cnt])]
        else:
            grouped_inter_feat_index = self.inter_feat.groupby(by=group_by).groups.values()
            next_index = [[] for i in range(len(ratios))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start: end])

        next_df = [self.inter_feat.loc[index].reset_index(drop=True) for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def _split_index_by_leave_one_out(self, grouped_index, leave_one_num):
        next_index = [[] for i in range(leave_one_num + 1)]
        for index in grouped_index:
            index = list(index)
            tot_cnt = len(index)
            legal_leave_one_num = min(leave_one_num, tot_cnt - 1)
            pr = tot_cnt - legal_leave_one_num
            next_index[0].extend(index[:pr])
            for i in range(legal_leave_one_num):
                next_index[-legal_leave_one_num + i].append(index[pr])
                pr += 1
        return next_index

    def leave_one_out(self, group_by, model_type, leave_one_num=1):
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        if model_type == ModelType.SEQUENTIAL:
            self.prepare_data_augmentation()
            grouped_index = pd.DataFrame(self.uid_list).groupby(by=0).groups.values()
            next_index = self._split_index_by_leave_one_out(grouped_index, leave_one_num)
            next_ds = []
            for index in next_index:
                ds = copy.copy(self)
                for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length']:
                    setattr(ds, field, np.array(getattr(ds, field)[index]))
                next_ds.append(ds)
        else:
            grouped_inter_feat_index = self.inter_feat.groupby(by=group_by).groups.values()
            next_index = self._split_index_by_leave_one_out(grouped_inter_feat_index, leave_one_num)
            next_df = [self.inter_feat.loc[index].reset_index(drop=True) for index in next_index]
            next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def shuffle(self):
        self.inter_feat = self.inter_feat.sample(frac=1).reset_index(drop=True)

    def sort(self, by, ascending=True):
        self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True, ignore_index=True)

    # TODO
    def build(self, eval_setting, model_type):
        ordering_args = eval_setting.ordering_args
        if ordering_args['strategy'] == 'shuffle':
            self.shuffle()
        elif ordering_args['strategy'] == 'by':
            self.sort(by=ordering_args['field'], ascending=ordering_args['ascending'])

        group_field = eval_setting.group_field

        split_args = eval_setting.split_args
        if split_args['strategy'] == 'by_ratio':
            datasets = self.split_by_ratio(split_args['ratios'], group_by=group_field)
        elif split_args['strategy'] == 'by_value':
            raise NotImplementedError()
        elif split_args['strategy'] == 'loo':
            datasets = self.leave_one_out(group_by=group_field, model_type=model_type,
                                          leave_one_num=split_args['leave_one_num'])
        else:
            datasets = self

        return datasets

    def save(self, filepath):
        if (filepath is None) or (not os.path.isdir(filepath)):
            raise ValueError('filepath [{}] need to be a dir'.format(filepath))

        basic_info = {
            'field2type': self.field2type,
            'field2source': self.field2source,
            'field2id_token': self.field2id_token,
            'field2seqlen': self.field2seqlen
        }

        with open(os.path.join(filepath, 'basic-info.json'), 'w', encoding='utf-8') as file:
            json.dump(basic_info, file)

        feats = ['inter', 'user', 'item']
        for name in feats:
            df = getattr(self, '{}_feat'.format(name))
            if df is not None:
                df.to_csv(os.path.join(filepath, '{}.csv'.format(name)))

    def get_item_feature(self):
        if self.item_feat is None:
            self._check_field('iid_field')
            tot_item_cnt = self.num(self.iid_field)
            return pd.DataFrame({self.iid_field: np.arange(tot_item_cnt)})
        else:
            return self.item_feat

    def inter_matrix(self, form='coo', value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')

        uids = self.inter_feat[self.uid_field].values
        iids = self.inter_feat[self.iid_field].values
        if value_field is None:
            data = np.ones(len(self.inter_feat))
        else:
            if value_field not in self.field2source:
                raise ValueError('value_field [{}] not exist.'.format(value_field))
            if self.field2source[value_field] != FeatureSource.INTERACTION:
                raise ValueError('value_field [{}] can only be one of the interaction features'.format(value_field))
            data = self.inter_feat[value_field].values
        mat = coo_matrix((data, (uids, iids)), shape=(self.user_num, self.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('interaction matrix format [{}] has not been implemented.')