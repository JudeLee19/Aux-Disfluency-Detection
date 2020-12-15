import numpy as np
import torch
import os
from torch.utils import data


class InputExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, pos_tags, labels, ner_labels, pos_labels):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels_a/labels_b: (Optional) string. The label seqence of the text_a/text_b. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        self.pos_tags = pos_tags

        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels
        self.ner_labels = ner_labels
        self.pos_labels = pos_labels


class InputFeatures(object):
    """A single set of features of data.
    result of convert_examples_to_features(InputExample)
    """

    def __init__(self, input_ids, input_mask, segment_ids,  predict_mask,
                 label_ids, ner_label_ids=None, pos_label_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids
        self.ner_label_ids = ner_label_ids
        self.pos_label_ids = pos_label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """
        Reads a BIO data.
        """
        with open(input_file) as f:
            # out_lines = []
            out_lists = []
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                words = []
                ner_labels = []
                pos_tags = []
                dis_tags = []
                for line in entry.splitlines():
                    pieces = line.strip().split()
                    if len(pieces) < 1:
                        continue
                    word = pieces[0]

                    pos_label = pieces[1]
                    if '^' in pos_label:
                        pos_label = pos_label.split('^')[1]
                    if '|' in pos_label:
                        pos_label = pos_label.split('|')[0]
                    pos_label = pos_label.replace('$', '')
                    pos_label = pos_label.strip()

                    words.append(word)
                    pos_tags.append(pos_label)
                    dis_tags.append(pieces[-2])
                    ner_labels.append(pieces[-1])
                out_lists.append([words, pos_tags, dis_tags, ner_labels])
        return out_lists


class SwbdDataProcessor(DataProcessor):

    def __init__(self, tagger_type='conll'):

        if tagger_type == 'conll':
            # {'LOC', 'MISC', 'O', 'ORG', 'PER'}
            self._ner_label_types = [ 'X', '[CLS]', '[SEP]', 'O', 'LOC', 'MISC', 'ORG', 'PER']
            self._label_types = ['X', '[CLS]', '[SEP]', 'O', 'B-1']
            self._pos_label_types = ['X', '[CLS]', '[SEP]', 'O', '``', 'jjs', 'md', 'nns', 'uh', "''", 'rb', 'vbz',
                                     'pdt', 'ex', 'vbg', 'wrb', ':', 'sym', 'rp', 'prp', 'nnp', 'vb', 'vbn', 'pos',
                                     'to', 'wdt', 'fw', 'in', 'ls', 'dt', 'hvs', 'bes', 'cc', 'vbd', 'rbs', 'rbr', 'nn',
                                     'wp', 'gw', 'nnps', 'cd', 'jjr', '.', 'vbp', 'jj', ',', 'xx']

        elif tagger_type == 'ontonotes':
            self._ner_label_types = ['X', '[CLS]', '[SEP]', 'O', 'ORDINAL', 'PRODUCT', 'QUANTITY', 'TIME', 'MONEY', 'DATE', 'EVENT', 'GPE', 'PERCENT', 'LOC', 'LAW', 'FAC',
         'NORP', 'PERSON', 'CARDINAL', 'ORG', 'WORK_OF_ART', 'LANGUAGE']
            self._label_types = ['X', '[CLS]', '[SEP]', 'O', 'B-1']
            self._pos_label_types = ['X', '[CLS]', '[SEP]', 'O', '``', 'jjs', 'md', 'nns', 'uh', "''", 'rb', 'vbz',
                                     'pdt', 'ex', 'vbg', 'wrb', ':', 'sym', 'rp', 'prp', 'nnp', 'vb', 'vbn', 'pos',
                                     'to', 'wdt', 'fw', 'in', 'ls', 'dt', 'hvs', 'bes', 'cc', 'vbd', 'rbs', 'rbr', 'nn',
                                     'wp', 'gw', 'nnps', 'cd', 'jjr', '.', 'vbp', 'jj', ',', 'xx']

        elif tagger_type == 'kakao':
            self._label_types = ['X', '[CLS]', '[SEP]', 'O', 'I-<FD>', 'I-<RP2>',
                                 'B-<RP1>', 'B-<FD>', 'B-<RP2>', 'I-<RP1>']

            self._pos_label_types = ['X', '[CLS]', '[SEP]', 'O','XSV','VA','SW','SP','EF','VCP','SL','JX','ETM','JKS','XR','EC','MM','NNG','VV','NNB','JKB','XPN','IC','ETN','NR','NA','MAG','EP','JKG','JKO','JC','JKC','NP','MAJ','SN','JKV','NNP','VX','XSB','XSN','XSA','SF']

            self._ner_label_types = ['X', '[CLS]', '[SEP]', 'O']
            dir_path = '/data/project/rw/jude/transformer_disfluency_removal/data/train_test_v_3/'
            train_file_name = dir_path + 'filtered_train.bmes'
            test_file_name = dir_path + 'filtered_test.bmes'
            from_file = open(train_file_name, 'r', encoding='utf-8')
            from_test_file = open(test_file_name, 'r', encoding='utf-8')
            ner_set = set()
            for line in from_file:
                line = line.strip()
                if len(line) < 1:
                    continue
                splited_line = line.split()
                word, pos_tag, dis_tag, ner_tag = splited_line[0], splited_line[1], splited_line[-2], splited_line[-1]
                ner_set.add(ner_tag)

            for line in from_test_file:
                line = line.strip()
                if len(line) < 1:
                    continue
                splited_line = line.split()
                word, pos_tag, dis_tag, ner_tag = splited_line[0], splited_line[1], splited_line[-2], splited_line[-1]
                ner_set.add(ner_tag)

            self._ner_label_types.extend(list(ner_set))
            print('number of ner_label_types: ', len(self._ner_label_types))

        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i,
                                        label in enumerate(self._label_types)}

        self._num_ner_labels = len(self._ner_label_types)
        self._ner_label_map = {label: i for i,
                                        label in enumerate(self._ner_label_types)}

        self._num_pos_labels = len(self._pos_label_types)
        self._pos_label_map = {label: i for i,
                                        label in enumerate(self._pos_label_types)}

    def get_train_examples(self, data_dir, tagger_type):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, 'with_handcraft_ner_'+ tagger_type + '_train')))

    def get_dev_examples(self, data_dir, tagger_type):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, 'with_handcraft_ner_'+ tagger_type + '_dev')))

    def get_test_examples(self, data_dir, tagger_type):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, 'with_handcraft_ner_'+ tagger_type + '_test')))

    def get_labels(self):
        return self._label_types

    def get_ner_labels(self):
        return self._ner_label_types

    def get_pos_labels(self):
        return self._pos_label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map

    def get_ner_label_map(self):
        return self._ner_label_map

    def get_pos_label_map(self):
        return self._pos_label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def get_start_ner_label_id(self):
        return self._ner_label_map['[CLS]']

    def get_stop_ner_label_id(self):
        return self._ner_label_map['[SEP]']

    def get_start_pos_label_id(self):
        return self._pos_label_map['[CLS]']

    def get_stop_pos_label_id(self):
        return self._pos_label_map['[SEP]']

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            pos_tags = one_lists[1]
            dis_labels = one_lists[2]
            ner_labels = one_lists[3]
            examples.append(InputExample(
                guid=guid, words=words, pos_tags=pos_tags, labels=dis_labels, ner_labels=ner_labels, pos_labels=pos_tags))
        return examples


def example2feature(example, tokenizer, label_map, max_seq_length,
                    ner_label_map=None, pos_label_map=None):

    add_label = 'X'
    # tokenize_count = []
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    ner_label_ids = [ner_label_map['[CLS]']]
    pos_label_ids = [pos_label_map['[CLS]']]

    for i, w in enumerate(example.words):
        # use bertTokenizer to split words
        # 1996-08-22 => 1996 - 08 - 22
        # sheepmeat => sheep ##me ##at
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        # tokenize_count.append(len(sub_words))
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
                ner_label_ids.append(ner_label_map[example.ner_labels[i]])
                pos_label_ids.append(pos_label_map[example.pos_labels[i]])
            else:
                # '##xxx' -> 'X' (see bert paper)
                predict_mask.append(0)
                label_ids.append(label_map[add_label])
                ner_label_ids.append(ner_label_map[add_label])
                pos_label_ids.append(pos_label_map[add_label])

    # truncate
    if len(tokens) > max_seq_length - 1:
        print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens), max_seq_length))
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
        ner_label_ids = ner_label_ids[0:(max_seq_length - 1)]
        pos_label_ids = pos_label_ids[0:(max_seq_length - 1)]
    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])
    ner_label_ids.append(ner_label_map['[SEP]'])
    pos_label_ids.append(pos_label_map['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    feat =InputFeatures(
        # guid=example.guid,
        # tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        predict_mask=predict_mask,
        label_ids=label_ids,
        ner_label_ids=ner_label_ids,
        pos_label_ids=pos_label_ids)
    return feat


class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map,
                 max_seq_length, ner_label_map=None, pos_label_map=None):
        self.examples =examples
        self.tokenizer =tokenizer
        self.label_map =label_map
        self.max_seq_length =max_seq_length
        self.ner_label_map = ner_label_map
        self.pos_label_map = pos_label_map

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat =example2feature(self.examples[idx], self.tokenizer, self.label_map,
                              self.max_seq_length, ner_label_map=self.ner_label_map, pos_label_map=self.pos_label_map)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids, feat.ner_label_ids, feat.pos_label_ids

    @classmethod
    def pad(cls, batch):

        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))
        ner_label_ids_list = torch.LongTensor(f(5, maxlen))
        pos_label_ids_list = torch.LongTensor(f(6, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list, ner_label_ids_list, pos_label_ids_list


def f1_score(y_true, y_pred):
    '''
    0,1,2,3 are [CLS],[SEP],[X],O
    '''
    ignore_id = 3

    num_proposed = len(y_pred[y_pred > ignore_id])
    num_correct = (np.logical_and(y_true == y_pred, y_true > ignore_id)).sum()
    num_gold = len(y_true[y_true > ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    return precision, recall, f1