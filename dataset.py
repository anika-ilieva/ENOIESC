import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import utils
from transformers import BertTokenizer, BertTokenizerFast
from utils.bio import pred_tag2idx, arg_tag2idx, all_conjunction_tag2idx, all_conj_expr_tag2idx, all_conj_term_tag2idx


def load_data(data_path,
              batch_size,
              max_len=64,
              train=True,
              tokenizer_config='bert-base-cased',
              raw=False,
              conj=False):
    if raw:
        return DataLoader(OieRawDataset(
                data_path,
                max_len,
                tokenizer_config),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
    if train:
        if conj:
            return DataLoader(
                dataset=OieConjDataset(
                    data_path,
                    max_len,
                    tokenizer_config),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True)
        else:
            return DataLoader(
                dataset=OieDataset(
                    data_path,
                    max_len,
                    tokenizer_config),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True)
    else:
        return DataLoader(
            dataset=OieEvalDataset(
                data_path,
                max_len,
                tokenizer_config),
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True)


class OieDataset(Dataset):
    def __init__(self, data_path, max_len=64, tokenizer_config='bert-base-cased'):
        data = utils.load_pkl(data_path)
        self.tokens = data['tokens']
        self.single_pred_labels = data['single_pred_labels']
        self.single_arg_labels = data['single_arg_labels']
        self.all_pred_labels = data['all_pred_labels']

        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_config)
        self.vocab = self.tokenizer.vocab

        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def add_pad(self, token_ids):
        diff = self.max_len - len(token_ids)
        if diff > 0:
            token_ids += [self.pad_idx] * diff
        else:
            token_ids = token_ids[:self.max_len-1] + [self.sep_idx]
        return token_ids

    def add_special_token(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def idx2mask(self, token_ids):
        return [token_id != self.pad_idx for token_id in token_ids]

    def add_pad_to_labels(self, pred_label, arg_label, all_pred_label):
        pred_outside = np.array([pred_tag2idx['O']])
        arg_outside = np.array([arg_tag2idx['O']])

        pred_label = np.concatenate([pred_outside, pred_label, pred_outside])
        arg_label = np.concatenate([arg_outside, arg_label, arg_outside])
        all_pred_label = np.concatenate([pred_outside, all_pred_label, pred_outside])

        diff = self.max_len - pred_label.shape[0]
        if diff > 0:
            pred_pad = np.array([pred_tag2idx['O']] * diff)
            arg_pad = np.array([arg_tag2idx['O']] * diff)
            pred_label = np.concatenate([pred_label, pred_pad])
            arg_label = np.concatenate([arg_label, arg_pad])
            all_pred_label = np.concatenate([all_pred_label, pred_pad])
        elif diff == 0:
            pass
        else:
            pred_label = np.concatenate([pred_label[:-1], pred_outside])
            arg_label = np.concatenate([arg_label[:-1], arg_outside])
            all_pred_label = np.concatenate([all_pred_label[:-1], pred_outside])
        return [pred_label, arg_label, all_pred_label]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokens[idx])
        token_ids_padded = self.add_pad(self.add_special_token(token_ids))
        att_mask = self.idx2mask(token_ids_padded)
        labels = self.add_pad_to_labels(
            self.single_pred_labels[idx],
            self.single_arg_labels[idx],
            self.all_pred_labels[idx])
        single_pred_label, single_arg_label, all_pred_label = labels

        assert len(token_ids_padded) == self.max_len
        assert len(att_mask) == self.max_len
        assert single_pred_label.shape[0] == self.max_len
        assert single_arg_label.shape[0] == self.max_len
        assert all_pred_label.shape[0] == self.max_len

        batch = [
            torch.tensor(token_ids_padded),
            torch.tensor(att_mask),
            torch.tensor(single_pred_label),
            torch.tensor(single_arg_label),
            torch.tensor(all_pred_label)
        ]
        return batch

class OieConjDataset(Dataset):
    def __init__(self, data_path, max_len=64, tokenizer_config='bert-base-cased'):
        data = utils.load_pkl(data_path)
        self.tokens = data['tokens']
        self.single_pred_labels = data['single_pred_labels']
        self.single_arg_labels = data['single_arg_labels']
        self.all_pred_labels = data['all_pred_labels']

        self.all_conjunction_labels = data['all_conjunction_labels']
        self.all_conj_expr_labels = data['all_conj_expr_labels']
        self.all_conj_term_labels = data['all_conj_term_labels']

        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_config)
        self.vocab = self.tokenizer.vocab

        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def add_pad(self, token_ids):
        diff = self.max_len - len(token_ids)
        if diff > 0:
            token_ids += [self.pad_idx] * diff
        else:
            token_ids = token_ids[:self.max_len-1] + [self.sep_idx]
        return token_ids

    def add_special_token(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def idx2mask(self, token_ids):
        return [token_id != self.pad_idx for token_id in token_ids]

    def add_pad_to_labels(self, pred_label, arg_label, all_pred_label, all_conjunction_label_param, all_conj_expr_label_param, all_conj_term_label_param):
        pred_outside = np.array([pred_tag2idx['O']])
        arg_outside = np.array([arg_tag2idx['O']])

        all_conjunction_outside = np.array([all_conjunction_tag2idx['O']])
        all_conj_expr_outside = np.array([all_conj_expr_tag2idx['O']])
        all_conj_term_outside = np.array([all_conj_term_tag2idx['O']])

        pred_label = np.concatenate([pred_outside, pred_label, pred_outside])
        arg_label = np.concatenate([arg_outside, arg_label, arg_outside])
        all_pred_label = np.concatenate([pred_outside, all_pred_label, pred_outside])
        
        all_conjunction_label = np.concatenate([all_conjunction_outside, all_conjunction_label_param, all_conjunction_outside])
        all_conj_expr_label = np.concatenate([all_conj_expr_outside, all_conj_expr_label_param, all_conj_expr_outside])
        all_conj_term_label = np.concatenate([all_conj_term_outside, all_conj_term_label_param, all_conj_term_outside])                    

        diff = self.max_len - pred_label.shape[0]
        if diff > 0:
            pred_pad = np.array([pred_tag2idx['O']] * diff)
            arg_pad = np.array([arg_tag2idx['O']] * diff)
            
            all_conjunction_pad = np.array([all_conjunction_tag2idx['O']] * diff)
            all_conj_expr_pad = np.array([all_conj_expr_tag2idx['O']] * diff)
            all_conj_term_pad = np.array([all_conj_term_tag2idx['O']] * diff)

            pred_label = np.concatenate([pred_label, pred_pad])
            arg_label = np.concatenate([arg_label, arg_pad])
            all_pred_label = np.concatenate([all_pred_label, pred_pad])
            
            all_conjunction_label = np.concatenate([all_conjunction_label, all_conjunction_pad])
            all_conj_expr_label = np.concatenate([all_conj_expr_label, all_conj_expr_pad])
            all_conj_term_label = np.concatenate([all_conj_term_label, all_conj_term_pad])
        elif diff == 0:
            pass
        else:
            pred_label = np.concatenate([pred_label[:-1], pred_outside])
            arg_label = np.concatenate([arg_label[:-1], arg_outside])
            all_pred_label = np.concatenate([all_pred_label[:-1], pred_outside])
            
            all_conjunction_label = np.concatenate([all_conjunction_label[:-1], all_conjunction_outside])
            all_conj_expr_label = np.concatenate([all_conj_expr_label[:-1], all_conj_expr_outside])
            all_conj_term_label = np.concatenate([all_conj_term_label[:-1], all_conj_term_outside])

        return [pred_label, arg_label, all_pred_label, all_conjunction_label, all_conj_expr_label, all_conj_term_label]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokens[idx])
        token_ids_padded = self.add_pad(self.add_special_token(token_ids))
        att_mask = self.idx2mask(token_ids_padded)
        labels = self.add_pad_to_labels(
            self.single_pred_labels[idx],
            self.single_arg_labels[idx],
            self.all_pred_labels[idx],
            self.all_conjunction_labels[idx],
            self.all_conj_expr_labels[idx],
            self.all_conj_term_labels[idx]
            )
        single_pred_label, single_arg_label, all_pred_label, all_conjunction_label, all_conj_expr_label, all_conj_term_label = labels

        assert len(token_ids_padded) == self.max_len
        assert len(att_mask) == self.max_len
        assert single_pred_label.shape[0] == self.max_len
        assert single_arg_label.shape[0] == self.max_len
        assert all_pred_label.shape[0] == self.max_len
        assert all_conjunction_label.shape[0] == self.max_len
        assert all_conj_expr_label.shape[0] == self.max_len
        assert all_conj_term_label.shape[0] == self.max_len

        batch = [
            torch.tensor(token_ids_padded),
            torch.tensor(att_mask),
            torch.tensor(single_pred_label),
            torch.tensor(single_arg_label),
            torch.tensor(all_pred_label),
            torch.tensor(all_conjunction_label),
            torch.tensor(all_conj_expr_label),
            torch.tensor(all_conj_term_label)
        ]
        return batch

class OieEvalDataset(Dataset):
    def __init__(self, data_path, max_len, tokenizer_config='bert-base-cased'):
        self.sentences = utils.load_pkl(data_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_config)
        self.vocab = self.tokenizer.backend_tokenizer.get_vocab()
        self.max_len = max_len

        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']
        
    def add_pad(self, tokens):
        diff = self.max_len - len(tokens['input_ids'])
        if diff > 0:
            tokens['input_ids'] += [self.pad_idx] * diff
            tokens['attention_mask'] += [0] * diff
            tokens['offset_mapping'] += [(0,0)] * diff
        else:
            tokens['input_ids'] = tokens['input_ids'][:self.max_len-1] + [self.sep_idx]
            tokens['attention_mask'] = tokens['attention_mask'][:self.max_len]
            tokens['offset_mapping'] = tokens['offset_mapping'][:self.max_len-1] + [(0,0)]

    def idx2mask(self, token_ids):
        return [token_id != self.pad_idx for token_id in token_ids]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        tokens = self.tokenizer.encode_plus(sentence, return_offsets_mapping=True)
        del tokens['token_type_ids']

        self.add_pad(tokens)
        token_ids = tokens['input_ids']
        att_mask = tokens['attention_mask']
        offset_mapping = torch.tensor(tokens['offset_mapping'])
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)

        assert len(token_ids) == self.max_len
        assert len(att_mask) == self.max_len
        assert len(token_strs) == self.max_len
        assert len(offset_mapping) == self.max_len
        batch = [
            torch.tensor(token_ids),
            torch.tensor(att_mask),
            token_strs,
            sentence
        ]
        return offset_mapping, batch

class OieRawDataset(Dataset):
    def __init__(self, sentences, max_len, tokenizer_config='bert-base-cased'):
        self.sentences = sentences
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_config)
        self.vocab = self.tokenizer.backend_tokenizer.get_vocab()
        self.max_len = max_len

        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']
        
    def add_pad(self, tokens):
        diff = self.max_len - len(tokens['input_ids'])
        if diff > 0:
            tokens['input_ids'] += [self.pad_idx] * diff
            tokens['attention_mask'] += [0] * diff
            tokens['offset_mapping'] += [(0,0)] * diff
        else:
            tokens['input_ids'] = tokens['input_ids'][:self.max_len-1] + [self.sep_idx]
            tokens['attention_mask'] = tokens['attention_mask'][:self.max_len]
            tokens['offset_mapping'] = tokens['offset_mapping'][:self.max_len-1] + [(0,0)]

    def idx2mask(self, token_ids):
        return [token_id != self.pad_idx for token_id in token_ids]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        tokens = self.tokenizer.encode_plus(sentence, return_offsets_mapping=True)
        del tokens['token_type_ids']

        self.add_pad(tokens)
        token_ids = tokens['input_ids']
        att_mask = tokens['attention_mask']
        offset_mapping = torch.tensor(tokens['offset_mapping'])
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)

        assert len(token_ids) == self.max_len
        assert len(att_mask) == self.max_len
        assert len(token_strs) == self.max_len
        assert len(offset_mapping) == self.max_len
        batch = [
            torch.tensor(token_ids),
            torch.tensor(att_mask),
            token_strs,
            sentence
        ]
        return offset_mapping, batch

