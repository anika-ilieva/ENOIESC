import torch
import numpy as np

import os
import sys
import pathlib

TOP_LEVEL_DIRECTORY = str(pathlib.Path(__file__).parent.resolve().parent.absolute())
sys.path.insert(0, TOP_LEVEL_DIRECTORY)

from thesis.models.Sentence import Sentence

pred_tag2idx = {
    'P-B': 0, 'P-I': 1, 'O': 2
}
arg_tag2idx = {
    'A0-B': 0, 'A0-I': 1,
    'A1-B': 2, 'A1-I': 3,
    'A2-B': 4, 'A2-I': 5,
    'A3-B': 6, 'A3-I': 7,
    'O': 8,
}

all_conjunction_tag2idx = {
    'C-B': 0, 'C-I': 1, 'O': 2,
}

all_conj_expr_tag2idx = {
    'CE-B': 0, 'CE-I': 1, 'O': 2,
}

all_conj_term_tag2idx = {
    'CT-B': 0, 'CT-I': 1, 'O': 2
}

from utils import utils

def get_pred_idxs(pred_tags):
    idxs = list()
    for pred_tag in pred_tags:
        idxs.append([idx.item() for idx in (pred_tag != 2).nonzero()])
    return idxs


def get_pred_mask(tensor):
    """
    Generate predicate masks by converting predicate index with 'O' tag to 1.
    Other indexes are converted to 0 which means non-masking.

    :param tensor: predicate tagged tensor with the shape of (B, L),
        where B is the batch size, L is the sequence length.
    :return: masked binary tensor with the same shape.
    """
    res = tensor.clone()
    res[tensor == pred_tag2idx['O']] = 1
    res[tensor != pred_tag2idx['O']] = 0
    return torch.tensor(res, dtype=torch.bool, device=tensor.device)


def get_conjunction_mask(tensor):
    """
    Generate conjunction masks by converting conjunction index with 'O' tag to 1.
    Other indexes are converted to 0 which means non-masking.

    :param tensor: conjunction tagged tensor with the shape of (B, L),
        where B is the batch size, L is the sequence length.
    :return: masked binary tensor with the same shape.
    """
    res = tensor.clone()
    res[tensor == all_conjunction_tag2idx['O']] = 1
    res[tensor != all_conjunction_tag2idx['O']] = 0
    return torch.tensor(res, dtype=torch.bool, device=tensor.device)


def filter_pred_tags(pred_tags, tokens):
    """
    Filter useless tokens by converting them into 'Outside' tag.
    We treat 'Inside' tag before 'Beginning' tag as meaningful signal,
    so changed them to 'Beginning' tag unlike [Stanovsky et al., 2018].

    :param pred_tags: predicate tags with the shape of (B, L).
    :param tokens: list format sentence pieces with the shape of (B, L)
    :return: tensor of filtered predicate tags with the same shape.
    """
    assert len(pred_tags) == len(tokens)
    assert len(pred_tags[0]) == len(tokens[0])

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for pred_idx, cur_tokens in enumerate(tokens):
        for tag_idx, token in enumerate(cur_tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                pred_tags[pred_idx][tag_idx] = pred_tag2idx['O']

    # filter by tags
    # ensure that the first tag of a pred is a beginning tag, fix if necessary
    pred_copied = pred_tags.clone()
    for pred_idx, cur_pred_tag in enumerate(pred_copied):
        flag = False
        tag_copied = cur_pred_tag.clone()
        for tag_idx, tag in enumerate(tag_copied):
            if not flag and tag == pred_tag2idx['P-B']:
                flag = True
            elif not flag and tag == pred_tag2idx['P-I']:
                pred_tags[pred_idx][tag_idx] = pred_tag2idx['P-B']
                flag = True
            elif flag and tag == pred_tag2idx['O']:
                flag = False
    return pred_tags


def filter_arg_tags(arg_tags, pred_tags, tokens):
    """
    Same as the description of @filter_pred_tags().

    :param arg_tags: argument tags with the shape of (B, L).
    :param pred_tags: predicate tags with the same shape.
        It is used to force predicate position to be allocated the 'Outside' tag.
    :param tokens: list of string tokens with the length of L.
        It is used to force special tokens like [CLS] to be allocated the 'Outside' tag.
    :return: tensor of filtered argument tags with the same shape.
    """

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for arg_idx, cur_arg_tag in enumerate(arg_tags):
        for tag_idx, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                arg_tags[arg_idx][tag_idx] = arg_tag2idx['O']

    # filter by tags
    arg_copied = arg_tags.clone()
    for arg_idx, (cur_arg_tag, cur_pred_tag) in enumerate(zip(arg_copied, pred_tags)):
        # ensure that pred idxs are marked as outside
        pred_idxs = [idx[0].item() for idx
                     in (cur_pred_tag != pred_tag2idx['O']).nonzero()]
        arg_tags[arg_idx][pred_idxs] = arg_tag2idx['O']
        cur_arg_copied = arg_tags[arg_idx].clone()
        flag_idx = 999

        # ensure that the first tag of a arg is a beginning tag, fix if necessary
        for tag_idx, tag in enumerate(cur_arg_copied):
            if tag == arg_tag2idx['O']:
                flag_idx = 999
                continue
            arg_n = tag // 2  # 0: A0 / 1: A1 / ...
            inside = tag % 2  # 0: begin / 1: inside
            if not inside and flag_idx != arg_n:
                flag_idx = arg_n
            # connect_args
            elif not inside and flag_idx == arg_n:
                arg_tags[arg_idx][tag_idx] = arg_tag2idx[f'A{arg_n}-I']
            elif inside and flag_idx != arg_n:
                arg_tags[arg_idx][tag_idx] = arg_tag2idx[f'A{arg_n}-B']
                flag_idx = arg_n
    return arg_tags


def filter_conjunction_tags(conjunction_tags, tokens):
    """
    Filter useless tokens by converting them into 'Outside' tag.
    We treat 'Inside' tag before 'Beginning' tag as meaningful signal,
    so changed them to 'Beginning' tag unlike [Stanovsky et al., 2018].

    :param conjunction_tags: conjunction tags with the shape of (B, L).
    :param tokens: list format sentence pieces with the shape of (B, L)
    :return: tensor of filtered conjunction tags with the same shape.
    """
    assert len(conjunction_tags) == len(tokens)
    assert len(conjunction_tags[0]) == len(tokens[0])

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for conjunction_idx, cur_tokens in enumerate(tokens):
        for tag_idx, token in enumerate(cur_tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                conjunction_tags[conjunction_idx][tag_idx] = all_conjunction_tag2idx['O']

    # filter by tags
    # ensure that the first tag of a conjunction is a beginning tag, fix if necessary
    conjunction_copied = conjunction_tags.clone()
    for conjunction_idx, cur_conjunction_tag in enumerate(conjunction_copied):
        flag = False
        tag_copied = cur_conjunction_tag.clone()
        for tag_idx, tag in enumerate(tag_copied):
            if not flag and tag == all_conjunction_tag2idx['C-B']:
                flag = True
            elif not flag and tag == all_conjunction_tag2idx['C-I']:
                conjunction_tags[conjunction_idx][tag_idx] = all_conjunction_tag2idx['C-B']
                flag = True
            elif flag and tag == all_conjunction_tag2idx['O']:
                flag = False
    return conjunction_tags

def filter_all_conj_expr_tags(all_conj_expr_tags, all_pred_tag, tokens):
    """
    Same as the description of @filter_pred_tags().

    :param all_conj_expr_tags: all_conj_expr_tags tags with the shape of (B, L).
    :param all_pred_tag: predicate tags with the same shape.
        It is used to force predicate position to be allocated the 'Outside' tag.
    :param tokens: list of string tokens with the length of L.
        It is used to force special tokens like [CLS] to be allocated the 'Outside' tag.
    :return: tensor of filtered conjunction tags with the same shape.
    """

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for all_conj_expr_idx, cur_tokens in enumerate(tokens):
        for tag_idx, token in enumerate(cur_tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                all_conj_expr_tags[all_conj_expr_idx][tag_idx] = all_conj_expr_tag2idx['O']

    # filter by tags
    all_conj_expr_copied = all_conj_expr_tags.clone()
    for all_conj_expr_idx, (cur_all_conj_expr_tag, cur_all_pred_tag) in enumerate(zip(all_conj_expr_copied, all_pred_tag)):
        # ensure that pred idxs are marked as outside
        pred_idxs = [idx[0].item() for idx
                    in (cur_all_pred_tag != pred_tag2idx['O']).nonzero()]
        all_conj_expr_tags[all_conj_expr_idx][pred_idxs] = all_conj_expr_tag2idx['O']

        tag_copied = cur_all_conj_expr_tag.clone()
        # ensure that the first tag of a conj expr is a beginning tag, fix if necessary
        flag = False
        for tag_idx, tag in enumerate(tag_copied):
            if not flag and tag == all_conj_expr_tag2idx['CE-B']:
                flag = True
            elif not flag and tag == all_conj_expr_tag2idx['CE-I']:
                all_conj_expr_tags[all_conj_expr_idx][tag_idx] = all_conj_expr_tag2idx['CE-B']
                flag = True
            elif flag and tag == all_conj_expr_tag2idx['O']:
                flag = False
    return all_conj_expr_tags

def filter_all_conj_term_tags(all_conj_term_tags, all_pred_tag, tokens):
    """
    Same as the description of @filter_pred_tags().

    :param all_conj_term_tags: all_conjunction_term tags with the shape of (B, L).
    :param all_pred_tag: predicate tags with the same shape.
        It is used to force predicate position to be allocated the 'Outside' tag.
    :param tokens: list of string tokens with the length of L.
        It is used to force special tokens like [CLS] to be allocated the 'Outside' tag.
    :return: tensor of filtered conjunction_term tags with the same shape.
    """

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for all_conj_term_idx, cur_tokens in enumerate(tokens):
        for tag_idx, token in enumerate(cur_tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                all_conj_term_tags[all_conj_term_idx][tag_idx] = all_conj_term_tag2idx['O']

    # filter by tags
    all_conj_term_copied = all_conj_term_tags.clone()
    for all_conj_term_idx, (cur_all_conj_term_tag, cur_all_pred_tag) in enumerate(zip(all_conj_term_copied, all_pred_tag)):
        # ensure that pred idxs are marked as outside
        pred_idxs = [idx[0].item() for idx
                    in (cur_all_pred_tag != pred_tag2idx['O']).nonzero()]
        all_conj_term_tags[all_conj_term_idx][pred_idxs] = all_conj_term_tag2idx['O']

        tag_copied = cur_all_conj_term_tag.clone()
        # ensure that the first tag of a conj term is a beginning tag, fix if necessary
        flag = False
        for tag_idx, tag in enumerate(tag_copied):
            if not flag and tag == all_conj_term_tag2idx['CT-B']:
                flag = True
            elif not flag and tag == all_conj_term_tag2idx['CT-I']:
                all_conj_term_tags[all_conj_term_idx][tag_idx] = all_conj_term_tag2idx['CT-B']
                flag = True
            elif flag and tag == all_conj_term_tag2idx['O']:
                flag = False
    return all_conj_term_tags


def get_max_prob_args(arg_tags, arg_probs):
    """
    Among predicted argument tags, remain only arguments with highest probs.
    The comparison of probability is made only between the same argument labels.

    :param arg_tags: argument tags with the shape of (B, L).
    :param arg_probs: argument softmax probabilities with the shape of (B, L, T),
        where B is the batch size, L is the sequence length, and T is the # of tag labels.
    :return: tensor of filtered argument tags with the same shape.
    """
    for cur_arg_tag, cur_probs in zip(arg_tags, arg_probs):
        cur_tag_probs = [cur_probs[idx][tag] for idx, tag in enumerate(cur_arg_tag)]
        for arg_n in range(4):
            b_tag = arg_tag2idx[f"A{arg_n}-B"]
            i_tag = arg_tag2idx[f"A{arg_n}-I"]
            flag = False
            total_tags = []
            cur_tags = []
            for idx, tag in enumerate(cur_arg_tag):
                if not flag and tag == b_tag:
                    flag = True
                    cur_tags.append(idx)
                elif flag and tag == i_tag:
                    cur_tags.append(idx)
                elif flag and tag == b_tag:
                    total_tags.append(cur_tags)
                    cur_tags = [idx]
                elif tag != b_tag or tag != i_tag:
                    total_tags.append(cur_tags)
                    cur_tags = []
                    flag = False
            max_idxs, max_prob = None, 0.0
            for idxs in total_tags:
                all_probs = [cur_tag_probs[idx].item() for idx in idxs]
                if len(all_probs) == 0:
                    continue
                cur_prob = all_probs[0]
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    max_idxs = idxs
            if max_idxs is None:
                continue
            del_idxs = [idx for idx, tag in enumerate(cur_arg_tag)
                        if (tag in [b_tag, i_tag]) and (idx not in max_idxs)]
            cur_arg_tag[del_idxs] = arg_tag2idx['O']
    return arg_tags


def get_single_predicate_idxs(pred_tags):
    """
    Divide each single batch based on predicted predicates.
    It is necessary for predicting argument tags with specific predicate.

    :param pred_tags: tensor of predicate tags with the shape of (B, L)
        EX >>> tensor([[2, 0, 0, 1, 0, 1, 0, 2, 2, 2],
                       [2, 2, 2, 0, 1, 0, 1, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 0, 1]])

    :return: list of tensors with the shape of (B, P, L)
        the number P can be different for each batch.
        EX >>> [tensor([[2., 0., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 0., 1., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 0., 1., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 2., 0., 2., 2., 2.]]),
                tensor([[2., 2., 2., 0., 1., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 0., 1., 2., 2., 2.]]),
                tensor([[2., 2., 2., 2., 2., 2., 2., 2., 0., 1.]])]
    """
    total_pred_tags = []
    for cur_pred_tag in pred_tags:
        cur_sent_preds = []
        begin_idxs = [idx[0].item() for idx in (cur_pred_tag == pred_tag2idx['P-B']).nonzero()]
        for i, b_idx in enumerate(begin_idxs):
            cur_pred = np.full(cur_pred_tag.shape[0], pred_tag2idx['O'])
            cur_pred[b_idx] = pred_tag2idx['P-B']
            if i == len(begin_idxs) - 1:
                end_idx = cur_pred_tag.shape[0]
            else:
                end_idx = begin_idxs[i + 1]
            for j, tag in enumerate(cur_pred_tag[b_idx:end_idx]):
                if tag.item() == pred_tag2idx['O']:
                    break
                elif tag.item() == pred_tag2idx['P-I']:
                    cur_pred[b_idx + j] = pred_tag2idx['P-I']
            cur_sent_preds.append(cur_pred)
        total_pred_tags.append(cur_sent_preds)
    return [torch.Tensor(pred_tags) for pred_tags in total_pred_tags]

def get_single_conjunction_idxs(conjunction_tags):
    """
    Divide each single batch based on predicted conjuncts.

    :param conjunction_tags: tensor of conjunction tags with the shape of (B, L)
        EX >>> tensor([[2, 0, 0, 1, 0, 1, 0, 2, 2, 2],
                       [2, 2, 2, 0, 1, 0, 1, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 0, 1]])

    :return: list of tensors with the shape of (B, P, L)
        the number P can be different for each batch.
        EX >>> [tensor([[2., 0., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 0., 1., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 0., 1., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 2., 0., 2., 2., 2.]]),
                tensor([[2., 2., 2., 0., 1., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 0., 1., 2., 2., 2.]]),
                tensor([[2., 2., 2., 2., 2., 2., 2., 2., 0., 1.]])]
    """
    total_conjunction_tags = []
    for cur_conjunction_tag in conjunction_tags:
        cur_sent_conjunctions = []
        begin_idxs = [idx[0].item() for idx in (
            cur_conjunction_tag == all_conjunction_tag2idx['C-B']).nonzero()]
        for i, b_idx in enumerate(begin_idxs):
            cur_conjunction = np.full(
                cur_conjunction_tag.shape[0], all_conjunction_tag2idx['O'])
            cur_conjunction[b_idx] = all_conjunction_tag2idx['C-B']
            if i == len(begin_idxs) - 1:
                end_idx = cur_conjunction_tag.shape[0]
            else:
                end_idx = begin_idxs[i + 1]
            for j, tag in enumerate(cur_conjunction_tag[b_idx:end_idx]):
                if tag.item() == all_conjunction_tag2idx['O']:
                    break
                elif tag.item() == all_conjunction_tag2idx['C-I']:
                    cur_conjunction[b_idx + j] = all_conjunction_tag2idx['C-I']
            cur_sent_conjunctions.append(cur_conjunction)
        total_conjunction_tags.append(cur_sent_conjunctions)
    return [torch.Tensor(conjunction_tags) for conjunction_tags in total_conjunction_tags]


def get_single_conj_expr_idxs(conj_expr_tags):
    """
    Divide each single batch based on predicted conj exprs.

    :param conj_tags: tensor of conj expr tags with the shape of (B, L)
        EX >>> tensor([[2, 0, 0, 1, 0, 1, 0, 2, 2, 2],
                       [2, 2, 2, 0, 1, 0, 1, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 0, 1]])

    :return: list of tensors with the shape of (B, P, L)
        the number P can be different for each batch.
        EX >>> [tensor([[2., 0., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 0., 1., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 0., 1., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 2., 0., 2., 2., 2.]]),
                tensor([[2., 2., 2., 0., 1., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 0., 1., 2., 2., 2.]]),
                tensor([[2., 2., 2., 2., 2., 2., 2., 2., 0., 1.]])]
    """
    total_conj_expr_tags = []
    for cur_conj_expr_tag in conj_expr_tags:
        cur_sent_conj_exprs = []
        begin_idxs = [idx[0].item() for idx in (
            cur_conj_expr_tag == all_conj_expr_tag2idx['CE-B']).nonzero()]
        for i, b_idx in enumerate(begin_idxs):
            cur_conj_expr = np.full(cur_conj_expr_tag.shape[0], all_conj_expr_tag2idx['O'])
            cur_conj_expr[b_idx] = all_conj_expr_tag2idx['CE-B']
            if i == len(begin_idxs) - 1:
                end_idx = cur_conj_expr_tag.shape[0]
            else:
                end_idx = begin_idxs[i + 1]
            for j, tag in enumerate(cur_conj_expr_tag[b_idx:end_idx]):
                if tag.item() == all_conj_expr_tag2idx['O']:
                    break
                elif tag.item() == all_conj_expr_tag2idx['CE-I']:
                    cur_conj_expr[b_idx + j] = all_conj_expr_tag2idx['CE-I']
            cur_sent_conj_exprs.append(cur_conj_expr)
        total_conj_expr_tags.append(cur_sent_conj_exprs)
    return [torch.Tensor(conj_tags) for conj_tags in total_conj_expr_tags]


def get_single_conj_term_idxs(conj_term_tags):
    """
    Divide each single batch based on predicted conj_terms.

    :param conj_term_tags: tensor of conj_term tags with the shape of (B, L)
        EX >>> tensor([[2, 0, 0, 1, 0, 1, 0, 2, 2, 2],
                       [2, 2, 2, 0, 1, 0, 1, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 0, 1]])

    :return: list of tensors with the shape of (B, P, L)
        the number P can be different for each batch.
        EX >>> [tensor([[2., 0., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 0., 1., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 0., 1., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 2., 0., 2., 2., 2.]]),
                tensor([[2., 2., 2., 0., 1., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 0., 1., 2., 2., 2.]]),
                tensor([[2., 2., 2., 2., 2., 2., 2., 2., 0., 1.]])]
    """
    total_conj_term_tags = []
    for cur_conj_term_tag in conj_term_tags:
        cur_sent_conj_terms = []
        begin_idxs = [idx[0].item() for idx in (
            cur_conj_term_tag == all_conj_term_tag2idx['CT-B']).nonzero()]
        for i, b_idx in enumerate(begin_idxs):
            cur_conj_term = np.full(
                cur_conj_term_tag.shape[0], all_conj_term_tag2idx['O'])
            cur_conj_term[b_idx] = all_conj_term_tag2idx['CT-B']
            if i == len(begin_idxs) - 1:
                end_idx = cur_conj_term_tag.shape[0]
            else:
                end_idx = begin_idxs[i + 1]
            for j, tag in enumerate(cur_conj_term_tag[b_idx:end_idx]):
                if tag.item() == all_conj_term_tag2idx['O']:
                    break
                elif tag.item() == all_conj_term_tag2idx['CT-I']:
                    cur_conj_term[b_idx + j] = all_conj_term_tag2idx['CT-I']
            cur_sent_conj_terms.append(cur_conj_term)
        total_conj_term_tags.append(cur_sent_conj_terms)
    return [torch.Tensor(conj_term_tags) for conj_term_tags in total_conj_term_tags]

def get_triple(sentence, pred_tags, arg_tags, all_conjunction_tags, cur_conjunction_tags, cur_conj_expr_tags, cur_conj_term_tags, tokenizer):
    """
    Get string format tuples from given predicate indexes and argument tags.

    :param sentence: string format raw sentence.
    :param pred_tags: tensor of predicate tags with the shape of (# of predicates, sequence length).
    :param arg_tags: tensor of argument tags with the same shape.
    :param cur_conjunction_tags: tensor of conjunction tags with the same shape.
    :param cur_conj_expr_tags: tensor of conj expr tags with the same shape.
    :param cur_conj_term_tags: tensor of conj term tags with the same shape.
    :param tokenizer: transformer BertTokenizer (bert-base-cased or bert-base-multilingual-cased)

    :return extractions: list of strings each element means predicate, arg0, arg1, ...
    :return extraction_idxs: list of indexes of each argument for calculating confidence score.
    :return extraction_token_idxs: list of indexes of each argument which represent the extractions (same indexes which are used to build the extraction strings).
    """
    assert pred_tags.shape[0] == arg_tags.shape[0]  # number of predicates

    sent = Sentence(sentence)
    conj_expr_to_conj_term_map = get_conj_term_map(
        cur_conjunction_tags, cur_conj_expr_tags, cur_conj_term_tags)

    extractions = []
    extraction_idxs = []
    extraction_token_idxs = []
    extraction_conj_term_idxs = []
    extraction_conjs_in_arg_idxs = []
    pred_map = []

    all_conjunct_idxs = [idx for idx, tag in enumerate(
        all_conjunction_tags[1:-1]) if tag != all_conjunction_tag2idx['O']]
    
    extraction_conjunct_idxs = [idx for idx, tag in enumerate(
        all_conjunction_tags) if tag != all_conjunction_tag2idx['O']]

    # loop for each predicate
    for pred_id, (cur_pred_tag, cur_arg_tags) in enumerate(zip(pred_tags, arg_tags)):
        cur_extractions = []
        cur_extraction_idxs_s = []
        cur_extraction_token_idxs_s = []
        cur_conj_term_idxs = []
        cur_conj_exprs_in_arg_idxs = []

        arg_mask = (cur_arg_tags != arg_tag2idx['O'])

        conj_exprs_in_arg = get_conj_exprs_in_arg(
            conj_expr_to_conj_term_map, arg_mask, cur_conj_term_tags)

        pred_extraction, pred_extraction_idxs, pred_extraction_token_idxs = sent.pred_label_to_string(
            cur_pred_tag[1:-1], shift=1) # shift is necessary to be compatible with the original multi2oie repository

        conj_exprs_in_arg_idxs = []

        no_conj_expr_arg_tag = cur_arg_tags.clone()
        for conj_expr in conj_exprs_in_arg:
            conj_mask = (conj_expr != all_conj_expr_tag2idx['O'])
            no_conj_expr_arg_tag[conj_mask] = arg_tag2idx['O']
            
            conj_exprs_in_arg_idxs += [idx for idx,
                                  flag in enumerate(conj_mask) if flag]

        domain = [conj_expr_to_conj_term_map[x] for x in conj_exprs_in_arg]

        if len(domain) > 0:
            options = np.array(np.meshgrid(*domain)).T.reshape(-1, len(domain))

            for option in options:
                conj_term_idxs = []

                skip_option = False
                tmp_arg_tag = no_conj_expr_arg_tag.clone()
                for conj_term_tag_id in option:
                    conj_term_mask = (
                        cur_conj_term_tags[conj_term_tag_id] != all_conj_term_tag2idx['O']).to(arg_mask.device)

                    if not torch.all((arg_mask | conj_term_mask) == arg_mask):
                        skip_option = True
                    conj_term_idxs += [idx for idx, flag in enumerate(conj_term_mask) if flag]
                    tmp_arg_tag[conj_term_mask] = cur_arg_tags[conj_term_mask]

                if not skip_option:
                    arg_extraction, arg_extraction_idxs, arg_extraction_token_idxs = sent.arg_label_to_string(
                        tmp_arg_tag[1:-1], all_conjunct_idxs, shift=1)  # shift is necessary to be compatible with the original multi2oie repository

                    pred_map.append(pred_id)
                    cur_extractions.append([pred_extraction] + arg_extraction)
                    cur_extraction_idxs_s.append([pred_extraction_idxs] + arg_extraction_idxs)
                    cur_extraction_token_idxs_s.append([pred_extraction_token_idxs] + arg_extraction_token_idxs)
                    cur_conj_term_idxs.append(conj_term_idxs)
                    cur_conj_exprs_in_arg_idxs.append(conj_exprs_in_arg_idxs)
        else:
            arg_extraction, arg_extraction_idxs, arg_extraction_token_idxs = sent.arg_label_to_string(
                cur_arg_tags[1:-1], shift=1) # shift is necessary to be compatible with the original multi2oie repository
            
            pred_map.append(pred_id)
            cur_extractions.append([pred_extraction] + arg_extraction)
            cur_extraction_idxs_s.append([pred_extraction_idxs] + arg_extraction_idxs)
            cur_extraction_token_idxs_s.append([pred_extraction_token_idxs] + arg_extraction_token_idxs)
            cur_conj_term_idxs.append([])
            cur_conj_exprs_in_arg_idxs.append([])
        
        extractions += cur_extractions
        extraction_idxs += cur_extraction_idxs_s
        extraction_token_idxs += cur_extraction_token_idxs_s
        extraction_conj_term_idxs += cur_conj_term_idxs
        extraction_conjs_in_arg_idxs += cur_conj_exprs_in_arg_idxs
    return extractions, extraction_idxs, extraction_token_idxs, extraction_conjunct_idxs, extraction_conj_term_idxs, extraction_conjs_in_arg_idxs, pred_map


def idxToOffsets(idx, offset_mapping):
    """
    convert idx to offset intervals

    :param idx: (indexes of the given predicate or argument)
    :param offset_mapping: (offset mapping of the sentence tokens)

    """
    offsets = []
    for idx_group in idx:
        if len(idx_group) == 0:
            offsets.append([])
            continue
        group_offsets = []
        begin = idx_group[0]
        tmp_index = begin
        for i in range(1, len(idx_group)+1):
            if i == len(idx_group) or idx_group[i] != tmp_index+1:
                group_offsets.append([offset_mapping[begin][0].item(), offset_mapping[tmp_index][1].item()])
                if i < len(idx_group):
                    begin = idx_group[i]
            
            if i < len(idx_group):
                tmp_index = idx_group[i]
        offsets.append(group_offsets)
    
    return offsets


def get_confidence_score(pred_probs, arg_probs, conjunction_probs, conj_expr_probs, conj_term_probs, extraction_idxs, conjunction_idxs, conj_term_idxs, conj_exprs_in_arg_idxs, pred_map):
    """
    get the confidence score of each extraction for drawing PR-curve.

    :param pred_probs: (sequence length, # of predicate labels)
    :param arg_probs: (# of predicates, sequence length, # of argument labels)
    :param extraction_idxs: [[[2, 3, 4], [0, 1], [9, 10]], [[0, 1, 2], [7, 8], [4, 5]], ...]
    """
    confidence_scores = list()
    assert len(conj_exprs_in_arg_idxs) == len(extraction_idxs)
    assert len(conj_term_idxs) == len(extraction_idxs)
    for cur_arg_prob, cur_ext_idxs, cur_conj_term_idxs, cur_conj_exprs_in_arg_idxs in zip(arg_probs[pred_map], extraction_idxs, conj_term_idxs, conj_exprs_in_arg_idxs):
        if len(cur_ext_idxs[0]) == 0:
            confidence_scores.append(0)
            continue
        cur_score = 0

        # conjunction score
        if len(conjunction_idxs) > 0:
            begin_idxs = _find_begins(conjunction_idxs)
            conjunct_score = np.mean([max(conjunction_probs[cur_idx]).item() for cur_idx in begin_idxs])
            cur_score += conjunct_score

        # conj term score
        if len(cur_conj_term_idxs) > 0:
            begin_idxs = _find_begins(cur_conj_term_idxs)
            conj_term_score = np.mean([max(conj_term_probs[cur_idx]).item() for cur_idx in begin_idxs])
            cur_score += conj_term_score

        # conj expr score
        if len(cur_conj_exprs_in_arg_idxs) > 0:
            begin_idxs = _find_begins(cur_conj_exprs_in_arg_idxs)
            conj_score = np.mean(
                [max(conj_expr_probs[cur_idx]).item() for cur_idx in begin_idxs])
            cur_score += conj_score

        # predicate score
        pred_score = max(pred_probs[cur_ext_idxs[0][0]]).item()
        cur_score += pred_score

        # argument score
        for arg_idx in cur_ext_idxs[1:]:
            if len(arg_idx) == 0:
                continue
            begin_idxs = _find_begins(arg_idx)
            arg_score = np.mean([max(cur_arg_prob[cur_idx]).item() for cur_idx in begin_idxs])
            cur_score += arg_score
        confidence_scores.append(cur_score)
    return confidence_scores


def _find_begins(idxs):
    found_begins = [idxs[0]]
    cur_flag_idx = idxs[0]
    for cur_idx in idxs[1:]:
        if cur_idx - cur_flag_idx != 1:
            found_begins.append(cur_idx)
        cur_flag_idx = cur_idx
    return found_begins


def get_conj_term_map(cur_conjunction_tags, cur_conj_expr_tags, cur_conj_term_tags):
  conj_conj_term_map = {}

  for conj_expr_tag in cur_conj_expr_tags:
    conj_mask = (conj_expr_tag != all_conj_expr_tag2idx['O'])
    conj_term_tags = []
    for i, conj_term_tag in enumerate(cur_conj_term_tags):
      # ensure that conjunction indices are marked as outside in a conj term
      for conjunction_tag in cur_conjunction_tags:
        conjunct_mask = (conjunction_tag != all_conjunction_tag2idx['O'])
        conj_term_tag[conjunct_mask] = all_conj_term_tag2idx['O']

      # collect conj terms corresponding to the current conj expr
      conj_term_mask = (conj_term_tag != all_conj_term_tag2idx['O'])
      overlap = (conj_mask | conj_term_mask)
      if torch.all(overlap == conj_mask):
        conj_term_tags.append(i)

    if len(conj_term_tags) > 1:
      conj_conj_term_map[conj_expr_tag] = conj_term_tags

  return conj_conj_term_map


def get_conj_exprs_in_arg(conj_expr_to_conj_term_map, arg_mask, cur_conj_term_tags):
  conj_exprs_in_arg = []

  for conj_expr_tag in conj_expr_to_conj_term_map:
    n_terms = 0

    for conj_term_id in conj_expr_to_conj_term_map[conj_expr_tag]:
      conj_term_mask = (cur_conj_term_tags[conj_term_id] !=
                        all_conj_term_tag2idx['O']).to(arg_mask.device)

      if torch.all((arg_mask | conj_term_mask) == arg_mask):
        n_terms += 1
    if n_terms > 1:
      conj_exprs_in_arg.append(conj_expr_tag)
  return conj_exprs_in_arg
