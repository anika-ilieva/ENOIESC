"""
Script for transforming original dataset for Multi^2OIE training and evaluation

1. if Mode == 'train'
- input: structured.json (https://github.com/zhanjunlang/Span_OIE)
- output: ../datasets/openie4_train.pkl

2. if Mode == 'dev_input'
- input: dev.oie.conll (https://github.com/gabrielStanovsky/supervised-oie/tree/master/data)
- output: ../datasets/oie2016_dev.pkl

3. if Mode == 'dev_gold'
- input: dev.oie.conll (https://github.com/gabrielStanovsky/supervised-oie/tree/master/data)
- output: ../evaluate/OIE2016_dev.txt

"""
import os
import sys
import pathlib

TOP_LEVEL_DIRECTORY = str(pathlib.Path(__file__).parent.resolve().parent.absolute())
sys.path.insert(0, TOP_LEVEL_DIRECTORY)

import json
import numpy as np
import argparse
import pickle
from tqdm import tqdm
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

max_arg_num = 4
max_conj_num = 4

def preprocess_train(args):
    print("loading dataset...")

    with open(args.data) as json_file:
        data = json.load(json_file)
    iterator = tqdm(data)
    print("done. preprocessing starts.")

    openie4_train = {
        'tokens': list(),
        'single_pred_labels': list(),
        'single_arg_labels': list(),
        'all_pred_labels': list(),
        'all_conjunction_labels': list(),
        'all_conj_expr_labels': list(),
        'all_conj_term_labels': list()
    }

    rel_pos_malformed = 0
    max_over_case = 0
    conj_expr_overlap_counter = 0
    conj_expr_overlap_tuple_counter = 0

    for cur_data in iterator:
        sent = Sentence(cur_data['sentence'])
        sentence_pieces = sent.sentence_pieces()
        word2piece = sent.word2piece()

        # if the length of sentencepieces is over maxlen-2, we skip the sentence.
        if len(sentence_pieces) > args.max_len - 2:
            max_over_case += 1
            continue

        conj_expr_overlaps_b = len(cur_data['conj_expr_overlaps']) > 0
        conj_expr_overlaps_with_supplements_b = len(
            cur_data['conj_expr_overlaps_with_supplements']) > 0
        conj_term_overlaps_with_supplements_b = len(
            cur_data['conj_term_overlaps_with_supplements']) > 0

        if conj_expr_overlaps_b or conj_expr_overlaps_with_supplements_b or conj_term_overlaps_with_supplements_b:
            conj_expr_overlap_counter += 1
            conj_expr_overlap_tuple_counter += len(cur_data['tuples'])
            continue

        # add label for all conjuncts
        all_conjunction_label = np.asarray([all_conjunction_tag2idx['O'] for _ in range(len(sentence_pieces))])
        
        for conj_expr in cur_data['all_conjunction_offsets']:
            for conjunct_terms in conj_expr:
                conjunct_term_pieces = sent.offsets_to_sentence_pieces(
                    conjunct_terms)
                for start_conjunction_idx, end_conjunction_idx in conjunct_term_pieces:
                    for conjunction_idx in range(start_conjunction_idx, end_conjunction_idx + 1):
                        all_conjunction_label[conjunction_idx] = all_conjunction_tag2idx['C-I']
                    all_conjunction_label[start_conjunction_idx] = all_conjunction_tag2idx['C-B']

        # add label for all conjunctions and conj terms
        all_conj_expr_label = np.asarray([all_conj_expr_tag2idx['O'] for _ in range(len(sentence_pieces))])
        all_conj_term_label = np.asarray([all_conj_term_tag2idx['O'] for _ in range(len(sentence_pieces))])
        for conj_expr in cur_data['all_conj_term_offsets_with_supplements']:
            for conj_terms in conj_expr:
                conj_term_pieces = sent.offsets_to_sentence_pieces(conj_terms)
                for start_conj_term_idx, end_conj_term_idx in conj_term_pieces:
                    for conj_term_idx in range(start_conj_term_idx, end_conj_term_idx + 1):
                        all_conj_term_label[conj_term_idx] = all_conj_term_tag2idx['CT-I']
                if len(conj_term_pieces) > 0:
                    all_conj_term_label[conj_term_pieces[0][0]] = all_conj_term_tag2idx['CT-B']

            first_conj_term_pieces = sent.offsets_to_sentence_pieces(conj_expr[0])
            last_conj_term_pieces = sent.offsets_to_sentence_pieces(conj_expr[-1])

            for conj_term_idx in range(first_conj_term_pieces[0][0], last_conj_term_pieces[-1][1] + 1):
                all_conj_expr_label[conj_term_idx] = all_conj_expr_tag2idx['CE-I']
            all_conj_expr_label[first_conj_term_pieces[0]
                                [0]] = all_conj_expr_tag2idx['CE-B']

        all_pred_label = np.asarray([pred_tag2idx['O'] for _ in range(len(sentence_pieces))])
        cur_tuple_malformed = 0
        for cur_tuple in cur_data['tuples']:

            # add predicate labels
            pred_label = np.asarray([pred_tag2idx['O'] for _ in range(len(sentence_pieces))])
            if -1 in cur_tuple['rel_pos']:
                rel_pos_malformed += 1
                cur_tuple_malformed += 1
                continue
            else:
                start_idx, end_idx = cur_tuple['rel_pos'][:2]
            for pred_word_idx in range(start_idx, end_idx + 1):
                pred_label[word2piece[pred_word_idx]] = pred_tag2idx['P-I']
                all_pred_label[word2piece[pred_word_idx]] = pred_tag2idx['P-I']
            pred_label[word2piece[start_idx][0]] = pred_tag2idx['P-B']
            all_pred_label[word2piece[start_idx][0]] = pred_tag2idx['P-B']
            openie4_train['single_pred_labels'].append(pred_label)

            # add argument-0 labels
            arg_label = np.asarray([arg_tag2idx['O'] for _ in range(len(sentence_pieces))])
            start_idx, end_idx = cur_tuple['arg0_pos']
            for arg_word_idx in range(start_idx, end_idx + 1):
                arg_label[word2piece[arg_word_idx]] = arg_tag2idx['A0-I']
            arg_label[word2piece[start_idx][0]] = arg_tag2idx['A0-B']

            # add additional argument labels
            for arg_n, arg_pos in enumerate(cur_tuple['args_pos'][:max_arg_num-1]):
                arg_n += 1
                start_idx, end_idx = arg_pos
                for arg_word_idx in range(start_idx, end_idx + 1):
                    arg_label[word2piece[arg_word_idx]] = arg_tag2idx[f'A{arg_n}-I']
                arg_label[word2piece[start_idx][0]] = arg_tag2idx[f'A{arg_n}-B']
            openie4_train['single_arg_labels'].append(arg_label)
            

        # add sentence pieces and total predicate label of current sentence
        for _ in range(len(cur_data['tuples']) - cur_tuple_malformed):
            openie4_train['tokens'].append(sentence_pieces)
            openie4_train['all_pred_labels'].append(all_pred_label)
            openie4_train['all_conjunction_labels'].append(all_conjunction_label)
            openie4_train['all_conj_expr_labels'].append(all_conj_expr_label)
            openie4_train['all_conj_term_labels'].append(all_conj_term_label)

        assert len(openie4_train['all_conj_term_labels']) == len(openie4_train['all_conjunction_labels'])
        assert len(openie4_train['all_conj_expr_labels']) == len(openie4_train['all_conj_term_labels'])
        assert len(openie4_train['all_conj_term_labels']) == len(openie4_train['tokens'])
        assert len(openie4_train['tokens']) == len(openie4_train['all_pred_labels'])
        assert len(openie4_train['all_pred_labels']) == len(openie4_train['single_pred_labels'])
        assert len(openie4_train['single_pred_labels']) == len(openie4_train['single_arg_labels'])

    save_pkl(args.save_path, openie4_train)
    print(f"# of data over max length: {max_over_case}")
    print(f"# of data with malformed relation positions: {rel_pos_malformed}")
    print(f"# of data with overlapping conj expr: {conj_expr_overlap_counter}")
    print(
        f"# of tuples in data overlapping conj expr: {conj_expr_overlap_tuple_counter}")
    print("\npreprocessing done.")

    """
    For English BERT,
    # of data over max length: 5097
    # of data with malformed relation positions: 1959

    For Multilingual BERT,
    # of data over max length: 2480
    # of data with malformed relation positions: 1974
    """


def preprocess_dev_input(args):
    print("loading dataset...")
    with open(args.data) as f:
        lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    print("done. preprocessing starts.")

    sentences = list()
    words_queue = list()
    for line in tqdm(lines[1:]):
        if len(line) != len(lines[0]):
            sentence = " ".join(words_queue)
            sentences.append(sentence)
            words_queue = list()
            continue
        words_queue.append(line[1])
    sentences = list(set(sentences))
    save_pkl(args.save_path, sentences)
    print("\npreprocessing done.")


def preprocess_dev_gold(args):
    print("loading dataset...")
    with open(args.data) as f:
        lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    print("done. preprocessing starts.")

    f = open(args.save_path, 'w')
    words_queue = list()
    args_queue = {
        'A0': list(), 'A1': list(), 'A2': list(), 'A3': list()
    }

    for line in tqdm(lines[1:]):
        if len(line) != len(lines[0]):
            new_line = list()
            new_line.append(" ".join(words_queue))
            new_line.append(words_queue[pred_head_id])
            new_line.append(pred)
            for label in list(args_queue.keys()):
                if len(args_queue[label]) != 0:
                    new_line.append(" ".join(args_queue[label]))
            f.write("\t".join(new_line) + "\n")

            words_queue = list()
            args_queue = {
                'A0': list(), 'A1': list(), 'A2': list(), 'A3': list()
            }
            continue
        word = line[1]
        pred = line[2]
        pred_head_id = int(line[4])
        words_queue.append(word)
        for label in list(args_queue.keys()):
            if label in line[-1]:
                args_queue[label].append(word)
    f.close()


def _get_word2piece(sentence, tokenizer):
    words = sentence.replace('\xa0', ' ').split(' ')
    word2piece = {idx: list() for idx in range(len(words))}
    sentence_pieces = list()
    piece_idx = 1
    for word_idx, word in enumerate(words):
        pieces = tokenizer.tokenize(word)
        sentence_pieces += pieces
        for piece_idx_added, piece in enumerate(pieces):
            word2piece[word_idx].append(piece_idx + piece_idx_added)
        piece_idx += len(pieces)
    assert len(sentence_pieces) == piece_idx - 1
    return word2piece


def save_pkl(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)

        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--data', default='../datasets/structured_data.json')
    parser.add_argument('--save_path', default='../datasets/openie4_train.pkl')
    parser.add_argument('--max_len', type=int, default=64)
    main_args = parser.parse_args()

    if main_args.mode == 'train':
        preprocess_train(main_args)
    elif main_args.mode == 'dev_input':
        preprocess_dev_input(main_args)
    elif main_args.mode == 'dev_gold':
        preprocess_dev_gold(main_args)
    else:
        raise ValueError(f"Invalid preprocessing mode: {main_args.mode}")
