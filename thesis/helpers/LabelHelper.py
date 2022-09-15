import os
import sys
import pathlib

pred_tag2idx = {
  'P-B': 0, 'P-I': 1, 'O': 2
}
arg_tag2idx = {
  'A0-B': 0, 'A0-I': 1,
  'A1-B': 2, 'A1-I': 3,
  'A2-B': 4, 'A2-I': 5,
  'A3-B': 6, 'A3-I': 7,
  'O': 8
}

all_conjunction_tag2idx = {
  'C-B': 0, 'C-I': 1, 'O': 2,
}

all_conj_expr_tag2idx = {
  'CE-B': 0, 'CE-I': 1, 'O': 2
}

all_conj_term_tag2idx = {
  'CT-B': 0, 'CT-I': 1,
  'O': 2
}

PUNCTUATION = [',', '.', '?', '!', ';', ':']


class LabelHelper:
  @staticmethod
  def pred_label_to_string(label, sent, shift=0):
    words = sent.words()
    word2piece = sent.word2piece()

    pred_labels = [pred_tag2idx['P-B'], pred_tag2idx['P-I']]
    pred_idxs = [idx for idx, tag in enumerate(
      label) if tag in pred_labels]
    pred_token_idxs = list()

    if len(pred_idxs) == 0:
      pred_str = ''
    else:
      pred_words = list()
      for word_idx, piece_idxs in word2piece.items():
        if set(piece_idxs) <= set(pred_idxs):
          pred_words.append(word_idx)
          pred_token_idxs += [idx_s for idx_s in piece_idxs]
      if len(pred_words) == 0:
        pred_str = ''
        pred_idxs = list()
      else:
        pred_str = ' '.join([words[idx] for idx in pred_words])
    if shift != 0:
      pred_idxs = [idx + shift for idx in pred_idxs]
    return pred_str, pred_idxs, pred_token_idxs

  @staticmethod
  def arg_label_to_string(label, sent, all_conjunction_idxs=[], shift=0):
    words = sent.words()
    word2piece = sent.word2piece()
    arg_strings = []
    all_arg_idxs = list()
    all_arg_token_idxs = list()

    for arg_n in range(4):
      arg_labels = [arg_tag2idx[f'A{arg_n}-B'], arg_tag2idx[f'A{arg_n}-I']]
      arg_idxs = [idx for idx, tag in enumerate(
        label) if tag in arg_labels]
      arg_token_idxs = list()
      if len(arg_idxs) == 0:
        arg_str = ''
      else:
        arg_words = list()
        for word_idx, piece_idxs in word2piece.items():
          piece_idxs_without_conjunction = set(
            piece_idxs) - set(all_conjunction_idxs)

          if piece_idxs_without_conjunction <= set(arg_idxs):
            word = words[word_idx]
            conjunction_idxs_to_remove = list(
              set(all_conjunction_idxs) & set(piece_idxs))
            if len(conjunction_idxs_to_remove) > 0:
              # remove conjunction from word
              for conjunction_idx in conjunction_idxs_to_remove:
                word = word.replace(sent.sentence_pieces()[conjunction_idx], '')
            if word != '':
              arg_words.append(word)
            if len(piece_idxs_without_conjunction) > 0:
              arg_token_idxs += sorted(
                [idx_s for idx_s in piece_idxs_without_conjunction])
        if len(arg_words) == 0:
          arg_str = ''
          arg_idxs = list()
        else:
          arg_str = ' '.join(arg_words)
      arg_strings.append(arg_str)
      all_arg_idxs.append(arg_idxs)
      all_arg_token_idxs.append(arg_token_idxs)
    if shift != 0:
      for i, idx_group in enumerate(all_arg_idxs):
        for j in range(len(idx_group)):
          all_arg_idxs[i][j] += shift
    return arg_strings, all_arg_idxs, all_arg_token_idxs

  @staticmethod
  def all_conjunction_label_to_string(label, sent):
    all_conjunction_strings = []

    all_conjunction_idxs = []
    conjunction_idxs = []

    for idx, tag in enumerate(label):
      if tag == all_conjunction_tag2idx['C-B']:
        # new conj term
        if conjunction_idxs != []:
          # add last conj term to set
          all_conjunction_idxs.append(conjunction_idxs)
        # add tokens to current conj term
        conjunction_idxs = [idx]
      elif tag == all_conjunction_tag2idx['C-I']:
        # add tokens to current conj term
        conjunction_idxs.append(idx)

    # add last conj term
    if conjunction_idxs != []:
      all_conjunction_idxs.append(conjunction_idxs)

    for all_conjunction_idx in all_conjunction_idxs:
      if len(all_conjunction_idx) == 0:
        all_conjunction_str = ''
      else:
        all_conjunction_str = [sent.sentence_pieces()[x]
                            for x in all_conjunction_idx]
      all_conjunction_strings.append(all_conjunction_str)
    return all_conjunction_strings

  @staticmethod
  def all_conj_expr_label_to_string(label, sent):
    words = sent.words()
    word2piece = sent.word2piece()
    all_conj_expr_strings = []

    all_conj_expr_idxs = []
    conj_expr_idxs = []

    for idx, tag in enumerate(label):
      if tag == all_conj_expr_tag2idx['CE-B']:
        # new conj term
        if conj_expr_idxs != []:
          # add last conj term to set
          all_conj_expr_idxs.append(conj_expr_idxs)
        # add tokens to current conj term
        conj_expr_idxs = [idx]
      elif tag == all_conj_expr_tag2idx['CE-I']:
        # add tokens to current conj term
        conj_expr_idxs.append(idx)

    # add last conj term
    if conj_expr_idxs != []:
      all_conj_expr_idxs.append(conj_expr_idxs)

    for all_conj_expr_idx in all_conj_expr_idxs:
      if len(all_conj_expr_idx) == 0:
        all_conj_expr_str = ''
      else:
        all_conj_expr_words = list()
        for word_idx, piece_idxs in word2piece.items():
          punctuation = []
          punctuation_idx_s = []
          for idx in piece_idxs:
            if sent.sentence_pieces()[idx] in PUNCTUATION:
              punctuation.append(sent.sentence_pieces()[idx])
              punctuation_idx_s.append(idx)
          if (set(piece_idxs) - set(punctuation_idx_s)) <= set(all_conj_expr_idx):
            word = words[word_idx]
            if len(set(punctuation_idx_s) & set(all_conj_expr_idx)) == 0:
              # remove punctuation from word if not in tags
              for punct in punctuation:
                word = word.replace(punct, '')
            all_conj_expr_words.append(word)
        if len(all_conj_expr_words) == 0:
          all_conj_expr_str = ''
          all_conj_expr_idx = list()
        else:
          all_conj_expr_str = ' '.join(all_conj_expr_words)
      all_conj_expr_strings.append(all_conj_expr_str)
    return all_conj_expr_strings

  @staticmethod
  def all_conj_term_label_to_string(label, sent):
    words = sent.words()
    word2piece = sent.word2piece()
    all_conj_term_strings = []

    all_conj_term_idxs = []
    conj_term_idxs = []

    for idx, tag in enumerate(label):
      if tag == all_conj_term_tag2idx['CT-B']:
        # new conj term
        if conj_term_idxs != []:
          # add last conj term to set
          all_conj_term_idxs.append(conj_term_idxs)
        # add tokens to current conj term
        conj_term_idxs = [idx]
      elif tag == all_conj_term_tag2idx['CT-I']:
        # add tokens to current conj term
        conj_term_idxs.append(idx)

    # add last conj term
    if conj_term_idxs != []:
      all_conj_term_idxs.append(conj_term_idxs)

    for all_conj_term_idxs in all_conj_term_idxs:
      if len(all_conj_term_idxs) == 0:
        all_conj_term_str = ''
      else:
        all_conj_term_words = list()
        for word_idx, piece_idxs in word2piece.items():
          punctuation = []
          punctuation_idx_s = []
          for idx in piece_idxs:
            if sent.sentence_pieces()[idx] in PUNCTUATION:
              punctuation.append(sent.sentence_pieces()[idx])
              punctuation_idx_s.append(idx)
          if (set(piece_idxs) - set(punctuation_idx_s)) <= set(all_conj_term_idxs):
            word = words[word_idx]
            if len(set(punctuation_idx_s) & set(all_conj_term_idxs)) == 0:
              # remove punctuation from word if not in tags
              for punct in punctuation:
                word = word.replace(punct, '')
            all_conj_term_words.append(word)
          
        if len(all_conj_term_words) == 0:
          all_conj_term_str = ''
          all_conj_term_idxs = list()
        else:
          all_conj_term_str = ' '.join(all_conj_term_words)
      all_conj_term_strings.append(all_conj_term_str)
    return all_conj_term_strings
