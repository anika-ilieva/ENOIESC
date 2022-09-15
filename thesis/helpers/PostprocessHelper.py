import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from thesis.models.Sentence import Sentence
from thesis.helpers.OffsetHelper import OffsetHelper

class PostprocessHelper:
  @staticmethod
  def postprocess(extraction_path, out_path, use_supplements=True):
    path = extraction_path + '/extraction.txt'
    df_extractions = pd.read_csv(path, sep="\t", header=None)
    df_extractions.columns = ["sentence", "score",
                              "predicate", "arg1", 'arg2', 'arg3', 'arg4']

    path = extraction_path + '/extraction_offsets.txt'
    df_offsets = pd.read_csv(path, sep="\t", header=None)
    df_offsets.columns = ["arg_offsets"]

    df_full = pd.merge(df_extractions, df_offsets,
                      on=df_extractions.index, how='outer')

    df_full = df_full.drop("key_0", axis=1)
    df_full["conj_expr_offsets_with_supplements"] = ""
    df_full["conj_expr_offsets"] = ""
    df_full["cc_offsets"] = ""

    for i in tqdm(range(len(df_full))):
      row = df_full.iloc[i]
      sentence = row['sentence']
      sent = Sentence(sentence)

      cc_offsets = sent.cc_offsets()
      conj_term_offsets = sent.conj_term_offsets()
      conj_term_offsets_with_supplements = sent.conj_term_offsets_with_supplements()

      df_full.at[i, "conj_expr_offsets_with_supplements"] = str(
        conj_term_offsets_with_supplements)
      df_full.at[i, "conj_expr_offsets"] = str(conj_term_offsets)
      df_full.at[i, "cc_offsets"] = str(cc_offsets)

    path = out_path + '/extraction_full.txt'
    df_full.to_csv(path, sep="\t", header=None, index=False)

    with open(out_path + '/extraction.txt', 'w') as out_file: 
      for row in tqdm(df_full.iloc, total=len(df_full)):
        sentence = row['sentence']
        score = str(row['score'])
        pred = row['predicate']
        if pred != pred:
          pred = ''
        extractions = PostprocessHelper.processRow(row, use_supplements=use_supplements)

        for extraction in extractions:
          combination = [sentence, score, pred] + extraction
          out_file.write("\t".join(combination) + '\n')
    print("Postprocessing Done.")

  @staticmethod
  def get_conj_term_map(conj_expr_offsets):
    conj_terms = []
    conj_conj_term_map = {}

    for i, conj_expr in enumerate(conj_expr_offsets):
      conj_term_ids = []
      for conj_term in conj_expr:
        conj_term_ids.append(len(conj_terms))
        conj_terms.append(conj_term)
      conj_conj_term_map[i] = conj_term_ids
    return conj_terms, conj_conj_term_map

  @staticmethod
  def get_conj_exprs_in_args(conj_expr_offsets, arg):
    conj_exprs_in_arg = []

    for i, conj_terms in enumerate(conj_expr_offsets):
      n_terms = 0

      for conj_term_parts in conj_terms:
        begin = conj_term_parts[0][0]
        end = conj_term_parts[-1][1]
        for j, (arg_begin, arg_end) in enumerate(arg):
          if begin >= arg_begin and end <= arg_end:
            n_terms += 1
      if n_terms > 1:
        conj_exprs_in_arg.append(i)
    return conj_exprs_in_arg

  @staticmethod
  def processRow(row, use_supplements=False):
    extractions = []
    sentence = row['sentence']
    sent = Sentence(sentence)

    arg_offsets = json.loads(row['arg_offsets'])[1:]
    conj_expr_offsets_with_supplements = json.loads(
      row['conj_expr_offsets_with_supplements'])
    conj_expr_offsets = json.loads(row['conj_expr_offsets'])
    if use_supplements:
      conj_expr_offsets = conj_expr_offsets_with_supplements

    conj_terms, conj_conj_term_map = PostprocessHelper.get_conj_term_map(
      conj_expr_offsets)

    char_offsets = []
    no_conj_expr_char_offsets = []

    domain = []

    for arg in arg_offsets:
      arg_char_offsets = set()
      for arg_begin, arg_end in arg:
        arg_char_offsets |= set(range(arg_begin, arg_end + 1))
      char_offsets.append(arg_char_offsets.copy())

      for conj_expr_id in PostprocessHelper.get_conj_exprs_in_args(conj_expr_offsets, arg):
        domain.append(conj_expr_id)
        first_conj_term = conj_terms[conj_conj_term_map[conj_expr_id][0]]
        last_conj_term = conj_terms[conj_conj_term_map[conj_expr_id][-1]]
        begin, end = first_conj_term[0][0], last_conj_term[-1][1]
        arg_char_offsets -= set(range(begin, end + 1))

      no_conj_expr_char_offsets.append(arg_char_offsets)

    domain = [conj_conj_term_map[conj_id] for conj_id in set(domain)]

    if len(domain) > 0:
      combinations = np.array(np.meshgrid(*domain)).T.reshape(-1, len(domain))

      for combination in combinations:
        skip_option = False
        tmp_char_offsets = [offset_set.copy()
                            for offset_set in no_conj_expr_char_offsets]
        for conj_term_id in combination:
          conj_term_is_in_arg = False
          for begin, end in conj_terms[conj_term_id]:
            conj_term_set = set(range(begin, end + 1))
            for original_arg, arg in zip(char_offsets, tmp_char_offsets):
              if conj_term_set <= original_arg:
                conj_term_is_in_arg = True
                arg |= conj_term_set
          if not conj_term_is_in_arg:
            skip_option = True
            break

        if not skip_option:
          arg_strings = []
          for arg in tmp_char_offsets:
            if arg != arg:
              arg_s = ''
            else:
              arg_s = re.sub(
                ' +', ' ', ' '.join(sent.offsets_to_strings(OffsetHelper.setToIntervals(arg)))).rstrip()
            arg_strings.append(arg_s)
          extractions.append(arg_strings)
    else:
      arg_strings = []
      for arg in char_offsets:
        if arg != arg:
          arg_s = ''
        else:
          arg_s = re.sub(
            ' +', ' ', ' '.join(sent.offsets_to_strings(OffsetHelper.setToIntervals(arg)))).rstrip()
        arg_strings.append(arg_s)
      extractions.append(arg_strings)
    return extractions
