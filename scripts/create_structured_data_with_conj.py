import os
import sys
import pathlib

TOP_LEVEL_DIRECTORY = str(pathlib.Path(__file__).parent.resolve().parent.absolute())
sys.path.insert(0, TOP_LEVEL_DIRECTORY)

from thesis.models.Sentence import Sentence
from tqdm import tqdm
import argparse
import json

def sort_conj(conj):
  return sorted(conj, key=lambda x: x[0])

def main(args):
  print("loading dataset...")
  with open(args.source_path, 'r') as f:
    data = json.load(f)
  print("done. preprocessing starts.")

  not_nlp_parsable = 0
  conj_expr_overlaps = 0

  if args.limit != -1:
    iter_data = data[:args.limit]
  else:
    iter_data = data

  for step, row in tqdm(enumerate(iter_data), total=len(iter_data)):
    sentence = row['sentence']
    sent = Sentence(sentence)

    row['conj_expr_overlaps'] = []
    row['conj_expr_overlaps_with_supplements'] = []
    row['conj_term_overlaps_with_supplements'] = []
    row['all_conjunction_offsets'] = []
    row['all_conj_term_offsets'] = []
    row['all_conj_term_offsets_with_supplements'] = []

    if not sent.parsable():
      not_nlp_parsable += 1
      continue

    row['conj_expr_overlaps'] = sent.conj_expr_overlaps()
    row['conj_expr_overlaps_with_supplements'] = sent.conj_expr_overlaps_with_supplements()
    row['conj_term_overlaps_with_supplements'] = sent.conj_term_overlaps_with_supplements()

    conj_expr_overlaps_b = len(row['conj_expr_overlaps']) > 0
    conj_expr_overlaps_with_supplements_b = len(
      row['conj_expr_overlaps_with_supplements']) > 0
    conj_term_overlaps_with_supplements_b = len(row['conj_term_overlaps_with_supplements']) > 0

    if conj_expr_overlaps_b or conj_expr_overlaps_with_supplements_b or conj_term_overlaps_with_supplements_b:
      conj_expr_overlaps += 1
      print('CONJ_EXPR OVERLAP', step)

    row['all_conjunction_offsets'] = sent.conjunction_offsets()
    row['all_conj_term_offsets'] = sent.conj_term_offsets()
    row['all_conj_term_offsets_with_supplements'] = sent.conj_term_offsets_with_supplements()

  with open(args.out_path, 'w') as f:
    json.dump(data, f)

  print('file successfully stored')

  print('not not_nlp_parsable', not_nlp_parsable)
  print('conj_expr_overlaps', conj_expr_overlaps)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # settings
  parser.add_argument('--limit', type=int, default=-1)
  parser.add_argument(
    '--source_path', default='../datasets/structured_data.json')
  parser.add_argument(
    '--out_path', default='../datasets/structured_data_with_conj.json')

  main_args = parser.parse_args()

  main(main_args)
