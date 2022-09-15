import torch
import os
import sys
import pathlib
import argparse
from argparse import Namespace
from tqdm import tqdm

TOP_LEVEL_DIRECTORY = str(pathlib.Path(__file__).parent.resolve().parent.absolute())
sys.path.insert(0, TOP_LEVEL_DIRECTORY)

from benchie.benchie import Benchie
from utils import utils
from test import do_eval
from extract import extract
from utils import utils
from dataset import load_data
from thesis.helpers.PostprocessHelper import PostprocessHelper
from thesis.helpers.BenchieHelper import BenchieHelper

def benchie_eval(title, extraction_path, gold_file):
  # Load gold annotations to BenchIE
  benchie = Benchie()
  benchie.load_gold_annotations(filename=gold_file)

  # Add OIE systems extractions
  benchie.add_oie_system_extractions(
    oie_system_name="m2oie_en", filename=extraction_path + '/extraction_benchie_format.txt', silent=True)

  # Compute scores
  benchie.compute_precision()
  benchie.compute_recall()
  benchie.compute_f1()
  print(title)
  benchie.print_scores()

def load_model(args):
  model = utils.get_models(
      bert_config=args.bert_config,
      pred_n_labels=args.pred_n_labels,
      arg_n_labels=args.arg_n_labels,
      n_arg_heads=args.n_arg_heads,
      n_all_conj_expr_heads=args.n_all_conj_expr_heads,
      n_all_conj_term_heads=args.n_all_conj_term_heads,
      n_arg_layers=args.n_arg_layers,
      n_all_conj_expr_layers=args.n_all_conj_expr_layers,
      n_all_conj_term_layers=args.n_all_conj_term_layers,
      pos_emb_dim=args.pos_emb_dim,
      use_lstm=args.use_lstm,
      conj_mode=args.conj_mode,
      device=args.device)
  model.load_state_dict(torch.load(
    args.model_path, map_location=torch.device(args.device)))
  model.zero_grad()
  model.eval()
  return model

def load_test_dataset(args):
  loader = load_data(
      data_path=args.test_data_path,
      batch_size=args.batch_size,
      tokenizer_config=args.bert_config,
      train=False)
  return loader
  
def evaluation_args(args):
  # baseline
  baseline_args = Namespace()
  baseline_args.model_path = args.baseline_model_path
  baseline_args.device = args.device
  baseline_args.visible_device = args.visible_device
  baseline_args.conj_mode = False
  baseline_args = clean_config(baseline_args)

  # method2
  method2_args = Namespace()
  method2_args.model_path = args.conj_model_path
  method2_args.device = args.device
  method2_args.visible_device = args.visible_device
  method2_args.conj_mode = True
  method2_args = clean_config(method2_args)

  carb_dataset_args = Namespace()
  carb_dataset_args.test_data_path = args.test_data_carb_path
  carb_dataset_args.bert_config = baseline_args.bert_config
  carb_dataset_args.batch_size = 1

  benchie_dataset_args = Namespace()
  benchie_dataset_args.test_data_path = args.test_data_benchie_path
  benchie_dataset_args.bert_config = baseline_args.bert_config
  benchie_dataset_args.batch_size = 1

  custom_dataset_args = Namespace()
  custom_dataset_args.test_data_path = args.test_data_custom_benchie_path
  custom_dataset_args.bert_config = baseline_args.bert_config
  custom_dataset_args.batch_size = 1

  return baseline_args, method2_args, carb_dataset_args, benchie_dataset_args, custom_dataset_args


def extract_baseline_and_method1(args, baseline_args, baseline_model, carb_dataset, benchie_dataset, custom_benchie_dataset):
    # carb test baseline
  carb_test_baseline_path = args.save_path + '/carb_test_baseline'
  baseline_args.test_data_path = args.test_data_carb_path
  extract(baseline_args, baseline_model, carb_dataset,
          carb_test_baseline_path, conj_mode=baseline_args.conj_mode)

  # carb test method1_ct_sup
  carb_test_postprocess_path = args.save_path + '/carb_test_postprocess'
  os.makedirs(carb_test_postprocess_path, exist_ok=True)
  PostprocessHelper.postprocess(
    carb_test_baseline_path, carb_test_postprocess_path)

  # carb test method1_ct
  carb_test_postprocess_without_supplements_path = args.save_path + '/carb_test_postprocess_without_supplements'
  os.makedirs(carb_test_postprocess_without_supplements_path, exist_ok=True)
  PostprocessHelper.postprocess(
    carb_test_baseline_path, carb_test_postprocess_without_supplements_path, use_supplements=False)

  # benchie baseline
  benchie_baseline_path = args.save_path + '/benchie_baseline'
  baseline_args.test_data_path = args.test_data_benchie_path
  os.makedirs(benchie_baseline_path, exist_ok=True)
  extract(baseline_args, baseline_model, benchie_dataset,
          benchie_baseline_path, conj_mode=baseline_args.conj_mode)

  # benchie method1_ct_sup
  benchie_postprocess_path = args.save_path + '/benchie_postprocess'
  os.makedirs(benchie_postprocess_path, exist_ok=True)
  PostprocessHelper.postprocess(
    benchie_baseline_path, benchie_postprocess_path)

  # benchie method1_ct
  benchie_postprocess_without_supplements_path = args.save_path + '/benchie_postprocess_without_supplements'
  os.makedirs(benchie_postprocess_without_supplements_path, exist_ok=True)
  PostprocessHelper.postprocess(
    benchie_baseline_path, benchie_postprocess_without_supplements_path, use_supplements=False)

  # custom benchie baseline
  custom_benchie_baseline_path = args.save_path + '/custom_benchie_baseline'
  baseline_args.test_data_path = args.test_data_custom_benchie_path
  os.makedirs(custom_benchie_baseline_path, exist_ok=True)
  extract(baseline_args, baseline_model, custom_benchie_dataset,
          custom_benchie_baseline_path, conj_mode=baseline_args.conj_mode)

  # custom benchie method1_ct_sup
  custom_benchie_postprocess_path = args.save_path + '/custom_benchie_postprocess'
  os.makedirs(custom_benchie_postprocess_path, exist_ok=True)
  PostprocessHelper.postprocess(
    custom_benchie_baseline_path, custom_benchie_postprocess_path)

  # custom benchie method1_ct
  custom_benchie_postprocess_without_supplements_path = args.save_path + '/custom_benchie_postprocess_without_supplements'
  os.makedirs(custom_benchie_postprocess_without_supplements_path, exist_ok=True)
  PostprocessHelper.postprocess(
    custom_benchie_baseline_path, custom_benchie_postprocess_without_supplements_path, use_supplements=False)

  # custom1 benchie baseline
  custom1_benchie_baseline_path = args.save_path + '/custom1_benchie_baseline'
  baseline_args.test_data_path = args.test_data_custom_benchie_path
  os.makedirs(custom1_benchie_baseline_path, exist_ok=True)
  extract(baseline_args, baseline_model, custom_benchie_dataset,
          custom1_benchie_baseline_path, conj_mode=baseline_args.conj_mode)

  # custom1 benchie method1_ct_sup
  custom1_benchie_postprocess_path = args.save_path + '/custom1_benchie_postprocess'
  os.makedirs(custom1_benchie_postprocess_path, exist_ok=True)
  PostprocessHelper.postprocess(
    custom1_benchie_baseline_path, custom1_benchie_postprocess_path)

  # custom1 benchie method1
  custom1_benchie_postprocess_without_supplements_path = args.save_path + '/custom1_benchie_postprocess_without_supplements'
  os.makedirs(custom1_benchie_postprocess_without_supplements_path, exist_ok=True)
  PostprocessHelper.postprocess(
    custom1_benchie_baseline_path, custom1_benchie_postprocess_without_supplements_path, use_supplements=False)

  # custom2 benchie baseline
  custom2_benchie_baseline_path = args.save_path + '/custom2_benchie_baseline'
  baseline_args.test_data_path = args.test_data_custom_benchie_path
  os.makedirs(custom2_benchie_baseline_path, exist_ok=True)
  extract(baseline_args, baseline_model, custom_benchie_dataset,
          custom2_benchie_baseline_path, conj_mode=baseline_args.conj_mode)

  # custom2 benchie method1_ct_sup
  custom2_benchie_postprocess_path = args.save_path + '/custom2_benchie_postprocess'
  os.makedirs(custom2_benchie_postprocess_path, exist_ok=True)
  PostprocessHelper.postprocess(
    custom2_benchie_baseline_path, custom2_benchie_postprocess_path)

  # custom2 benchie method1
  custom2_benchie_postprocess_without_supplements_path = args.save_path + '/custom2_benchie_postprocess_without_supplements'
  os.makedirs(custom2_benchie_postprocess_without_supplements_path, exist_ok=True)
  PostprocessHelper.postprocess(
    custom2_benchie_baseline_path, custom2_benchie_postprocess_without_supplements_path, use_supplements=False)

def extract_method2(args, method2_args, method2_model,
                    carb_dataset, benchie_dataset, custom_benchie_dataset):
  # carb test method2
  carb_test_method2_path = args.save_path + '/carb_test_method2'
  method2_args.test_data_path = args.test_data_carb_path
  extract(method2_args, method2_model, carb_dataset,
          carb_test_method2_path, conj_mode=method2_args.conj_mode)

  # benchie method2
  benchie_method2_path = args.save_path + '/benchie_method2'
  method2_args.test_data_path = args.test_data_benchie_path
  os.makedirs(benchie_method2_path, exist_ok=True)
  extract(method2_args, method2_model, benchie_dataset,
          benchie_method2_path, conj_mode=method2_args.conj_mode)

  # custom benchie method2
  custom_benchie_method2_path = args.save_path + '/custom_benchie_method2'
  method2_args.test_data_path = args.test_data_custom_benchie_path
  os.makedirs(custom_benchie_method2_path, exist_ok=True)
  extract(method2_args, method2_model, custom_benchie_dataset,
          custom_benchie_method2_path, conj_mode=method2_args.conj_mode)

  # custom1 benchie method2
  custom1_benchie_method2_path = args.save_path + '/custom1_benchie_method2'
  method2_args.test_data_path = args.test_data_custom_benchie_path
  os.makedirs(custom1_benchie_method2_path, exist_ok=True)
  extract(method2_args, method2_model, custom_benchie_dataset,
          custom1_benchie_method2_path, conj_mode=method2_args.conj_mode)

  # custom2 benchie method2
  custom2_benchie_method2_path = args.save_path + '/custom2_benchie_method2'
  method2_args.test_data_path = args.test_data_custom_benchie_path
  os.makedirs(custom2_benchie_method2_path, exist_ok=True)
  extract(method2_args, method2_model, custom_benchie_dataset,
          custom2_benchie_method2_path, conj_mode=method2_args.conj_mode)


def extract_all(args, baseline_args, method2_args, carb_dataset_args, benchie_dataset_args, custom_dataset_args):
  carb_dataset = load_test_dataset(carb_dataset_args)
  benchie_dataset = load_test_dataset(benchie_dataset_args)
  custom_benchie_dataset = load_test_dataset(custom_dataset_args)

  baseline_model = load_model(baseline_args)

  method2_model = load_model(method2_args)

  extract_baseline_and_method1(args, baseline_args, baseline_model,
                   carb_dataset, benchie_dataset, custom_benchie_dataset)

  extract_method2(args, method2_args, method2_model,
                  carb_dataset, benchie_dataset, custom_benchie_dataset)


def clean_config(config):
  device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
  config.device = device
  config.pred_n_labels = 3
  config.arg_n_labels = 9
  config.conjunction_n_labels = 3
  config.all_conj_expr_n_labels = 3
  config.all_conj_term_n_labels = 3

  config.bert_config = 'bert-base-cased'

  config.pos_emb_dim = 64

  config.n_arg_heads = 8
  config.n_arg_layers = 4

  config.n_all_conj_expr_heads = 8
  config.n_all_conj_expr_layers = 4

  config.n_all_conj_term_heads = 8
  config.n_all_conj_term_layers = 4

  config.use_lstm = False
  config.binary = False

  return config

def evaluate_carb(args):
  baseline_path = args.save_path + '/carb_test_baseline'
  baseline_carb_test_results = do_eval(baseline_path, args.test_gold_path_carb)

  postprocess_path = args.save_path + '/carb_test_postprocess'
  baseline_postprocess_carb_test_results = do_eval(postprocess_path, args.test_gold_path_carb)

  postprocess_without_supplements_path = args.save_path + '/carb_test_postprocess_without_supplements'
  baseline_postprocess_without_supplements_carb_test_results = do_eval(postprocess_without_supplements_path, args.test_gold_path_carb)
  
  method2_path = args.save_path + '/carb_test_method2'
  method2_carb_test_results = do_eval(method2_path, args.test_gold_path_carb)
  
  utils.print_results("CARB BASELINE TEST RESULT", baseline_carb_test_results, [
                      "F1  ", "PREC", "REC ", "AUC "])

  utils.print_results("CARB METHOD1_CT TEST RESULT", baseline_postprocess_without_supplements_carb_test_results, [
                      "F1  ", "PREC", "REC ", "AUC "])

  utils.print_results("CARB METHOD1_CT_SUP TEST RESULT", baseline_postprocess_carb_test_results, [
                      "F1  ", "PREC", "REC ", "AUC "])

  utils.print_results("CARB METHOD2 TEST RESULT", method2_carb_test_results, [
                      "F1  ", "PREC", "REC ", "AUC "])

def evaluate_benchie(args):
  benchie_baseline_path = args.save_path + '/benchie_baseline'
  BenchieHelper.fix_format(benchie_baseline_path, args.test_data_benchie_path)
  benchie_postprocess_path = args.save_path + '/benchie_postprocess'
  BenchieHelper.fix_format(benchie_postprocess_path, args.test_data_benchie_path)
  benchie_postprocess_without_supplements_path = args.save_path + '/benchie_postprocess_without_supplements'
  BenchieHelper.fix_format(benchie_postprocess_without_supplements_path, args.test_data_benchie_path)
  benchie_method2_path = args.save_path + '/benchie_method2'
  BenchieHelper.fix_format(benchie_method2_path, args.test_data_benchie_path)

  custom_benchie_baseline_path = args.save_path + '/custom_benchie_baseline'
  BenchieHelper.fix_format(custom_benchie_baseline_path, args.test_data_custom_benchie_path)
  custom_benchie_postprocess_path = args.save_path + '/custom_benchie_postprocess'
  BenchieHelper.fix_format(custom_benchie_postprocess_path, args.test_data_custom_benchie_path)
  custom_benchie_postprocess_without_supplements_path = args.save_path + '/custom_benchie_postprocess_without_supplements'
  BenchieHelper.fix_format(custom_benchie_postprocess_without_supplements_path, args.test_data_custom_benchie_path)
  custom_benchie_method2_path = args.save_path + '/custom_benchie_method2'
  BenchieHelper.fix_format(custom_benchie_method2_path, args.test_data_custom_benchie_path)

  custom1_benchie_baseline_path = args.save_path + '/custom1_benchie_baseline'
  BenchieHelper.fix_format(custom1_benchie_baseline_path, args.test_data_custom_benchie_path)
  custom1_benchie_postprocess_path = args.save_path + '/custom1_benchie_postprocess'
  BenchieHelper.fix_format(custom1_benchie_postprocess_path, args.test_data_custom_benchie_path)
  custom1_benchie_postprocess_without_supplements_path = args.save_path + '/custom1_benchie_postprocess_without_supplements'
  BenchieHelper.fix_format(custom1_benchie_postprocess_without_supplements_path, args.test_data_custom_benchie_path)
  custom1_benchie_method2_path = args.save_path + '/custom1_benchie_method2'
  BenchieHelper.fix_format(custom1_benchie_method2_path, args.test_data_custom_benchie_path)

  custom2_benchie_baseline_path = args.save_path + '/custom2_benchie_baseline'
  BenchieHelper.fix_format(custom2_benchie_baseline_path, args.test_data_custom_benchie_path)
  custom2_benchie_postprocess_path = args.save_path + '/custom2_benchie_postprocess'
  BenchieHelper.fix_format(custom2_benchie_postprocess_path, args.test_data_custom_benchie_path)
  custom2_benchie_postprocess_without_supplements_path = args.save_path + '/custom2_benchie_postprocess_without_supplements'
  BenchieHelper.fix_format(custom2_benchie_postprocess_without_supplements_path, args.test_data_custom_benchie_path)
  custom2_benchie_method2_path = args.save_path + '/custom2_benchie_method2'
  BenchieHelper.fix_format(custom2_benchie_method2_path, args.test_data_custom_benchie_path)

  benchie_eval("Benchie BASELINE", benchie_baseline_path,
               args.test_gold_path_benchie)
  benchie_eval("Benchie CUSTOM BASELINE", custom_benchie_baseline_path,
               args.test_gold_path_custom_benchie)
  benchie_eval("Benchie CUSTOM_1 BASELINE", custom1_benchie_baseline_path,
               args.test_gold_path_custom1_benchie)
  benchie_eval("Benchie CUSTOM_2 BASELINE", custom2_benchie_baseline_path,
               args.test_gold_path_custom2_benchie)

  benchie_eval("Benchie METHOD1_CT", benchie_postprocess_without_supplements_path,
               args.test_gold_path_benchie)
  benchie_eval("Benchie CUSTOM METHOD1_CT", custom_benchie_postprocess_without_supplements_path,
               args.test_gold_path_custom_benchie)
  benchie_eval("Benchie CUSTOM_1 METHOD1_CT", custom1_benchie_postprocess_without_supplements_path,
               args.test_gold_path_custom1_benchie)
  benchie_eval("Benchie CUSTOM_2 METHOD1_CT", custom2_benchie_postprocess_without_supplements_path,
               args.test_gold_path_custom2_benchie)

  benchie_eval("Benchie METHOD1_CT_SUP", benchie_postprocess_path,
               args.test_gold_path_benchie)
  benchie_eval("Benchie CUSTOM METHOD1_CT_SUP", custom_benchie_postprocess_path,
               args.test_gold_path_custom_benchie)
  benchie_eval("Benchie CUSTOM_1 METHOD1_CT_SUP", custom1_benchie_postprocess_path,
               args.test_gold_path_custom1_benchie)
  benchie_eval("Benchie CUSTOM_2 METHOD1_CT_SUP", custom2_benchie_postprocess_path,
               args.test_gold_path_custom2_benchie)

  benchie_eval("Benchie METHOD2", benchie_method2_path,
               args.test_gold_path_benchie)
  benchie_eval("Benchie CUSTOM METHOD2", custom_benchie_method2_path,
               args.test_gold_path_custom_benchie)
  benchie_eval("Benchie CUSTOM_1 METHOD2", custom1_benchie_method2_path,
               args.test_gold_path_custom1_benchie)
  benchie_eval("Benchie CUSTOM_2 METHOD2", custom2_benchie_method2_path,
               args.test_gold_path_custom2_benchie)


def evaluate_all(args):
  evaluate_benchie(args)
  evaluate_carb(args)

def main(args):
  baseline_args, method2_args, carb_dataset_args, benchie_dataset_args, custom_dataset_args = evaluation_args(args)
  
  if not args.skip_extraction:
    extract_all(args, baseline_args, method2_args, carb_dataset_args, benchie_dataset_args, custom_dataset_args)
  evaluate_all(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # settings
  parser.add_argument(
      '--baseline_model_path', default='../results/original/baseline_model.bin')
  parser.add_argument(
      '--conj_model_path', default='../results/conj_with_score/conj_model.bin')

  parser.add_argument('--save_path', default='../results/evaluation')

  parser.add_argument('--test_data_benchie_path', default='../datasets/sample300_en.pkl')
  parser.add_argument('--test_data_custom_benchie_path', default='../datasets/sampleCustom_en.pkl')
  parser.add_argument('--test_data_carb_path', default='../datasets/carb_test.pkl')

  parser.add_argument('--test_gold_path_carb', default='../carb/CaRB_test.tsv')
  parser.add_argument('--test_gold_path_benchie', default='../benchie/gold/benchie_gold_annotations_en.txt')
  parser.add_argument('--test_gold_path_custom_benchie', default='../benchie/gold/benchie_gold_annotations_custom.txt')
  parser.add_argument('--test_gold_path_custom1_benchie', default='../benchie/gold/benchie_gold_annotations_custom1.txt')
  parser.add_argument('--test_gold_path_custom2_benchie', default='../benchie/gold/benchie_gold_annotations_custom2.txt')

  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--visible_device', default="0")

  parser.add_argument("--skip_extraction", action='store_true')

  main_args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = main_args.visible_device

  os.makedirs(main_args.save_path, exist_ok=True)

  main(main_args)
