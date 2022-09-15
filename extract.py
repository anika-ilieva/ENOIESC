import os
import torch
import numpy as np
import utils.bio as bio
from transformers import BertTokenizerFast
from tqdm import tqdm
import json


def extract_iter(args,
            model,
            loader,
            conj_mode=False):
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_config)

    for step, (offset_mapping, batch) in tqdm(enumerate(loader), desc='eval_steps', total=len(loader)):
        token_strs = [[word for word in sent] for sent in np.asarray(batch[-2]).T]
        sentences = batch[-1]
        token_ids, att_mask = map(lambda x: x.to(args.device), batch[:-2])
        batch_size = token_ids.shape[0]
        max_len = token_ids.shape[1]

        with torch.no_grad():
            """
            We will iterate B(batch_size) times
            because there are more than one predicate in one batch.
            In feeding to argument extractor, # of predicates takes a role as batch size.

            pred_logit: (B, L, 3)
            pred_hidden: (B, L, D)
            pred_tags: (B, P, L) ~ list of tensors, where P is # of predicate in each batch
            """
            pred_logit, pred_hidden = model.extract_predicate(
                input_ids=token_ids, attention_mask=att_mask)
            all_pred_tag = torch.argmax(pred_logit, 2)
            all_pred_tag = bio.filter_pred_tags(all_pred_tag, token_strs)
            pred_tags = bio.get_single_predicate_idxs(all_pred_tag)
            pred_probs = torch.nn.Softmax(2)(pred_logit)

            all_conjunction_tags = torch.ones(
                (batch_size, max_len)) * bio.all_conjunction_tag2idx['O']
            all_conj_expr_tags = torch.ones(
                (batch_size, max_len)) * bio.all_conj_expr_tag2idx['O']
            all_conj_term_tags = torch.ones(
                (batch_size, max_len)) * bio.all_conj_term_tag2idx['O']

            all_conjunction_probs = torch.ones((batch_size, max_len))
            all_conj_expr_probs = torch.ones((batch_size, max_len))
            all_conj_term_probs = torch.ones((batch_size, max_len))

            if conj_mode:
                all_conjunction_logit, conjunction_hidden = model.extract_conjunction(
                    input_ids=token_ids, attention_mask=att_mask)
                all_conjunction_tags = torch.argmax(all_conjunction_logit, 2)
                all_conjunction_tags = bio.filter_conjunction_tags(
                    all_conjunction_tags, token_strs)
                all_conjunction_probs = torch.nn.Softmax(2)(all_conjunction_logit)

                conjunction_mask = bio.get_conjunction_mask(all_conjunction_tags).to(args.device)
                if not torch.all(conjunction_mask):
                    all_conj_term_logit = model.extract_all_conj_term(
                        input_ids=token_ids, conjunction_hidden=conjunction_hidden, conjunction_mask=conjunction_mask)

                    all_conj_term_tags = torch.argmax(all_conj_term_logit, 2)
                    all_conj_term_tags = bio.filter_all_conj_term_tags(
                        all_conj_term_tags, all_pred_tag, token_strs)
                    all_conj_term_probs = torch.nn.Softmax(2)(all_conj_term_logit)

                    all_conj_expr_logit = model.extract_all_conj_expr(
                        input_ids=token_ids, conjunction_hidden=conjunction_hidden, conjunction_mask=conjunction_mask)

                    all_conj_expr_tags = torch.argmax(all_conj_expr_logit, 2)
                    all_conj_expr_tags = bio.filter_all_conj_expr_tags(
                        all_conj_expr_tags, all_pred_tag, token_strs)
                    all_conj_expr_probs = torch.nn.Softmax(2)(all_conj_expr_logit)
            
            single_conjunction_tags = bio.get_single_conjunction_idxs(
                all_conjunction_tags)
            single_conj_expr_tags = bio.get_single_conj_expr_idxs(all_conj_expr_tags)
            single_conj_term_tags = bio.get_single_conj_term_idxs(
                all_conj_term_tags)

            # iterate B times (one iteration means extraction for one sentence)
            for cur_pred_tags, cur_pred_hidden, cur_att_mask, cur_token_id, cur_all_conjunction_tags, cur_conjunction_tags, cur_conj_expr_tags, cur_conj_term_tags, cur_all_conjunction_probs, cur_all_conj_expr_probs, cur_all_conj_term_probs, cur_pred_probs, token_str, sentence, cur_offset_mapping \
                    in zip(pred_tags, pred_hidden, att_mask, token_ids, all_conjunction_tags, single_conjunction_tags, single_conj_expr_tags, single_conj_term_tags, all_conjunction_probs, all_conj_expr_probs, all_conj_term_probs, pred_probs, token_strs, sentences, offset_mapping):

                # generate temporary batch for this sentence and feed to argument module
                cur_pred_masks = bio.get_pred_mask(cur_pred_tags).to(args.device)
                n_predicates = cur_pred_masks.shape[0]
                if n_predicates == 0:
                    continue  # if there is no predicate, we cannot extract.
                cur_pred_hidden = torch.cat(n_predicates * [cur_pred_hidden.unsqueeze(0)])
                cur_token_id = torch.cat(n_predicates * [cur_token_id.unsqueeze(0)])
                cur_arg_logit = model.extract_argument(
                    input_ids=cur_token_id,
                    predicate_hidden=cur_pred_hidden,
                    predicate_mask=cur_pred_masks)

                # filter and get argument tags with highest probability
                cur_arg_tags = torch.argmax(cur_arg_logit, 2)
                cur_arg_probs = torch.nn.Softmax(2)(cur_arg_logit)
                cur_arg_tags = bio.filter_arg_tags(cur_arg_tags, cur_pred_tags, token_str)

                # get string tuples and write results
                cur_extractions, cur_extraction_idxs, cur_extraction_token_idxs, cur_conjunction_idxs, cur_conj_term_idxs, cur_conj_exprs_in_arg_idxs, pred_map = bio.get_triple(
                    sentence, cur_pred_tags, cur_arg_tags, cur_all_conjunction_tags, cur_conjunction_tags, cur_conj_expr_tags, cur_conj_term_tags, tokenizer)
                cur_confidences = bio.get_confidence_score(
                    cur_pred_probs, cur_arg_probs, cur_all_conjunction_probs, cur_all_conj_expr_probs, cur_all_conj_term_probs, cur_extraction_idxs, cur_conjunction_idxs, cur_conj_term_idxs, cur_conj_exprs_in_arg_idxs, pred_map)
                assert len(cur_confidences) == len(cur_extractions)
                for extraction, idx, confidence in zip(cur_extractions, cur_extraction_token_idxs, cur_confidences):
                    offsets = json.dumps(bio.idxToOffsets(idx, cur_offset_mapping[1:-1]))

                    yield sentence, extraction, offsets, confidence
    


def extract(args,
            model,
            loader,
            output_path,
            conj_mode=False):
    model.eval()
    os.makedirs(output_path, exist_ok=True)
    extraction_path = os.path.join(output_path, "extraction.txt")
    extraction_offset_path = os.path.join(output_path, "extraction_offsets.txt")
    f = open(extraction_path, 'w')
    f_offsets = open(extraction_offset_path, 'w')

    for sentence, extraction, offsets, confidence in extract_iter(args, model, loader, conj_mode):
        if args.binary:
            f.write(
                "\t".join([sentence] + [str(1.0)] + extraction[:3]) + '\n')
            f_offsets.write(
                "\t".join([offsets]) + '\n')
        else:
            f.write("\t".join([sentence] + [str(confidence)] + extraction) + '\n')
            f_offsets.write(
                "\t".join([offsets]) + '\n')

    f_offsets.close()
    f.close()
    print("\nExtraction Done.\n")
