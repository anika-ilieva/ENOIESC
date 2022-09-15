import os
import torch
import torch.nn as nn
import utils.bio as bio
from tqdm import tqdm
from extract import extract
from utils import utils
from test import do_eval


def train(args,
          epoch,
          model,
          trn_loader,
          dev_loaders,
          summarizer,
          optimizer,
          scheduler,
          conj_mode):
    total_pred_loss, total_arg_loss, total_conjunction_loss ,total_conj_expr_loss, total_conj_term_loss, trn_results = 0, 0, 0, 0, 0, None
    epoch_steps = int(args.total_steps / args.epochs)

    iterator = tqdm(enumerate(trn_loader), desc='steps', total=epoch_steps)
    for step, batch in iterator:
        batch = map(lambda x: x.to(args.device), batch)
        if conj_mode:
            token_ids, att_mask, single_pred_label, single_arg_label, all_pred_label, conjunction_label, all_conj_expr_label, all_conj_term_label = batch
            conjunction_mask = bio.get_conjunction_mask(conjunction_label)
        else:
            token_ids, att_mask, single_pred_label, single_arg_label, all_pred_label = batch
        pred_mask = bio.get_pred_mask(single_pred_label) # predicate label to mask converter

        model.train()
        model.zero_grad()

        # feed to predicate model
        if conj_mode:
            batch_loss, pred_loss, arg_loss, conjunction_loss, all_conj_expr_loss, all_conj_term_loss = model(
                input_ids=token_ids,
                attention_mask=att_mask,
                predicate_mask=pred_mask,
                total_pred_labels=all_pred_label,
                arg_labels=single_arg_label,
                conjunction_mask=conjunction_mask,
                conjunction_label=conjunction_label,
                all_conj_expr_label=all_conj_expr_label,
                all_conj_term_label=all_conj_term_label)
            total_conjunction_loss += conjunction_loss.item()
            total_conj_expr_loss += all_conj_expr_loss.item()
            total_conj_term_loss += all_conj_term_loss.item()
        else:
            batch_loss, pred_loss, arg_loss = model(
                input_ids=token_ids,
                attention_mask=att_mask,
                predicate_mask=pred_mask,
                total_pred_labels=all_pred_label,
                arg_labels=single_arg_label)

        # get performance on this batch
        total_pred_loss += pred_loss.item()
        total_arg_loss += arg_loss.item()

        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if conj_mode:
            trn_results = [total_pred_loss / (step + 1), total_arg_loss / (step + 1), total_conjunction_loss / (step + 1), total_conj_expr_loss / (step + 1), total_conj_term_loss / (step + 1)]
        else:
            trn_results = [total_pred_loss / (step + 1), total_arg_loss / (step + 1)]
        if step > epoch_steps:
            break

        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print('Free:', f)


        # interim evaluation
        if step % 1000 == 0 and step != 0:
            dev_iter = zip(args.dev_data_path, args.dev_gold_path, dev_loaders)
            dev_results = list()
            total_sum = 0
            for dev_input, dev_gold, dev_loader in dev_iter:
                dev_name = dev_input.split('/')[-1].replace('.pkl', '')
                output_path = os.path.join(args.save_path, f'epoch{epoch}_dev/step{step}/{dev_name}')
                extract(args, model, dev_loader, output_path, conj_mode=conj_mode)
                dev_result = do_eval(output_path, dev_gold)
                utils.print_results(f"EPOCH{epoch} STEP{step} EVAL",
                                    dev_result, ["F1  ", "PREC", "REC ", "AUC "])
                total_sum += dev_result[0] + dev_result[-1]
                dev_result.append(dev_result[0] + dev_result[-1])
                dev_results += dev_result
            summarizer.save_results([step] + trn_results + dev_results + [total_sum])
            model_name = utils.set_model_name(total_sum, epoch, step)
            torch.save(model.state_dict(), os.path.join(args.save_path, model_name))

        if step % args.summary_step == 0 and step != 0:
            if conj_mode:
                utils.print_results(f"EPOCH{epoch} STEP{step} TRAIN",
                                    trn_results, ["PRED LOSS", "ARG LOSS ", "CONJUNCTION LOSS", "CONJ_EXPR LOSS", "CONJ_TERM LOSS"])
            else:
                utils.print_results(f"EPOCH{epoch} STEP{step} TRAIN",
                                    trn_results, ["PRED LOSS", "ARG LOSS "])
    # end epoch summary
    if conj_mode:
        utils.print_results(f"EPOCH{epoch} TRAIN",
                            trn_results, ["PRED LOSS", "ARG LOSS ", "CONJUNCTION LOSS", "CONJ_EXPR LOSS", "CONJ_TERM LOSS"])
    else:
        utils.print_results(f"EPOCH{epoch} STEP{step} TRAIN",
                            trn_results, ["PRED LOSS", "ARG LOSS "])
    return trn_results

