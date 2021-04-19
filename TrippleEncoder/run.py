# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import

import argparse
import logging
import os
import random
from io import open
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from bleu import _bleu
from model import TrippleEncoderModel

# from knockknock import email_sender

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source1,
                 source2,
                 source3,
                 target,
                 ):
        self.idx = idx
        self.source1 = source1
        self.source2 = source2
        self.source3 = source3
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 4
    src1_filename = filename.split(',')[0]
    src2_filename = filename.split(',')[1]
    src3_filename = filename.split(',')[2]
    trg_filename = filename.split(',')[3]
    idx = 0
    with open(src1_filename) as f1, open(src2_filename) as f2, open(src3_filename) as f3, open(trg_filename) as f4:
        for line1, line2, line3, line4 in zip(f1, f2, f3, f4):
            examples.append(
                Example(
                    idx=idx,
                    source1=line1.strip(),
                    source2=line2.strip(),
                    source3=line3.strip(),
                    target=line4.strip(),
                )
            )
            idx += 1
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids1,
                 source_ids2,
                 source_ids3,
                 target_ids,
                 source_mask1,
                 source_mask2,
                 source_mask3,
                 target_mask,
        ):
        self.example_id = example_id
        self.source_ids1 = source_ids1
        self.source_ids2 = source_ids2
        self.source_ids3 = source_ids3
        self.target_ids = target_ids
        self.source_mask1 = source_mask1
        self.source_mask2 = source_mask2
        self.source_mask3 = source_mask3
        self.target_mask = target_mask


def get_max_lengths(examples, tokenizer, args):
    s1l, s2l, s3l = 0, 0, 0
    l1, l2, l3 = [], [], []
    for example_index, example in enumerate(tqdm(examples)):
        # source1
        source_tokens1 = tokenizer.tokenize(example.source1)
        l1.append(len(source_tokens1))
        # source2
        source_tokens2 = tokenizer.tokenize(example.source2)
        l2.append(len(source_tokens2))
        # source 3
        source_tokens3 = tokenizer.tokenize(example.source3)
        l3.append(len(source_tokens3))
        pass
    a, b, c = min(max(l1) + 2, args.max_source_length), \
           min(max(l2) + 2, args.max_source_length), \
           min(max(l3) + 2, args.max_source_length)
    print('Source1 Max Length : \t', a)
    print('Source2 Max Length : \t', b)
    print('Source3 Max Length : \t', c)
    print('=' * 100)
    return a, b, c


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    l1, l2, l3 = get_max_lengths(examples, tokenizer, args)
    for example_index, example in enumerate(tqdm(examples)):
        # source1
        source_tokens1 = tokenizer.tokenize(example.source1)[:l1 - 2]
        source_tokens1 = [tokenizer.cls_token] + source_tokens1 + [tokenizer.sep_token]
        source_ids1 = tokenizer.convert_tokens_to_ids(source_tokens1)
        source_mask1 = [1] * (len(source_tokens1))
        padding_length1 = l1 - len(source_ids1)
        source_ids1 += [tokenizer.pad_token_id] * padding_length1
        source_mask1 += [0] * padding_length1

        # source2
        source_tokens2 = tokenizer.tokenize(example.source2)[:(l2 - 2)]
        source_tokens2 = [tokenizer.cls_token] + source_tokens2 + [tokenizer.sep_token]
        source_ids2 = tokenizer.convert_tokens_to_ids(source_tokens2)
        source_mask2 = [1] * (len(source_tokens2))
        padding_length2 = l2 - len(source_ids2)
        source_ids2 += [tokenizer.pad_token_id] * padding_length2
        source_mask2 += [0] * padding_length2

        # source 3
        source_tokens3 = tokenizer.tokenize(example.source3)[:(l2 - 2)]
        source_tokens3 = [tokenizer.cls_token] + source_tokens3 + [tokenizer.sep_token]
        source_ids3 = tokenizer.convert_tokens_to_ids(source_tokens3)
        source_mask3 = [1] * (len(source_tokens3))
        padding_length3 = l2 - len(source_ids3)
        source_ids3 += [tokenizer.pad_token_id] * padding_length3
        source_mask3 += [0] * padding_length3

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(
            InputFeatures(
                example_index,
                source_ids1=source_ids1,
                source_ids2=source_ids2,
                source_ids3=source_ids3,
                target_ids=target_ids,
                source_mask1=source_mask1,
                source_mask2=source_mask2,
                source_mask3=source_mask3,
                target_mask=target_mask
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# @email_sender(recipient_emails=["saikatc@cs.columbia.edu"], sender_email="saikatch107@gmail.com")
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--tokenizer_name", default="", required=True,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--max_patience', default=5, type=int)
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    # budild model
    encoder1 = model_class.from_pretrained(args.model_name_or_path, config=config)
    encoder2 = model_class.from_pretrained(args.model_name_or_path, config=config)
    encoder3 = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = TrippleEncoderModel(encoder1=encoder1, encoder2=encoder2, encoder3=encoder3, decoder=decoder, config=config,
                             beam_size=args.beam_size, max_length=args.max_target_length,
                             sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        logger.info(model)
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids1 = torch.tensor([f.source_ids1 for f in train_features], dtype=torch.long)
        all_source_mask1 = torch.tensor([f.source_mask1 for f in train_features], dtype=torch.long)
        all_source_ids2 = torch.tensor([f.source_ids2 for f in train_features], dtype=torch.long)
        all_source_mask2 = torch.tensor([f.source_mask2 for f in train_features], dtype=torch.long)
        all_source_ids3 = torch.tensor([f.source_ids3 for f in train_features], dtype=torch.long)
        all_source_mask3 = torch.tensor([f.source_mask3 for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids1, all_source_mask1, all_source_ids2, all_source_mask2,
                                   all_source_ids3, all_source_mask3, all_target_ids,
                                   all_target_mask)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)
        if args.num_train_epochs > 0:
            num_train_optimization_steps = int(np.ceil(args.num_train_epochs * len(train_examples) / (
                        args.train_batch_size // args.gradient_accumulation_steps)))
            eval_steps = len(train_examples) // args.train_batch_size
            pass
        else:
            num_train_optimization_steps = args.train_steps * args.gradient_accumulation_steps
            eval_steps = args.eval_steps
        logger.info('Max Train Steps : %d\nEval Steps : %d' % (num_train_optimization_steps, eval_steps))
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size // (
                    len(train_examples) * args.gradient_accumulation_steps))

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        bar = range(num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        patience_counter = 0
        for _ in tqdm(bar):
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids1, source_mask1, source_ids2, source_mask2, \
            source_ids3, source_mask3, target_ids, target_mask = batch
            loss, _, _ = model(source_ids1=source_ids1, source_mask1=source_mask1,
                               source_ids2=source_ids2, source_mask2=source_mask2,
                               source_ids3=source_ids3, source_mask3=source_mask3,
                               target_ids=target_ids, target_mask=target_mask)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            # if (global_step + 1) % 100 == 0:
            #     logger.info("  step {} loss {}".format(global_step + 1, train_loss))
            nb_tr_examples += source_ids1.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % eval_steps == 0) and eval_flag:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    all_source_ids1 = torch.tensor([f.source_ids1 for f in eval_features], dtype=torch.long)
                    all_source_mask1 = torch.tensor([f.source_mask1 for f in eval_features], dtype=torch.long)
                    all_source_ids2 = torch.tensor([f.source_ids2 for f in eval_features], dtype=torch.long)
                    all_source_mask2 = torch.tensor([f.source_mask2 for f in eval_features], dtype=torch.long)
                    all_source_ids3 = torch.tensor([f.source_ids3 for f in eval_features], dtype=torch.long)
                    all_source_mask3 = torch.tensor([f.source_mask3 for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids1, all_source_mask1, all_source_ids2, all_source_mask2,
                                              all_source_ids3, all_source_mask3, all_target_ids, all_target_mask)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in tqdm(eval_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids1_eval, source_mask1_eval, source_ids2_eval, source_maks2_eval, \
                    source_ids3_eval, source_maks3_eval, target_ids_eval, target_mask_eval = batch

                    with torch.no_grad():
                        _, loss, num = model(source_ids1=source_ids1_eval, source_mask1=source_mask1_eval,
                                             source_ids2=source_ids2_eval, source_mask2=source_maks2_eval,
                                             source_ids3=source_ids3_eval, source_mask3=source_maks3_eval,
                                             target_ids=target_ids_eval, target_mask=target_mask_eval)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                    # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples, min(500, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    all_source_ids1 = torch.tensor([f.source_ids1 for f in eval_features], dtype=torch.long)
                    all_source_mask1 = torch.tensor([f.source_mask1 for f in eval_features], dtype=torch.long)
                    all_source_ids2 = torch.tensor([f.source_ids2 for f in eval_features], dtype=torch.long)
                    all_source_mask2 = torch.tensor([f.source_mask2 for f in eval_features], dtype=torch.long)
                    all_source_ids3 = torch.tensor([f.source_ids3 for f in eval_features], dtype=torch.long)
                    all_source_mask3 = torch.tensor([f.source_mask3 for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids1, all_source_mask1,
                                              all_source_ids2, all_source_mask2,
                                              all_source_ids3, all_source_mask3)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                for batch in tqdm(eval_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids1_dev, source_mask1_dev, source_ids2_dev, \
                    source_mask2_dev, source_ids3_dev, source_mask3_dev = batch
                    with torch.no_grad():
                        preds = model(source_ids1=source_ids1_dev, source_mask1=source_mask1_dev,
                                      source_ids2=source_ids2_dev, source_mask2=source_mask2_dev,
                                      source_ids3=source_ids3_dev, source_mask3=source_mask3_dev,)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                accs = []
                with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
                        os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx) + '\t' + ref)
                        f.write(ref + '\n')
                        f1.write(gold.target + '\n')
                        accs.append(ref == gold.target)

                dev_bleu = round(
                    _bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")), 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  %s = %s " % ("xMatch", str(round(np.mean(accs) * 100, 4))))
                logger.info("  " + "*" * 20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    patience_counter = 0
                    pass
                else:
                    patience_counter += 1
                    pass
            if patience_counter == args.max_patience:
                logger.info("Met Early Stopping Criterion: Stop Training!")
                break
                pass
            pass


    if args.do_test:
        files = []
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids1 = torch.tensor([f.source_ids1 for f in eval_features], dtype=torch.long)
            all_source_mask1 = torch.tensor([f.source_mask1 for f in eval_features], dtype=torch.long)
            all_source_ids2 = torch.tensor([f.source_ids2 for f in eval_features], dtype=torch.long)
            all_source_mask2 = torch.tensor([f.source_mask2 for f in eval_features], dtype=torch.long)
            all_source_ids3 = torch.tensor([f.source_ids3 for f in eval_features], dtype=torch.long)
            all_source_mask3 = torch.tensor([f.source_mask3 for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids1, all_source_mask1, all_source_ids2,
                                      all_source_mask2, all_source_ids3, all_source_mask3)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids1, source_mask1, source_ids2, source_mask2, source_ids3, source_mask3 = batch
                with torch.no_grad():
                    preds = model(source_ids1=source_ids1, source_mask1=source_mask1, source_ids2=source_ids2,
                                  source_mask2=source_mask2, source_ids3=source_ids3, source_mask3=source_mask3)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            predictions = []
            accs = []
            with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(
                    os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
                for ref, gold in zip(p, eval_examples):
                    predictions.append(str(gold.idx) + '\t' + ref)
                    f.write(ref + '\n')
                    f1.write(gold.target + '\n')
                    accs.append(ref == gold.target)
            dev_bleu = round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file),
                                   os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)), 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  %s = %s " % ("xMatch", str(round(np.mean(accs) * 100, 4))))
            logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()
