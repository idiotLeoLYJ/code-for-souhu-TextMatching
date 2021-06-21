# coding:utf-8
# Author:Yuanjie Liu (CamilleLeo)

import argparse
import glob
import json
import logging
import os
from utils import utils
from functools import partial
import random
from tqdm import tqdm, trange
from sklearn.metrics import f1_score

import numpy as np
import torch

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from multiprocessing import Pool
from models.bert import BertTextClassification
from models.bert_average import BertTextClassificationWithAveragePooling
from models.bert_cls_avg import BertTextClassificationWithClsAveragePooling
from models.bert_ncls import BertTextClassificationNCls
from models.electra import ElectraTextClassification
from models.electra_avg import ElectraTextClassificationWithAveragePooling
from models.electra_cls_avg import ElectraTextClassificationWithClsAveragePooling

from utils.pgd import PGD
from utils.ema import EMA
from utils.fgm import FGM

from transformers import BertTokenizer, ElectraTokenizer

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    'bert': {
        'cls': BertTextClassification,
        'average': BertTextClassificationWithAveragePooling,
        'bert_cls_avg': BertTextClassificationWithClsAveragePooling,
        'bert_ncls': BertTextClassificationNCls,

    },
    'roberta-wwm-ext-large': {
        'cls': BertTextClassification,
        'average': BertTextClassificationWithAveragePooling,
        'bert_cls_avg': BertTextClassificationWithClsAveragePooling,
        'bert_ncls': BertTextClassificationNCls,
    },
    'electra-large': {
        'cls': ElectraTextClassification,
        'average': ElectraTextClassificationWithAveragePooling,
        'bert_cls_avg': ElectraTextClassificationWithClsAveragePooling,
    },
    'electra-base': {
        'cls': ElectraTextClassification,
    },
    'mac-bert': {
        'cls': BertTextClassification,
        'average': BertTextClassificationWithAveragePooling,
        'bert_cls_avg': BertTextClassificationWithClsAveragePooling,
        'bert_ncls': BertTextClassificationNCls,
    }
}
TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'roberta-wwm-ext-large': BertTokenizer,
    'electra-large': ElectraTokenizer,
    'electra-base': ElectraTokenizer,
    'mac-bert': BertTokenizer,
}
MODEL_PATHS = {
    'bert': 'model_hub/bert-base-chinese',
    'roberta-wwm-ext-large': 'model_hub/chinese-roberta-wwm-ext-large',
    'electra-large': 'model_hub/chinese-electra-180g-large-discriminator',
    'electra-base': 'model_hub/chinese-electra-180g-base-discriminator',
    'mac-bert': 'model_hub/chinese-macbert-large'
}
CACHE_NAME = {
    'bert': './cached_data/cached_bert_tokenizer_data',
    'electra': './cached_data/cached_electra_tokenizer_data',
    'roberta-wwm-ext-large': './cached_data/cached_bert_tokenizer_data',
    'mac-bert': './cached_data/cached_bert_tokenizer_data',
}
DATA_DIR = {
    'A': 'merged_data/all_train_data_label_A.json',
    'B': 'merged_data/all_train_data_label_B.json',
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def tokenizer_single(example, args, tokenizer):
    example_encoding = tokenizer(
        [(example[1], example[2])],
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
    )
    if args.use_ghmloss or args.use_focalloss:
        temp = [0., 1.] if float(example[3]) == 1. else [1., 0.]
        example_encoding['label'] = temp
    else:
        example_encoding['label'] = example[3]
    return example_encoding


def load_dataset(args, tokenizer):

    cached_features_file_train = \
        "{}_{}_{}_{}".format(
            CACHE_NAME[args.model_type],
            args.task_name,
            'train',
            args.seed,
        )
    cached_features_file_valid = \
        "{}_{}_{}_{}".format(
            CACHE_NAME[args.model_type],
            args.task_name,
            'valid',
            args.seed,
        )

    if os.path.exists(cached_features_file_train) and os.path.exists(cached_features_file_valid):
        print("Loading features from cached file %s", cached_features_file_train)
        train_features = torch.load(cached_features_file_train)

        print("Loading features from cached file %s", cached_features_file_valid)
        test_features = torch.load(cached_features_file_valid)
    else:
        data = utils.load_json(DATA_DIR[args.task_name])
        features = []
        for example in data:
            features.append(tokenizer_single(example, args, tokenizer))
        train_features, test_features = train_test_split(features, test_size=args.test_size, random_state=args.seed)

        print("Saving features into cached file")
        torch.save(train_features, cached_features_file_train)
        torch.save(test_features, cached_features_file_valid)

    # Convert to Tensors and build dataset
    # TRAIN
    all_input_ids_train = torch.tensor([f.input_ids[0] for f in train_features], dtype=torch.long)
    all_attention_mask_train = torch.tensor([f.attention_mask[0] for f in train_features], dtype=torch.long)
    if train_features[0].token_type_ids is None:
        # For RoBERTa (a potential bug!)
        all_token_type_ids_train = torch.tensor([[0] * args.max_seq_length for f in train_features], dtype=torch.long)
    else:
        all_token_type_ids_train = torch.tensor([f.token_type_ids[0] for f in train_features], dtype=torch.long)
    if args.use_ghmloss or args.use_focalloss:
        all_labels_train = torch.tensor([f.label for f in train_features], dtype=torch.float)
    else:
        all_labels_train = torch.tensor([f.label for f in train_features], dtype=torch.long)

    # TEST
    all_input_ids_test = torch.tensor([f.input_ids[0] for f in test_features], dtype=torch.long)
    all_attention_mask_test = torch.tensor([f.attention_mask[0] for f in test_features], dtype=torch.long)

    if test_features[0].token_type_ids is None:
        # For RoBERTa (a potential bug!)
        all_token_type_ids_test = torch.tensor([[0] * args.max_seq_length for f in test_features], dtype=torch.long)
    else:
        all_token_type_ids_test = torch.tensor([f.token_type_ids[0] for f in test_features], dtype=torch.long)

    if args.use_ghmloss or args.use_focalloss:
        all_labels_test = torch.tensor([f.label for f in test_features], dtype=torch.float)
    else:
        all_labels_test = torch.tensor([f.label for f in test_features], dtype=torch.long)

    train_dataset = TensorDataset(all_input_ids_train, all_attention_mask_train, all_token_type_ids_train,
                                  all_labels_train)
    test_dataset = TensorDataset(all_input_ids_test, all_attention_mask_test, all_token_type_ids_test, all_labels_test)

    return train_dataset, test_dataset


def train(args, train_dataset, eval_dataset, model):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # 一个gpu一次跑的batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # 这里是仅仅保持 layernorm的参数 和 所有的bias参数 不变
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    scaler = GradScaler()

    # multi-gpu training (should be after apex fp16 initialization)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    print(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_f1 = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility

    if args.use_pgd:
        pgd = PGD(model)
        K = 3

    if args.use_ema:
        ema = EMA(model, 0.999)
        ema.register()

    if args.use_fgm:
        fgm = FGM(model)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            with autocast():
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            # PGD
            if args.use_pgd:
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(**inputs)[0]
                    if args.n_gpu > 1:
                        loss_adv = loss_adv.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss_adv = loss_adv / args.gradient_accumulation_steps
                    scaler.scale(loss_adv).backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数

            # FGM
            if args.use_fgm:
                fgm.attack()
                loss_adv = model(**inputs)[0]
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / args.gradient_accumulation_steps
                scaler.scale(loss_adv).backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                scaler.step(optimizer)
                scaler.update()

                # EMA
                if args.use_ema:
                    ema.update()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if (
                            args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        if args.use_ema:
                            ema.apply_shadow()
                        results = evaluate(args, model, eval_dataset)
                        if args.use_ema:
                            ema.restore()
                    f1 = results['f1']

                    if f1 > best_f1:
                        best_f1 = f1
                        # Save model checkpoint
                        model_to_save = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())
                        torch.save({
                            'model': model_to_save,
                            'results': results,
                        }, args.output_dir)

                        print("Saving model checkpoint to %s", global_step, args.output_dir)

    return global_step, tr_loss / global_step, best_f1


def evaluate(args, model, eval_dataset):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    if args.use_ghmloss or args.use_focalloss:
        out_label_ids = np.argmax(out_label_ids, axis=1)

    f1 = f1_score(out_label_ids, preds)
    print('eval_loss:', eval_loss)
    print('f1:', f1)
    if f1 == 0.:
        torch.save({'out_label_ids': out_label_ids,
                    'preds': preds}, 'question_f1')
    results['f1'] = f1
    return results


def main(args):
    # gpu
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # 模型位置
    args.model_name_or_path = MODEL_PATHS[args.model_type]

    args.output_dir = 'model_results/' + args.output_dir + '_' + args.model_type + '_' + args.task_name + '_seed_' + str(
        args.seed)

    # 获得tokenizer
    tokenizer_class = TOKENIZER_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # 获得dataset
    train_dataset, test_dataset = load_dataset(args, tokenizer)

    # 获得模型
    model_class = MODEL_CLASSES[args.model_type][args.train_mode]
    model = model_class(args, args.model_name_or_path)
    model.to(args.device)

    # 训练
    _, _, best_f1 = train(args, train_dataset, test_dataset, model)

    results = evaluate(args, model, test_dataset)
    f1 = results['f1']

    if f1 > best_f1:
        # Save model checkpoint
        model_to_save = (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        )  # Take care of distributed/parallel training
        torch.save({
            'model': model_to_save,
            'results': results,
        }, args.output_dir)

        print("Saving model checkpoint to %s", args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type")
    parser.add_argument("--train_mode", default=None, type=str, required=True, help="Train mode")
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory")

    parser.add_argument('--max_length', default=512, type=int, help='max length of sentence')

    parser.add_argument('--test_size', default=0.1, type=float, help='prob of test data')

    parser.add_argument('--per_gpu_train_batch_size', default=8, type=int, help='per_gpu_train_batch_size')
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )

    # 炼丹
    # GHM_Loss
    parser.add_argument(
        "--use_ghmloss", action="store_true", help="ghm loss",
    )
    parser.add_argument(
        "--ghmloss_bins", default=10, type=int, help="bins.",
    )
    parser.add_argument(
        "--ghmloss_alpha", default=0.75, type=float, help="alpha_similar with momentum.",
    )
    # Focal_Loss
    parser.add_argument(
        "--use_focalloss", action="store_true", help="focal loss",
    )
    # PGD adversarial
    parser.add_argument(
        "--use_pgd", action="store_true", help="pgd",
    )
    # EMA
    parser.add_argument(
        "--use_ema", action="store_true", help="ema",
    )
    # FGM
    parser.add_argument(
        "--use_fgm", action="store_true", help="fgm",
    )

    args = parser.parse_args()

    main(args)
