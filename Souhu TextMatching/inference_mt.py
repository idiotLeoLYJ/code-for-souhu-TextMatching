# coding:utf-8
# Author:Yuanjie Liu (CamilleLeo)

import os
import torch
import argparse
from tqdm import tqdm
from utils import utils

import numpy as np
from utils.utils import load_json, save_json

from models.bert_mt import BertTextClassification
from models.bert_average_mt import BertTextClassificationWithAveragePooling
from models.bert_cls_avg import BertTextClassificationWithClsAveragePooling
from models.bert_ncls import BertTextClassificationNCls
from models.electra import ElectraTextClassification
from transformers import BertTokenizer, ElectraTokenizer

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

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
    'bert': './cached_test_data/cached_bert_tokenizer_data',
    'electra': './cached_test_data/cached_electra_tokenizer_data',
    'roberta-wwm-ext-large': './cached_test_data/cached_bert_tokenizer_data',
    'mac-bert': './cached_test_data/cached_bert_tokenizer_data',
}
DATA_DIR = 'test_data/all_test_multi_task_data.json'

def load_model(model, model_path):
    model_info = torch.load(model_path, map_location="cpu")
    f1 = model_info['results']['f1']
    if hasattr(model, "module"):
        model.module.load_state_dict(model_info['model'], strict=False)
    else:
        model.load_state_dict(model_info['model'], strict=False)
    return model, f1

def tokenizer_single(example, args, tokenizer):
    example_encoding = tokenizer(
        [(example[1], example[2])],
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
    )
    if args.use_ghmloss or args.use_focalloss:
        example_encoding['label'] = [0., 1.]
    else:
        example_encoding['label'] = example[3]
        
    if example[0] in [0, 2, 4]:
        example_encoding['type'] = 0
    else:
        example_encoding['type'] = 1
    return example_encoding

def inference(args):
    args.model_name_or_path = MODEL_PATHS[args.model_type]
    
    model_class = MODEL_CLASSES[args.model_type][args.train_mode]
    model = model_class(args, args.model_name_or_path)

    print(args.trained_model_path)

    # 获得tokenizer
    tokenizer_class = TOKENIZER_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    
    cached_features_file_test = \
        "{}_{}_{}".format(
            CACHE_NAME[args.model_type],
            'mt',
            'test',
        )
    test_data = utils.load_json(DATA_DIR)

    if os.path.exists(cached_features_file_test):
        features = torch.load(cached_features_file_test)
    else:
        features = []
        for example in tqdm(test_data):
            features.append(tokenizer_single(example, args, tokenizer))
        print("Saving features into cached file")
        torch.save(features, cached_features_file_test)
        
    all_input_ids_test = torch.tensor([f.input_ids[0] for f in features], dtype=torch.long)
    all_attention_mask_test = torch.tensor([f.attention_mask[0] for f in features], dtype=torch.long)

    if features[0].token_type_ids is None:
        # For RoBERTa (a potential bug!)
        all_token_type_ids_test = torch.tensor([[0] * args.max_seq_length for f in features], dtype=torch.long)
    else:
        all_token_type_ids_test = torch.tensor([f.token_type_ids[0] for f in features], dtype=torch.long)

    if args.use_ghmloss or args.use_focalloss:
        all_labels_test = torch.tensor([f.label for f in features], dtype=torch.float)
    else:
        all_labels_test = torch.tensor([f.label for f in features], dtype=torch.long)
    all_types_test = torch.tensor([f.type for f in features], dtype=torch.long)
    
    test_dataset = TensorDataset(all_input_ids_test, all_attention_mask_test, all_token_type_ids_test, 
                                 all_labels_test, all_types_test)
    
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gpu_id = 'cuda:' + str(args.gpu_id)
    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    print(device)

    model, f1 = load_model(model, args.trained_model_path)
    model.to(device)

    results = {}
    preds = []
    
    softmax = torch.nn.Softmax(dim=1)
    
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.eval()
            types = batch[4]
            type_index_a = [key for key, value in enumerate(types) if value == 0]
            type_index_b = [key for key, value in enumerate(types) if value == 1]
            batch.append(torch.tensor(type_index_a, dtype=torch.long))
            batch.append(torch.tensor(type_index_b, dtype=torch.long))
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],
                      "type_index_a": batch[5], "type_index_b": batch[6]}

            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )

            outputs = model(**inputs)
            logits = outputs[1]

            preds.extend(np.array(softmax(logits).detach().cpu()))
    
    for item, pred in zip(test_data, preds):
        test_id = item[3]
        results[test_id] = [str(pred[0]), str(pred[1])]
    
    args.output_path = args.output_dir + args.trained_model_path.split('/')[1] + 'infer_results_f1_' + str(f1)[:6] + '.json'

    save_json(results, args.output_path)


def main(args):

    args.output_path = args.output_dir + args.trained_model_path.split('/')[1] + 'infer_results.json'
    inference(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type")
    parser.add_argument("--train_mode", default=None, type=str, required=True, help="Model type")
    parser.add_argument("--trained_model_path", default=None, type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_dir", default='inference_results/', type=str, help="The output directory")

    parser.add_argument('--max_length', default=512, type=int, help='max length of sentence')

    parser.add_argument('--gpu_id', default=7, type=int, help='gpu id')

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

    args = parser.parse_args()

    main(args)
