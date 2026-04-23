import os
import re
import html
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def load_dataset(domain):
    train_path = f'data/{domain}/train.csv.gz'
    valid_path = f'data/{domain}/valid.csv.gz'
    test_path = f'data/{domain}/test.csv.gz'
    
    train_df = pd.read_csv(train_path, compression='gzip')
    valid_df = pd.read_csv(valid_path, compression='gzip')
    test_df = pd.read_csv(test_path, compression='gzip')

    for df in [train_df, valid_df, test_df]:
        df['history'] = df['history'].fillna('')

    return {'train': train_df, 'valid': valid_df, 'test': test_df}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='Musical_Instruments', choices=['Industrial_and_Scientific', 'Musical_Instruments', 'CDs_and_Vinyl'], help='domain of the dataset')
    parser.add_argument('--max_his_len', type=int, default=50, help='maximum length of the history')
    parser.add_argument('--n_workers', type=int, default=16, help='number of worker threads for parallel processing')
    parser.add_argument('--output_dir', type=str, default='processed/', help='directory to save the processed datasets')
    parser.add_argument('--device', type=str, default='mps', help='device to use')
    parser.add_argument('--plm', type=str, default='hyp1231/blair-roberta-base', help='pretrained language model to use')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    return parser.parse_args()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def filter_items_wo_metadata_row(row, item2meta):
    if row['parent_asin'] not in item2meta:
        row['history'] = ''
    history = row['history'].split(' ')
    filtered_history = [_ for _ in history if _ in item2meta]
    row['history'] = ' '.join(filtered_history)
    return row

def truncate_history_row(row, max_his_len):
    history_items = row['history'].split(' ')
    if len(history_items) > max_his_len:
        row['history'] = ' '.join(history_items[-max_his_len:])
    return row


def remap_id(datasets):
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']

    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for user_id, item_id, history in zip(dataset['user_id'], dataset['parent_asin'], dataset['history']):
            if user_id not in user2id:
                user2id[user_id] = len(id2user)
                id2user.append(user_id)
            if item_id not in item2id:
                item2id[item_id] = len(id2item)
                id2item.append(item_id)
            items_in_history = history.split(' ')
            for item in items_in_history:
                if item not in item2id:
                    item2id[item] = len(id2item)
                    id2item.append(item)

    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}
    return data_maps


def list_to_str(l):
    if isinstance(l, list):
        return ', '.join(list_to_str(item) for item in l)
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '

def clean_metadata(row):
    meta_text = ''
    features_needed = ['title', 'features', 'categories', 'description']
    
    for feature in features_needed:
        meta_text += feature_process(row[feature])
    
    row['cleaned_metadata'] = meta_text
    return row

def process_meta(args):
    metadata_path = f'data/{args.domain}/item_metadata.jsonl.gz'
    meta_dataset = pd.read_json(metadata_path, lines=True, compression='gzip')
    
    meta_dataset = meta_dataset.apply(clean_metadata, axis=1)

    item2meta = {}
    for parent_asin, cleaned_metadata in zip(meta_dataset['parent_asin'], meta_dataset['cleaned_metadata']):
        item2meta[parent_asin] = cleaned_metadata

    return item2meta


if __name__ == '__main__':
    args = parse_args()

    datasets = load_dataset(args.domain)
    item2meta = process_meta(args)
    truncated_datasets = {}

    output_dir = os.path.join(args.output_dir, args.domain)
    check_path(output_dir)

    for split in ['train', 'valid', 'test']:
        filtered_dataset = datasets[split].apply(
            lambda row: filter_items_wo_metadata_row(row, item2meta),
            axis=1
        )
        filtered_dataset = filtered_dataset[filtered_dataset['history'] != '']

        truncated_dataset = filtered_dataset.apply(
            lambda row: truncate_history_row(row, args.max_his_len),
            axis=1
        )
        truncated_datasets[split] = truncated_dataset

        output_path = os.path.join(output_dir, f'{args.domain}.{split}.inter')
        with open(output_path, 'w') as f:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for row in truncated_dataset.itertuples():
                f.write(f"{row.user_id}\t{row.history}\t{row.parent_asin}\n")

    data_maps = remap_id(truncated_datasets)
    id2meta = {0: '[PAD]'}
    for item in item2meta:
        if item not in data_maps['item2id']:
            continue
        item_id = data_maps['item2id'][item]
        id2meta[item_id] = item2meta[item]
    data_maps['id2meta'] = id2meta
    output_path = os.path.join(output_dir, f'data.maps')
    with open(output_path, 'w') as f:
        json.dump(data_maps, f)

    device = torch.device(args.device)
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    model = AutoModel.from_pretrained(args.plm).to(device)
    sorted_text = []
    for i in range(1, len(data_maps['item2id'])):
        sorted_text.append(data_maps['id2meta'][i])
    
    all_embeddings = []
    for pr in tqdm(range(0, len(sorted_text), args.batch_size)):
        batch = sorted_text[pr:pr + args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_embeddings.tofile(os.path.join(output_dir, f'{args.plm.split("/")[-1]}.feature'))

    print(f"#Users: {len(data_maps['user2id']) - 1}")
    print(f"#Items: {len(data_maps['item2id']) - 1}")
    n_interactions = {}
    for split in ['train', 'valid', 'test']:
        n_interactions[split] = len(truncated_datasets[split])
        for history in truncated_datasets[split]['history']:
            history_items = history.split(' ')
            n_interactions[split] += len(history_items)
    print(f"#Interaction in total: {sum(n_interactions.values())}")
    print(n_interactions)
    avg_his_length = 0
    for split in ['train', 'valid', 'test']:
        avg_his_length += sum([len(_.split(' ')) for _ in truncated_datasets[split]['history']])
    avg_his_length /= sum([len(truncated_datasets[split]) for split in ['train', 'valid', 'test']])
    print(f"Average history length: {avg_his_length}")
    print(f"Average character length of metadata: {np.mean([len(_) for _ in sorted_text])}")