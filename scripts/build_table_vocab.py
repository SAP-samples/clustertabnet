import json
import os
import sys
import time
import multiprocessing
from functools import partial
from collections import Counter

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.table import charmer_word_normalization
from train.table_extraction import get_data_paths_and_maps, parse_args
from train_data_preparation.tables_dataset import get_page_ids


def process_files(page_id, ocr_labels_folder):
    word_freq = Counter()
    char_freq = Counter()

    root, page_id = page_id.split(' ')

    ocr_json_path = os.path.join(root, ocr_labels_folder, page_id + '_words.json')
    if not os.path.exists(ocr_json_path):
        ocr_json_path = os.path.join(root, ocr_labels_folder, page_id + '.ocr.json')

    with open(ocr_json_path) as fp:
        gt_json = json.load(fp)

        if ocr_labels_folder == 'ocr_gt':
            if 'words' in gt_json:
                gt_json = gt_json['words']

            for word in gt_json:
                word_normalized = charmer_word_normalization(word['text'])
                word_freq[word_normalized] += 1
                char_freq.update(word['text'])
        else:
            for line in gt_json['line_boxes']:
                for word in line['word_boxes']:
                    word_normalized = charmer_word_normalization(word['content'])
                    word_freq[word_normalized] += 1
                    char_freq.update(word['content'])
    return word_freq, char_freq


def build_vocab(page_ids, word_map_max_size, char_map_max_size, min_word_freq,
                min_char_freq, num_clustering_heads, ocr_labels_folder):
    """
    Build word_map and char_map from the training data
    :param word_map_max_size: maximum number of words in the mapping
    :param char_map_max_size: maximum bumber of characters in the mapping
    :param min_word_freq: minimum frequency of a word to be included in the mapping
    :param min_char_freq: minimum frequency of a character to be included in the mapping
    :return: the mapping of words and characters to their indices
    """
    print('Building word and character frequency dictionary')
    t0 = time.time()
    word_freq = Counter()
    char_freq = Counter()

    page_ids_list = []
    for root in page_ids:
        print('root', root)
        for f in page_ids[root]:
            page_ids_list.append(root + ' ' + f)

    n_jobs = 16
    with multiprocessing.Pool(n_jobs) as p:
        result = list(tqdm(p.imap(partial(process_files, ocr_labels_folder=ocr_labels_folder),
                                  page_ids_list),
                           total=len(page_ids_list)))

    for r in result:
        word_freq.update(r[0])
        char_freq.update(r[1])

    word_map = {'<pad>': 0, '<unk>': 1}
    char_map = {'<pad>': 0, '<unk>': 1} # , '<bow>': 2, '<eow>': 3, '<skip>': 4}
    top_word_freq = word_freq.most_common(word_map_max_size - 2)
    top_char_freq = char_freq.most_common(char_map_max_size - 2) # 5

    i = len(word_map)
    for w, f in top_word_freq:
        if f >= min_word_freq:
            word_map[w] = i
            i += 1

    i = len(char_map)
    for c, f in top_char_freq:
        if f >= min_char_freq:
            char_map[c] = i
            i += 1

    print(f'all words: {len(word_freq)}, selected words: {len(word_map)}')
    print(f'all chars: {len(char_freq)}, selected chars: {len(char_map)}')
    print(f'time used: {time.time() - t0:.1f}')

    file_suffix = ''
    if num_clustering_heads == 1:
        file_suffix = '_detection'
    elif num_clustering_heads == 4:
        file_suffix = '_all_new_fix'

    with open('word_map' + file_suffix + '.json', 'w') as word_map_fp:
        word_map_fp.write(json.dumps(word_map))

    with open('char_map' + file_suffix + '.json', 'w') as char_map_fp:
        char_map_fp.write(json.dumps(char_map))

    return word_map, char_map


if __name__ == '__main__':
    args = parse_args()
    dataset_paths, dataset_paths_validation, num_clustering_heads, class_map = get_data_paths_and_maps(args.task, args.use_dox_datasets)

    print(dataset_paths)
    print(args.ocr_labels_folder)
    page_ids, _ = get_page_ids(dataset_paths, 'train', args.ocr_labels_folder)
    build_vocab(page_ids, 30016, 200, 1, 1, num_clustering_heads, args.ocr_labels_folder)
