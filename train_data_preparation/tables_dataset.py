import os
import sys
import json
from abc import ABC, abstractmethod
from random import uniform, seed, choices, randint
from math import ceil

from tqdm import tqdm
from collections import defaultdict
import cv2

from PIL import Image
import torch
import numpy as np
from collections import Counter
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bbox import anticlockwise_vertices_from_top_left, compute_ioa_batch, optimized_minimum_bounding_rectangle
from data_preparation.perspective_transform_helper import PerspectiveTransformHelper
from utils.table import charmer_word_normalization, get_width_height_min_angle

MAX_DOC_SIZE = 1024
MAX_NUM_OF_WORDS = 1000
IMAGE_PATCHES_CHANNELS = 1
IOA_THRESHOLD = 0.7
IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg']



# remove overlapping boxes of the same class
def remove_overlaping_boxes_of_same_class(bboxes, labels, cell_pointers):
    indices_to_remove = []
    ioas = compute_ioa_batch(np.asarray(bboxes), np.asarray(bboxes))
    np.fill_diagonal(ioas, 0)
    for current_label in set(labels):
        ioas_tmp = ioas.copy()
        current_label_matrix = np.expand_dims(labels, axis=0) == current_label
        current_label_matrix = current_label_matrix * current_label_matrix.T
        # set ioa's of boxes from different class to 0
        ioas_tmp[~current_label_matrix] = 0

        if np.amax(ioas_tmp) > 0.7:
            ioa_idxs = np.where(ioas_tmp > 0.7)
            for i, j in zip(ioa_idxs[0], ioa_idxs[1]):
                bbox_area_1 = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] -
                                                               bboxes[i][1])
                bbox_area_2 = (bboxes[j][2] - bboxes[j][0]) * (bboxes[j][3] -
                                                               bboxes[j][1])

                if bbox_area_2 >= bbox_area_1:
                    indices_to_remove.append(i)
                else:
                    indices_to_remove.append(j)

    indices_to_remove = set(indices_to_remove)
    bboxes = [
        bboxes[idx] for idx, _ in enumerate(bboxes)
        if idx not in indices_to_remove
    ]
    labels = [
        labels[idx] for idx, _ in enumerate(labels)
        if idx not in indices_to_remove
    ]

    if cell_pointers is not None:
        cell_pointers = [
            cell_pointers[idx] for idx, _ in enumerate(cell_pointers)
            if idx not in indices_to_remove
        ]
    return bboxes, labels, cell_pointers


def read_json_gt(json_file: str, class_map=None):
    with open(json_file, 'r') as openfile:
        data = json.load(openfile)
    width, height = data['image_size']
    bboxes = []
    labels = []
    cell_pointers = []

    for label in data['labels']:
        label_name = label['name']
        bbox = label['bbox']
        cell_pointer = label.get('extra_cells', None)

        if label_name == 'table rotated':
            label_name = 'table'

        if class_map.get(label_name, None) is None:
            continue

        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            label_id = int(class_map[label_name])
            bboxes.append(bbox)
            labels.append(label_id)
            cell_pointers.append(cell_pointer)
        else:
            print('wrong label box coordinates', json_file)
    if len(set(labels)) > 1:
        bboxes, labels, cell_pointers = remove_overlaping_boxes_of_same_class(
            bboxes, labels, cell_pointers)
    return bboxes, labels, cell_pointers, [width, height]


def get_page_ids(roots, dataset_type, ocr_labels_folder):
    json_page_ids = set()
    print('Looking for jsons')
    for root in tqdm(roots):
        lines = os.listdir(os.path.join(root, dataset_type))
        json_page_ids.update(
            set([
                root + ' ' + f.strip().replace(".json", "") for f in lines
                if f.strip().endswith(".json")
            ]))

    print('Looking for images')
    png_page_ids = set()
    for root in tqdm(roots):
        image_directory = os.path.join(root, "images")
        lines = os.listdir(image_directory)

        for f in lines:
            for ext in IMAGE_EXTENSIONS:
                if f.strip().endswith(ext):
                    png_page_ids.add(root + ' ' + f.strip()[:-len(ext)])

    page_ids = sorted(json_page_ids.intersection(png_page_ids))

    # keep page_ids that have a corresponding ocr file
    init_len_page_ids = len(page_ids)
    new_page_ids = defaultdict(list)
    print('Looking for OCR')
    for page_id in tqdm(page_ids):
        root, page_id = page_id.split(' ')
        ocr_json_path = os.path.join(root, ocr_labels_folder,
                                     page_id + '_words.json')
        if not os.path.exists(ocr_json_path):
            ocr_json_path = os.path.join(root, ocr_labels_folder,
                                         page_id + '.ocr.json')
        if os.path.exists(ocr_json_path):
            new_page_ids[root].append(page_id)
    page_ids = new_page_ids

    len_page_ids = 0
    for root in page_ids:
        len_page_ids += len(page_ids[root])

    if init_len_page_ids != len_page_ids:
        print('initial number of gt json files', init_len_page_ids)
        print('pages that have a corresponding json ocr file', len_page_ids)

    return page_ids, len_page_ids


class PDFTablesDataset(torch.utils.data.Dataset, ABC):

    def __init__(self,
                 roots,
                 dataset_type,
                 class_map=None,
                 num_clustering_heads=1,
                 ocr_labels_folder='ocr_gt',
                 is_use_4_points=False,
                 is_use_image_patches=False,
                 is_one_model=False,
                 use_cell_pointers=True,
                 is_augment_in_eval=False):
        self.roots = roots
        self.dataset_type = dataset_type
        self.class_map = class_map
        self.class_list = list(class_map)
        self.class_set = set(class_map.values())
        self.class_set.remove(class_map['no object'])

        self.num_clustering_heads = num_clustering_heads
        self.is_one_model = is_one_model
        self.ocr_labels_folder = ocr_labels_folder
        self.is_use_4_points = is_use_4_points
        self.is_use_image_patches = is_use_image_patches
        self.coco_eval_n_classes = 6
        self.use_cell_pointers = use_cell_pointers
        self.is_augment_in_eval = is_augment_in_eval

        self.page_ids, self.len_page_ids = get_page_ids(
            roots, dataset_type, ocr_labels_folder)

        suffix = 'all_new_fix'
        try:
            with open(f'word_map_{suffix}.json', 'r', encoding="utf8") as f:
                self.word_map = json.load(f)
            # Currently unused; disabled to save memory
            # with open(f'char_map_{suffix}.json', 'r', encoding="utf8") as f:
            #     self.char_map = json.load(f)
        except FileNotFoundError as e:
            raise ValueError(
                f'Missing file(s) word_map_{suffix}.json and/or char_map_{suffix}.json, please run `build_table_vocab.py`'
            )
        self.word_map = pd.Series(self.word_map).astype("string")

        # For the PubTables-1M dataset, get the mapping from the structure dataset to the detection dataset
        # in order to train one model for detection and recognition
        with open('td_to_tr_matching_ocr_gt.json', 'r', encoding="utf8") as f:
            self.td_to_tr_matching = json.load(f)

    def read_word_boxes(self, gt_json, image_size):
        word_boxes = []
        contents_idx = []
        contents = []

        if self.ocr_labels_folder == 'ocr_gt':
            scale = 1.0
            if 'words' in gt_json:
                scale = max(image_size) / max(gt_json['page_rect'])
                gt_json = gt_json['words']

            for word in gt_json:
                box = word['bbox']
                if self.is_use_4_points:
                    box = [[box[0], box[1]], [box[2], box[1]],
                           [box[2], box[3]], [box[0], box[3]]]

                word_boxes.append(np.asarray(box) * scale)
                word_content = charmer_word_normalization(word['text'])
                contents_idx.append(self.word_map.get(word_content, 1))
                contents.append(word_content)
        else:
            for line in gt_json['line_boxes']:
                for word in line['word_boxes']:
                    box = np.asarray(word['bbox'])
                    box = anticlockwise_vertices_from_top_left(box)

                    if box[0][0] < box[2][0] and box[0][1] < box[2][1]:
                        if self.is_use_4_points:
                            # save coordinates in clockwise order
                            box = np.asarray([box[0], box[3], box[2], box[1]])
                            word_boxes.append(box)
                        else:
                            word_boxes.append(np.concatenate([box[0], box[2]]))
                        word_content = charmer_word_normalization(
                            word['content'])
                        contents_idx.append(self.word_map.get(word_content, 1))
                        contents.append(word_content)
        return word_boxes, contents_idx

    @staticmethod
    def get_encapsulating_box(picked_lines):
        min_x = np.amin(picked_lines[:, :, 0])
        max_x = np.amax(picked_lines[:, :, 0])
        min_y = np.amin(picked_lines[:, :, 1])
        max_y = np.amax(picked_lines[:, :, 1])
        label = np.asarray([[min_x, min_y], [max_x, min_y], [max_x, max_y],
                            [min_x, max_y]])
        label = np.expand_dims(label, axis=0)
        return label

    @staticmethod
    def pad_to_max_words(array, max_num_of_words):
        return np.pad(array, [(0, max_num_of_words - len(array))] +
                      [(0, 0) for _ in range(array.ndim - 1)])

    @staticmethod
    def get_adjacency_matrix(clusters_ids, passing_mask=None):
        if passing_mask is not None:
            cluster_counter = Counter(clusters_ids)

        # Make 1 in any position (i, j) where clusters_ids[i] == clusters_ids[j]
        adjacency_matrix = (
            clusters_ids[:, None] == clusters_ids[None]).astype(int)

        # If passing mask is provided, zero out clusters of size 1,
        # unless they are already in passing_mask
        if passing_mask is not None:
            assert passing_mask.shape[0] == passing_mask.shape[
                1] == clusters_ids.size, f'Passed clusters_ids of shape {clusters_ids.shape} and passing_mask of shape {passing_mask.shape}'
            for j, cluster_id in enumerate(clusters_ids):
                if not passing_mask[j, j] and cluster_counter[cluster_id] == 1:
                    adjacency_matrix[j, j] = 0

        return adjacency_matrix

    @abstractmethod
    # currently used only to visualize labels
    def create_target(self,
                      item_idx,
                      bboxes,
                      labels,
                      image_size,
                      img_path,
                      word_boxes,
                      cell_pointers=None):
        ...

    @staticmethod
    def image_crop_resize(img_crop):
        crop_w = ceil(img_crop.width / 32.0)
        crop_h = ceil(img_crop.height / 32.0)
        if crop_h > 0 and crop_w > 0:
            img_crop = img_crop.resize((crop_w * 32, crop_h * 32))
            img_patch = np.asarray(img_crop).reshape(
                (32, crop_h, 32, crop_w)).min(axis=3).min(axis=1)
        else:
            img_patch = 255 * np.ones((32, 32), dtype=np.uint8)
        return img_patch

    @staticmethod
    def get_image_patches(img, word_boxes, is_use_4_points=False):
        img_patches = []
        for wb in word_boxes:
            if is_use_4_points:
                x_min = np.amin(wb[:, 0])
                y_min = np.amin(wb[:, 1])
                x_max = np.amax(wb[:, 0])
                y_max = np.amax(wb[:, 1])
            else:
                x_min = wb[0]
                y_min = wb[1]
                x_max = wb[2]
                y_max = wb[3]

            pad_x = x_max - x_min
            pad_y = y_max - y_min
            img_crop_asymmetric = img.crop(
                (max(0, x_min - pad_x), max(0, y_min - pad_y),
                 min(MAX_DOC_SIZE,
                     x_max + pad_x), min(MAX_DOC_SIZE, y_max + pad_y)))
            img_patch = PDFTablesDataset.image_crop_resize(img_crop_asymmetric)
            img_patches.append(img_patch)

            if IMAGE_PATCHES_CHANNELS > 1:
                pad_x = min(x_min, MAX_DOC_SIZE - x_max, x_max - x_min)
                pad_y = min(y_min, MAX_DOC_SIZE - y_max, y_max - y_min)
                img_crop_symmetric = img.crop((x_min - pad_x, y_min - pad_y,
                                               x_max + pad_x, y_max + pad_y))
                img_patch = PDFTablesDataset.image_crop_resize(img_crop_symmetric)
                img_patches.append(img_patch)

            if IMAGE_PATCHES_CHANNELS > 2:
                max_pad = max(x_max - x_min, y_max - y_min)
                pad = min(max_pad, x_min, y_min, MAX_DOC_SIZE - x_max,
                          MAX_DOC_SIZE - y_max)
                img_crop_same_x_y_pad = img.crop(
                    (x_min - pad, y_min - pad, x_max + pad, y_max + pad))
                img_patch = PDFTablesDataset.image_crop_resize(img_crop_same_x_y_pad)
                img_patches.append(img_patch)

        img_patches = np.asarray(img_patches) / 255.0
        img_patches = img_patches.astype(np.float32)
        return img_patches

    @abstractmethod
    def fix_random_perspective_seed(self, item_idx):
        ...

    def apply_random_perspective_transform(self, item_idx, word_boxes, target,
                                           image_size):
        encaps_box = self.get_encapsulating_box(word_boxes)

        min_border_x = encaps_box[0, 0, 0]
        min_border_y = encaps_box[0, 0, 1]
        max_border_x = encaps_box[0, 2, 0]
        max_border_y = encaps_box[0, 2, 1]
        min_val_x = min_border_x + (max_border_x - min_border_x) * 0.1
        min_val_y = min_border_y + (max_border_y - min_border_y) * 0.1
        max_val_x = max_border_x - (max_border_x - min_border_x) * 0.1
        max_val_y = max_border_y - (max_border_y - min_border_y) * 0.1

        self.fix_random_perspective_seed(item_idx)

        top_left = np.expand_dims([
            uniform(min_border_x, min_val_x),
            uniform(min_border_y, min_val_y)
        ],
                                  axis=0)
        bottom_right = np.expand_dims([
            uniform(max_val_x, max_border_x),
            uniform(max_val_y, max_border_y)
        ],
                                      axis=0)
        top_right = np.expand_dims([
            uniform(max_val_x, max_border_x),
            uniform(min_border_y, min_val_y)
        ],
                                   axis=0)
        bottom_left = np.expand_dims([
            uniform(min_border_x, min_val_x),
            uniform(max_val_y, max_border_y)
        ],
                                     axis=0)
        four_random_points = np.concatenate(
            (top_left, top_right, bottom_right, bottom_left), axis=0)

        perspective_transform = cv2.getPerspectiveTransform(
            encaps_box.astype(np.float32),
            four_random_points.astype(np.float32))
        word_boxes = PerspectiveTransformHelper.transform_line_boxes(
            word_boxes, perspective_transform)

        widths, heights, min_angles = get_width_height_min_angle(word_boxes)
        for w_idx, wb in enumerate(word_boxes):
            rotation = 0
            if widths[w_idx] < heights[w_idx]:
                rotation = 90
            word_boxes[w_idx] = np.asarray(
                optimized_minimum_bounding_rectangle(wb, rotation))

        if target:
            target_boxes = []
            for box in target['boxes']:
                box = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]],
                       [box[0], box[3]]]
                target_boxes.append(box)
            target_boxes = np.asarray(target_boxes)
            target_boxes = target_boxes.astype(float)
            target_boxes[:, :,
                         0] = target_boxes[:, :, 0] * MAX_DOC_SIZE / float(
                             image_size[0])
            target_boxes[:, :,
                         1] = target_boxes[:, :, 1] * MAX_DOC_SIZE / float(
                             image_size[1])
            target_boxes = PerspectiveTransformHelper.transform_line_boxes(
                np.asarray(target_boxes), perspective_transform)
            target_boxes[:, :, 0] = target_boxes[:, :, 0] * float(
                image_size[0]) / float(MAX_DOC_SIZE)
            target_boxes[:, :, 1] = target_boxes[:, :, 1] * float(
                image_size[1]) / float(MAX_DOC_SIZE)

            widths, heights, min_angles = get_width_height_min_angle(
                target_boxes)
            for w_idx, wb in enumerate(target_boxes):
                rotation = 0
                if widths[w_idx] < heights[w_idx]:
                    rotation = 90
                target_boxes[w_idx] = np.asarray(
                    optimized_minimum_bounding_rectangle(wb, rotation))

            target_boxes = np.round(target_boxes)
            target["boxes"] = target_boxes
        return word_boxes, target, perspective_transform


    def apply_zoom_out_perspective_transform(self, word_boxes):
        encaps_box = self.get_encapsulating_box(word_boxes)

        min_border_x = encaps_box[0, 0, 0]
        min_border_y = encaps_box[0, 0, 1]
        max_border_x = encaps_box[0, 2, 0]
        max_border_y = encaps_box[0, 2, 1]
        min_val_x = min_border_x + (max_border_x - min_border_x) * 0.2
        min_val_y = min_border_y + (max_border_y - min_border_y) * 0.2
        max_val_x = max_border_x - (max_border_x - min_border_x) * 0.2
        max_val_y = max_border_y - (max_border_y - min_border_y) * 0.2

        # TODO: maybe x and y should be correlated somehow
        min_x = uniform(min_border_x, min_val_x)
        min_y = uniform(min_border_y, min_val_y)
        max_x = uniform(max_val_x, max_border_x)
        max_y = uniform(max_val_y, max_border_y)

        top_left = np.expand_dims([min_x, min_y], axis=0)
        top_right = np.expand_dims([max_x, min_y], axis=0)
        bottom_right = np.expand_dims([max_x, max_y], axis=0)
        bottom_left = np.expand_dims([min_x, max_y], axis=0)

        four_random_points = np.concatenate(
            (top_left, top_right, bottom_right, bottom_left), axis=0)

        perspective_transform = cv2.getPerspectiveTransform(
            encaps_box.astype(np.float32),
            four_random_points.astype(np.float32))
        word_boxes = PerspectiveTransformHelper.transform_line_boxes(
            word_boxes, perspective_transform)

        widths, heights, min_angles = get_width_height_min_angle(word_boxes)
        for w_idx, wb in enumerate(word_boxes):
            rotation = 0
            if widths[w_idx] < heights[w_idx]:
                rotation = 90
            word_boxes[w_idx] = np.asarray(
                optimized_minimum_bounding_rectangle(wb, rotation))
        return word_boxes

    def map_tr_on_td_dataset_point(self, root, page_id, bboxes, cell_pointers,
                                   labels):
        shadow_mask = 1
        if 'PubTables1M-Detection-PASCAL-VOC' in root and self.is_one_model:
            corresponding_tr_ids = self.td_to_tr_matching.get(page_id, None)
            if corresponding_tr_ids is not None:
                try:
                    root_tr = root.replace('PubTables1M-Detection-PASCAL-VOC',
                                           'PubTables1M-Structure-PASCAL-VOC')
                    for tr_idx, table_id in enumerate(corresponding_tr_ids):
                        bs_tr, ls_tr, structure_cell_pointers, image_size_tr = read_json_gt(
                            os.path.join(root_tr, self.dataset_type,
                                         table_id + '.json'), self.class_map)
                        table_label_index = ls_tr.index(0)
                        box_tr_reference = bs_tr[table_label_index]

                        for b, l, cp in zip(bs_tr, ls_tr,
                                            structure_cell_pointers):
                            if l == 0:
                                continue
                            box_tr_mapped = [
                                b[0] + bboxes[tr_idx][0] - box_tr_reference[0],
                                b[1] + bboxes[tr_idx][1] - box_tr_reference[1],
                                b[2] + bboxes[tr_idx][0] - box_tr_reference[0],
                                b[3] + bboxes[tr_idx][1] - box_tr_reference[1]
                            ]
                            bboxes.append(box_tr_mapped)
                            labels.append(l)
                            new_cp = None
                            if cp is not None and cp != []:
                                new_cp = []
                                for cp_single in cp:
                                    cp_single_mapped = [
                                        cp_single[0] + bboxes[tr_idx][0] -
                                        box_tr_reference[0],
                                        cp_single[1] + bboxes[tr_idx][1] -
                                        box_tr_reference[1],
                                        cp_single[2] + bboxes[tr_idx][0] -
                                        box_tr_reference[0], cp_single[3] +
                                        bboxes[tr_idx][1] - box_tr_reference[1]
                                    ]
                                    new_cp.append(cp_single_mapped)
                            cell_pointers.append(new_cp)
                except Exception as e:
                    print(e)
                    shadow_mask = 0
            else:
                shadow_mask = 0
        return shadow_mask, bboxes, cell_pointers, labels

    def get_root_and_page_id(self, item_idx):
        root = choices(list(self.roots),
                       weights=list(self.roots.values()),
                       k=1)[0]
        page_id = self.page_ids[root][item_idx % len(self.page_ids[root])]
        return root, page_id

    def __getitem__(self, item_idx):
        root, page_id = self.get_root_and_page_id(item_idx)

        for ext in IMAGE_EXTENSIONS:
            img_path = os.path.join(root, "images", page_id + ext)
            if os.path.exists(img_path):
                break

        ocr_json_path = os.path.join(root, self.ocr_labels_folder,
                                     page_id + '_words.json')
        if not os.path.exists(ocr_json_path):
            ocr_json_path = os.path.join(root, self.ocr_labels_folder,
                                         page_id + '.ocr.json')
        if not os.path.exists(ocr_json_path):
            return self.__getitem__(item_idx + 1)

        with open(ocr_json_path) as fp:
            gt_json = json.load(fp)

        img = None
        if self.is_use_image_patches:
            img = Image.open(img_path).convert('L')

        annot_path = os.path.join(root, self.dataset_type, page_id + ".json")

        bboxes, labels, cell_pointers, image_size = read_json_gt(
            annot_path, class_map=self.class_map)

        # if only one model then map corresponding table structure image on the table detection image
        shadow_mask, bboxes, cell_pointers, labels = self.map_tr_on_td_dataset_point(
            root, page_id, bboxes, cell_pointers, labels)

        if ('companies_tables' in root or 'ICDAR2019_cTDaR/detection' in root) and self.is_one_model:
            shadow_mask = 0

        word_boxes, contents_idx = self.read_word_boxes(gt_json, image_size)
        word_boxes = np.asarray(word_boxes).astype(np.float32)
        contents_idx = np.asarray(contents_idx)

        # crop out some word boxes
        # if task == 'detection'
        if self.__class__.__name__ == 'TrainTablesDataset':
            word_boxes_init = word_boxes.copy()
            contents_idx_init = contents_idx.copy()
            if randint(0, 9) < 3:
                try:
                    # crop out some boxes
                    x_min = int(np.amin(word_boxes[:, :, 0]))
                    y_min = int(np.amin(word_boxes[:, :, 1]))
                    x_max = int(np.amax(word_boxes[:, :, 0]))
                    y_max = int(np.amax(word_boxes[:, :, 1]))

                    new_x_min = randint(x_min, int(x_min + 0.2 * (x_max - x_min)))
                    new_x_max = randint(int(x_max - 0.2 * (x_max - x_min)), x_max)
                    new_y_min = randint(y_min, int(y_min + 0.2 * (y_max - y_min)))
                    new_y_max = randint(int(y_max - 0.2 * (y_max - y_min)), y_max)

                    keep_boxes_x = np.logical_and(np.all(word_boxes[:, :, 0] > new_x_min, axis=1), np.all(word_boxes[:, :, 0] < new_x_max, axis=1))
                    keep_boxes_y = np.logical_and(np.all(word_boxes[:, :, 1] > new_y_min, axis=1), np.all(word_boxes[:, :, 1] < new_y_max, axis=1))
                    keep_boxes = np.logical_and(keep_boxes_x, keep_boxes_y)
                    word_boxes = word_boxes[keep_boxes]
                    if contents_idx is not None:
                        contents_idx = contents_idx[keep_boxes]
                except:
                    word_boxes = word_boxes_init
                    contents_idx = contents_idx_init

        crop_size = np.asarray([]).astype(int)

        if self.is_use_image_patches:
            img = img.resize((MAX_DOC_SIZE, MAX_DOC_SIZE))

        if len(word_boxes) > MAX_NUM_OF_WORDS or len(word_boxes) == 0 or len(
                bboxes) == 0:
            return self.__getitem__(item_idx + 1)

        # Reduce class set
        keep_indices = [idx for idx, label in enumerate(labels) if label in self.class_set]
        bboxes = [bboxes[idx] for idx in keep_indices]
        labels = [labels[idx] for idx in keep_indices]
        if cell_pointers is not None:
            cell_pointers = [cell_pointers[idx] for idx in keep_indices]

        target = self.create_target(item_idx, bboxes, labels, image_size,
                                    img_path, word_boxes, cell_pointers)
        word_boxes = np.asarray(word_boxes).astype(np.float32)
        contents_idx = np.asarray(contents_idx)
        num_of_words = len(word_boxes)

        adjacency_matrices = self.create_adjacency_matrices(
            word_boxes, bboxes, labels, cell_pointers)
        adjacency_matrices = np.asarray(adjacency_matrices)

        word_boxes = word_boxes.astype(float)
        if self.is_use_4_points:
            word_boxes[:, :, 0] = word_boxes[:, :, 0] * MAX_DOC_SIZE / float(
                image_size[0])
            word_boxes[:, :, 1] = word_boxes[:, :, 1] * MAX_DOC_SIZE / float(
                image_size[1])
        else:
            word_boxes *= MAX_DOC_SIZE / np.expand_dims([
                float(image_size[0]),
                float(image_size[1]),
                float(image_size[0]),
                float(image_size[1])
            ], 0)
        word_boxes = np.clip(word_boxes, 0, MAX_DOC_SIZE - 1)

        img_patches = []
        if self.is_use_image_patches:
            img_patches = self.get_image_patches(img, word_boxes, self.is_use_4_points)
        else:
            img_patches = np.asarray(img_patches)

        perspective_transform = []
        if self.is_use_4_points and (self.__class__.__name__ == 'TrainTablesDataset'
                                     or self.is_augment_in_eval):
            if randint(0, 9) < 3:
                word_boxes, target, perspective_transform = self.apply_random_perspective_transform(
                    item_idx, word_boxes, target, image_size)
            else:
                perspective_transform = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        if self.__class__.__name__ == 'TrainTablesDataset':
            # with 40% prob zoom out and translation coordinates
            if randint(0, 9) < 4:
                padding = 20
                word_boxes = self.apply_zoom_out_perspective_transform(word_boxes)
                min_x = np.amin(word_boxes[:, :, 0])
                min_y = np.amin(word_boxes[:, :, 1])
                max_x = np.amax(word_boxes[:, :, 0])
                max_y = np.amax(word_boxes[:, :, 1])

                if randint(0, 9) < 3:
                    if randint(0, 9) < 5:
                        random_move_x = -uniform(0, max(0, min_x - padding))
                    else:
                        random_move_x = uniform(0, max(0, MAX_DOC_SIZE - max_x + padding))
                    word_boxes[:, :, 0] += int(random_move_x)

                if randint(0, 9) < 3:
                    if randint(0, 9) < 5:
                        random_move_y = -uniform(0, max(0, min_y - padding))
                    else:
                        random_move_y = uniform(0, max(0, MAX_DOC_SIZE - max_y + padding))
                    word_boxes[:, :, 1] += int(random_move_y)

        word_boxes = np.round(word_boxes)
        word_boxes = np.clip(word_boxes, 0, MAX_DOC_SIZE - 1)

        word_boxes = word_boxes.astype(np.int32)
        contents_idx = contents_idx.astype(np.int32)

        mask = np.zeros(len(word_boxes))
        mask[:num_of_words] = 1

        image_size = np.asarray(image_size)
        if 'cell_pointers' in target:
            del target['cell_pointers']

        header_mask = 1
        if ('fintabnet' in root and 'synthtabnet' not in root) or 'icdar2019' in root:
            header_mask = 0

        return word_boxes, contents_idx, target, adjacency_matrices, num_of_words, mask, image_size, \
               img_patches, perspective_transform, shadow_mask, crop_size, header_mask

    def create_adjacency_matrices(self, word_boxes, bboxes, labels,
                                  cell_pointers):
        adjacency_matrices = [
            np.zeros((len(word_boxes), len(word_boxes)))
            for _ in range(self.num_clustering_heads)
        ]

        for object_box, object_label, cell_ptrs in zip(bboxes, labels,
                                                       cell_pointers):
            # Create clusters and then adjacency matrix for strong connections (without spanning cells)
            ioa = compute_ioa_batch(word_boxes,
                                    np.expand_dims(object_box, axis=0),
                                    max_ioa=True)
            belong_to_object = ioa > IOA_THRESHOLD
            belong_to_object = belong_to_object[:, 0]
            if np.any(belong_to_object):
                for i in np.where(belong_to_object)[0]:
                    for j in np.where(belong_to_object)[0]:
                        adjacency_matrices[object_label][i, j] = 1
                        adjacency_matrices[object_label][j, i] = 1
                # Set weak connections (for spanning cells) in the adjacency matrix.
                # Word boxes in spanning cells can belong to multiple clusters
                if self.use_cell_pointers and cell_ptrs is not None and cell_ptrs != []:
                    ioa_cell = compute_ioa_batch(word_boxes,
                                                 np.asarray(cell_ptrs),
                                                 max_ioa=True)
                    belong_cells = ioa_cell > IOA_THRESHOLD
                    for cell_idx in range(belong_cells.shape[1]):
                        belong_to_cell = belong_cells[:, cell_idx]
                        if np.any(belong_to_cell):
                            # Weak connection (one-sided) - connect words to the spanning cell
                            for i in np.where(belong_to_object)[0]:
                                for j in np.where(belong_to_cell)[0]:
                                    adjacency_matrices[object_label][i, j] = 1
        return adjacency_matrices

    def __len__(self):
        return self.len_page_ids


class TrainTablesDataset(PDFTablesDataset):

    def __init__(self, roots, class_map, num_clustering_heads,
                 ocr_labels_folder, is_use_4_points, is_use_image_patches,
                 is_one_model, use_cell_pointers):
        super().__init__(roots=roots,
                         dataset_type='train',
                         class_map=class_map,
                         num_clustering_heads=num_clustering_heads,
                         ocr_labels_folder=ocr_labels_folder,
                         is_use_4_points=is_use_4_points,
                         is_use_image_patches=is_use_image_patches,
                         is_one_model=is_one_model,
                         use_cell_pointers=use_cell_pointers)

    def fix_random_perspective_seed(self, item_idx):
        pass

    def create_target(self,
                      item_idx,
                      bboxes,
                      labels,
                      image_size,
                      img_path,
                      word_boxes,
                      cell_pointers=None):
        return {}


class ValidTablesDataset(PDFTablesDataset):

    def __init__(self,
                 roots,
                 dataset_type,
                 class_map,
                 num_clustering_heads,
                 ocr_labels_folder,
                 is_use_4_points,
                 is_use_image_patches,
                 is_one_model,
                 use_cell_pointers,
                 is_augment_in_eval=False,
                 is_debug=False):
        super().__init__(roots=roots,
                         dataset_type=dataset_type,
                         class_map=class_map,
                         num_clustering_heads=num_clustering_heads,
                         ocr_labels_folder=ocr_labels_folder,
                         is_use_4_points=is_use_4_points,
                         is_use_image_patches=is_use_image_patches,
                         is_one_model=is_one_model,
                         use_cell_pointers=use_cell_pointers,
                         is_augment_in_eval=is_augment_in_eval)

        for root in self.page_ids:
            self.page_ids[root] = self.page_ids[root][:(1000 // len(self.page_ids))]

        self.len_page_ids = 0
        for root in self.page_ids:
            self.len_page_ids += len(self.page_ids[root])

    def get_root_and_page_id(self, item_idx):
        c = 0
        for root in self.page_ids:
            if item_idx < len(self.page_ids[root]) + c:
                page_id = self.page_ids[root][item_idx - c]
                break
            else:
                c += len(self.page_ids[root])
        return root, page_id

    def fix_random_perspective_seed(self, item_idx):
        seed(item_idx)

    def create_target(self,
                      item_idx,
                      bboxes,
                      labels,
                      image_size,
                      img_path,
                      word_boxes,
                      cell_pointers=None):
        # expand main box to spanning cell y coordinate (for columns) and x coordinate (for rows and headers)
        if self.use_cell_pointers:
            bboxes_extended = []
            for bbox, cell_ptrs, label in zip(bboxes, cell_pointers, labels):
                ioa = compute_ioa_batch(word_boxes,
                                        np.expand_dims(np.asarray(bbox),
                                                       axis=0),
                                        max_ioa=True)
                belong = ioa > IOA_THRESHOLD
                belong = belong[:, 0]
                # if there is any word in the main box then expand it by spanning cells
                if np.any(belong):
                    if cell_ptrs is None or cell_ptrs == []:
                        bboxes_extended.append(bbox)
                    else:
                        cluster_boxes = []
                        for c in cell_ptrs:
                            cluster_boxes.append(c.copy())
                        cluster_boxes.append(bbox)
                        cluster_boxes = np.asarray(cluster_boxes)
                        if label == 1:
                            # if it is column then expand only y coordinate
                            bboxes_extended.append([
                                bbox[0],
                                np.amin(cluster_boxes[:, 1]), bbox[2],
                                np.amax(cluster_boxes[:, 3])
                            ])
                        elif label in [2, 3]:
                            # if it is row or header then expand only x coordinate
                            bboxes_extended.append([
                                np.amin(cluster_boxes[:, 0]), bbox[1],
                                np.amax(cluster_boxes[:, 2]), bbox[3]
                            ])
                else:
                    bboxes_extended.append(bbox)
            bboxes = bboxes_extended

        if len(bboxes) > 0:
            # Convert to Torch Tensor
            if len(labels) > 0:
                bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
            else:
                # Not clear if it's necessary to force the shape of bboxes to be (0, 4)
                bboxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0, ), dtype=torch.int64)
        else:
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0, ), dtype=torch.int64)

        num_objs = bboxes.shape[0]

        # Create target
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = torch.as_tensor([item_idx])
        target["area"] = bboxes[:, 2] * bboxes[:, 3]  # COCO area
        target["iscrowd"] = torch.zeros((num_objs, ), dtype=torch.int64)
        w, h = image_size
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["img_path"] = img_path
        return target
