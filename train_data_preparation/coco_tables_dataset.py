import os
import json
import itertools
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from utils.bbox import compute_ioa_batch, minimum_bounding_rectangle
from train_data_preparation.tables_dataset import ValidTablesDataset, read_json_gt, MAX_NUM_OF_WORDS, IOA_THRESHOLD


class CocoValidDataset(ValidTablesDataset):
    def __init__(self, roots, dataset_type, class_map, num_clustering_heads, ocr_labels_folder, is_use_4_points, is_use_image_patches, is_one_model, use_cell_pointers, is_augment_in_eval):
        super().__init__(roots, dataset_type, class_map, num_clustering_heads, ocr_labels_folder, is_use_4_points, is_use_image_patches, is_one_model, use_cell_pointers, is_augment_in_eval)

        self.coco_eval_n_classes = 6
        self.dataset = {}
        self.dataset['images'] = []
        self.dataset['annotations'] = []
        image_id = 0
        ann_id = 1
        for root in self.page_ids:
            for page_id in tqdm(self.page_ids[root]):
                try:
                    self.dataset['images'].append({'id': image_id})
                    annot_path = os.path.join(root, self.dataset_type, page_id + ".json")

                    # get word boxes from ocr json
                    ocr_json_path = os.path.join(root, self.ocr_labels_folder, page_id + '_words.json')
                    if not os.path.exists(ocr_json_path):
                        ocr_json_path = os.path.join(root, self.ocr_labels_folder, page_id + '.ocr.json')

                    if not os.path.exists(ocr_json_path):
                        print('file does not exist', ocr_json_path)
                        continue

                    with open(ocr_json_path) as fp:
                        gt_json = json.load(fp)

                    # read labels from json files
                    bboxes, labels, cell_pointers, image_size = read_json_gt(annot_path, class_map=self.class_map)
                    word_boxes, _ = self.read_word_boxes(gt_json, image_size)
                    word_boxes = np.asarray(word_boxes).astype(np.float32)

                    if '/data/chargrid_ocr/coe-da-' in root:
                        if labels[0] != 0 or np.sum(np.asarray(labels) == 0) != 1:
                            continue
                        _, word_boxes, bboxes, labels, cell_pointers, _, _, image_size = self.crop_dox_data(word_boxes, bboxes, labels, cell_pointers)

                    if len(word_boxes) > MAX_NUM_OF_WORDS or len(word_boxes) == 0 or len(bboxes) == 0:
                        continue

                    _, bboxes, cell_pointers, labels = self.map_tr_on_td_dataset_point(root, page_id, bboxes, cell_pointers, labels)

                    # Reduce class set
                    keep_indices = [idx for idx, label in enumerate(labels) if label in self.class_set]
                    bboxes = [bboxes[idx] for idx in keep_indices]
                    labels = [labels[idx] for idx in keep_indices]
                    cell_pointers = [cell_pointers[idx] for idx in keep_indices]

                    ioas = compute_ioa_batch(word_boxes, np.asarray(bboxes), max_ioa=True)
                    belongs = ioas > IOA_THRESHOLD

                    for box_idx, (bbox, cell_ptrs, label) in enumerate(zip(bboxes, cell_pointers, labels)):
                        selected_boxes = word_boxes[belongs[:, box_idx]]
                        if len(selected_boxes) > 0:
                            if self.is_use_4_points:
                                new_bbox = np.asarray([
                                    np.amin(selected_boxes[:, :, 0]),
                                    np.amin(selected_boxes[:, :, 1]),
                                    np.amax(selected_boxes[:, :, 0]),
                                    np.amax(selected_boxes[:, :, 1])
                                ])
                            else:
                                new_bbox = np.asarray([
                                    np.amin(selected_boxes[:, 0]),
                                    np.amin(selected_boxes[:, 1]),
                                    np.amax(selected_boxes[:, 2]),
                                    np.amax(selected_boxes[:, 3])
                                ])

                            # expand main box to spanning cell y coordinate (for columns) and x coordinate (for rows and headers)
                            if self.use_cell_pointers and cell_ptrs is not None and cell_ptrs != []:
                                # get words inside the spanning cell and expand till then (not necessarily coords of the spanning cell)
                                ioa_cell = compute_ioa_batch(word_boxes, np.asarray(cell_ptrs), max_ioa=True)
                                belong_cells = ioa_cell > IOA_THRESHOLD
                                cell_boxes = np.asarray([new_bbox])
                                if np.any(belong_cells):
                                    word_boxes_in_cells = word_boxes[np.any(belong_cells, axis=1)]
                                    if self.is_use_4_points:
                                        word_boxes_in_cells = np.asarray([np.amin(word_boxes_in_cells[:, :, 0]),
                                                                          np.amin(word_boxes_in_cells[:, :, 1]),
                                                                          np.amax(word_boxes_in_cells[:, :, 0]),
                                                                          np.amax(word_boxes_in_cells[:, :, 1])
                                                                         ])
                                        word_boxes_in_cells = np.expand_dims(word_boxes_in_cells, axis=0)
                                    cell_boxes = np.concatenate([cell_boxes, word_boxes_in_cells])

                                    if label == 1:
                                        # if it is column then expand only y coordinate
                                        new_bbox = [new_bbox[0],
                                                    np.amin(cell_boxes[:, 1]),
                                                    new_bbox[2],
                                                    np.amax(cell_boxes[:, 3])
                                                   ]
                                    elif label in [2, 3]:
                                        # if it is row or header then expand only x coordinate
                                        new_bbox = [np.amin(cell_boxes[:, 0]),
                                                    new_bbox[1],
                                                    np.amax(cell_boxes[:, 2]),
                                                    new_bbox[3]
                                                   ]
                        else:
                            # Skip label as it does not contain any word boxes
                            continue

                        ann = {
                            'area': (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]),
                            'iscrowd': 0,
                            'bbox': [new_bbox[0], new_bbox[1], new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]],
                            'category_id': label,
                            'image_id': image_id,
                            'id': ann_id,
                            'ignore': 0,
                            'segmentation': []
                        }
                        self.dataset['annotations'].append(ann)
                        ann_id += 1
                finally:
                    image_id += 1
        self.dataset['categories'] = [{'id': idx} for idx in self.class_list[:-1]]
        self.createIndex()


    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    #####################################
    # The following is actually unused? #
    #####################################

    def getImgIds(self):
        return range(self.len_page_ids)

    def getCatIds(self):
        return range(self.coco_eval_n_classes)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]

            ids = [ann['id'] for ann in anns]
        return ids


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')