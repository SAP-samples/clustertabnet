import glob
import json
import os
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
from tqdm import tqdm

from utils.bbox import ensure_4_vertices, compute_ioa_batch
from train_data_preparation.tables_dataset import PDFTablesDataset
from train.table_extraction import get_data_paths_and_maps, parse_args


def build_td_tr_correspondence_dict(dataset, is_debug_plot=False):
    print('ERROR, it should not be called as correspondance dict was already built')
    td_to_tr_matching = defaultdict(list)
    tr_to_td_matching = {}

    root_td = '/data/chargrid_ocr/PubTables-1M/pubtables-1m/pubtables1m/PubTables1M-Detection-PASCAL-VOC'
    root_tr = '/data/chargrid_ocr/PubTables-1M/pubtables-1m/pubtables1m/PubTables1M-Structure-PASCAL-VOC'

    class_map_td = {'table': 0, 'table rotated': 0, 'no object': 2}
    class_map_tr = {'table': 0,
                    'table column': 1,
                    'table row': 2,
                    'table column header': 3,
                    'table projected row header': 4,
                    'table spanning cell': 5,
                    'no object': 6
                    }

    table_detection_paths = glob.glob(os.path.join(root_td, 'images', '*.jpg'))
    table_detection_images = [os.path.basename(x) for x in table_detection_paths]
    table_detection_images_prefixes = defaultdict(list)
    for x in table_detection_images:
        table_detection_images_prefixes[x.split('_')[0]].append(x)

    table_structure_paths = glob.glob(os.path.join(root_tr, 'images', '*.jpg'))
    table_structure_images = [os.path.basename(x) for x in table_structure_paths]
    table_structure_images_prefixes = defaultdict(list)
    for x in table_structure_images:
        table_structure_images_prefixes[x.split('_')[0]].append(x)

    dataset_type_td = ['train', 'val', 'test']

    for d_type in dataset_type_td:
        for f in tqdm(os.listdir(os.path.join(root_td, d_type))):
            f_basename_td = f.split('.')[0]

            if is_debug_plot:
                f_image_td = os.path.join(root_td, 'images', f_basename_td + '.jpg')
                im_td = Image.open(f_image_td)
                _, ax_td = plt.subplots(figsize=(30, 15))
                ax_td.imshow(im_td)

            # read table boxes
            table_xml_file = os.path.join(root_td, d_type, f_basename_td + '.xml')
            if not os.path.exists(table_xml_file):
                continue
            bs_td, _, image_size_td = read_pascal_voc(table_xml_file, class_map_td)
            rects_td = ensure_4_vertices(np.asarray(bs_td))

            if is_debug_plot:
                rects_td_plot = []
                for b in bs_td:
                    box = [b[0], b[1],
                           b[2], b[1],
                           b[2], b[3],
                           b[0], b[3]]
                    rects_td_plot.append(Polygon(np.asarray(box).reshape(-1, 2)))
                ax_td.add_collection(PatchCollection(rects_td_plot, facecolor='none', edgecolor='r'))

            # read word boxes
            if os.path.exists(os.path.join(root_td, 'ocr_gt', f_basename_td + '_words.json')):
                with open(os.path.join(root_td, 'ocr_gt', f_basename_td + '_words.json')) as fp:
                    gt_json_td = json.load(fp)
                    td_word_boxes, _ = dataset.read_word_boxes(gt_json_td, image_size_td)
                    td_word_boxes = ensure_4_vertices(np.asarray(td_word_boxes))

                    # if is_save:
                    #     rects = []
                    #     for w in td_word_boxes:
                    #         rects.append(Polygon(w))
                    #     ax_td.add_collection(PatchCollection(rects, facecolor='none', edgecolor='b'))
            else:
                # print('missing ocr word boxes file for table detection image')
                continue

            # get number of word boxes in each table
            ioa = compute_ioa_batch(np.asarray(rects_td), np.asarray(td_word_boxes), max_ioa=True)
            num_of_boxes_in_tables_td = np.sum(ioa > 0.7, axis=1)

            # find all table detection images with a given prefix
            image_with_prefix_td = table_detection_images_prefixes[f_basename_td.split('_')[0]]
            idx_to_sort_by = [int(f.split('_')[-1].replace('.jpg', '')) for f in image_with_prefix_td]
            # sort based on suffixes like `_1.jpg`, `_5.jpg`, `_10.jpg`, `_11.jpg` etc.
            image_with_prefix_td = [x for _, x in sorted(zip(idx_to_sort_by, image_with_prefix_td), key=lambda pair: pair[0])]

            num_of_tables = []
            num_of_tables_cumulative = [0]
            for image_f in image_with_prefix_td:
                _, ls, _ = read_pascal_voc(os.path.join(root_td, d_type, image_f.replace('.jpg', '.xml')), class_map_td)
                num_of_tables.append(len(ls))
            for n in num_of_tables:
                num_of_tables_cumulative.append(num_of_tables_cumulative[-1] + n)
            index_of_td_image = image_with_prefix_td.index(f_basename_td + '.jpg')

            # get the corresponding images from table structure dataset
            table_structure_image_paths = table_structure_images_prefixes[f_basename_td.split('_')[0]]
            idx_to_sort_by = [int(f.split('_')[-1].replace('.jpg', '')) for f in table_structure_image_paths]
            # sort based on suffixes like `_1.jpg`, `_5.jpg`, `_10.jpg`, `_11.jpg` etc.
            table_structure_image_paths = [x for _, x in sorted(zip(idx_to_sort_by, table_structure_image_paths), key=lambda pair: pair[0])]
            table_structure_image_paths_selected = table_structure_image_paths[num_of_tables_cumulative[index_of_td_image] : num_of_tables_cumulative[index_of_td_image+1]]

            for tr_idx, table_structure_image_path in enumerate(table_structure_image_paths_selected):
                f_basename_tr = os.path.basename(table_structure_image_path).replace('.jpg', '')
                dataset_type_tr = None
                if os.path.exists(os.path.join(root_tr, 'train', f_basename_tr + '.xml')):
                    dataset_type_tr = 'train'
                elif os.path.exists(os.path.join(root_tr, 'val', f_basename_tr + '.xml')):
                    dataset_type_tr = 'val'
                elif os.path.exists(os.path.join(root_tr, 'test', f_basename_tr + '.xml')):
                    dataset_type_tr = 'test'
                else:
                    print('missing xml file for the image', f_basename_tr)
                    continue

                bs, ls, image_size_tr = read_pascal_voc(os.path.join(root_tr, dataset_type_tr, f_basename_tr + '.xml'), class_map_tr)
                box_tr_reference = None
                table_label_index = ls.index(0)
                box_tr_reference = ensure_4_vertices(np.asarray([bs[table_label_index]]))

                if os.path.exists(os.path.join(root_tr, 'ocr_gt', f_basename_tr + '_words.json')):
                    with open(os.path.join(root_tr, 'ocr_gt', f_basename_tr + '_words.json')) as fp:
                        gt_json_tr = json.load(fp)
                    tr_word_boxes, _ = dataset.read_word_boxes(gt_json_tr, image_size_tr)

                ioa = compute_ioa_batch(np.asarray(box_tr_reference), np.asarray(tr_word_boxes), max_ioa=True)
                num_of_boxes_in_tables_tr = np.sum(ioa > 0.7, axis=1)[0]

                # if different in number of word boxes inside the table between TR and TD image is bigger than 10% then skip the image
                if abs(num_of_boxes_in_tables_tr - num_of_boxes_in_tables_td[tr_idx]) / num_of_boxes_in_tables_tr > 0.1:
                    print('ERROR, different number of word boxes in td and tr files!!!!!!!', f_basename_td)
                    print(f_basename_tr)
                    continue

                td_to_tr_matching[f_basename_td].append(f_basename_tr)
                tr_to_td_matching[f_basename_tr] = f_basename_td

    with open('td_to_tr_matching_ocr_gt.json', 'w') as matching_fp:
        json.dump(td_to_tr_matching, matching_fp)
    with open('tr_to_td_matching_ocr_gt.json', 'w') as matching_fp:
        json.dump(tr_to_td_matching, matching_fp)


if __name__ == '__main__':
    args = parse_args()
    dataset_paths, dataset_paths_validation, num_clustering_heads, class_map = get_data_paths_and_maps(args.task, args.use_dox_datasets)
    dataset = PDFTablesDataset(dataset_paths,
                            'train',
                            include_eval=True,
                            make_coco=False,
                            class_map=class_map,
                            num_clustering_heads=num_clustering_heads,
                            ocr_labels_folder=args.ocr_labels_folder,
                            is_use_4_points=args.is_use_4_points,
                            is_use_image_patches=args.is_use_image_patches,
                            coco_eval_n_classes=args.coco_eval_n_classes,
                            is_one_model=args.task == 'both',
                            use_cell_pointers=not args.no_cell_pointers)
