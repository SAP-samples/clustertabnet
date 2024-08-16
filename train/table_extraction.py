import os
import sys
import argparse
from pprint import pprint

import torch
from PIL import Image, ImageDraw
import numpy as np
import cv2
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_data_preparation.tables_dataset import TrainTablesDataset, ValidTablesDataset, MAX_DOC_SIZE, IMAGE_PATCHES_CHANNELS
from train_data_preparation.coco_tables_dataset import CocoValidDataset
from utils.bbox import minimum_bounding_rectangle

from model_functions.transformer_tf_copy import TransformerEncoderTable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def validate_coco(valid_dataset,
                  pred_boxes,
                  pred_classes,
                  pred_probs,
                  restrict_to_classes=None):
    from train.coco_eval import CocoEvaluator
    if restrict_to_classes is not None:
        num_classes_backup = valid_dataset.coco_eval_n_classes
        print(f'For {restrict_to_classes} classes')
        valid_dataset.coco_eval_n_classes = restrict_to_classes
    else:
        print(f'For all ({valid_dataset.coco_eval_n_classes}) classes')
    coco_evaluator = CocoEvaluator(valid_dataset, ('bbox', ))

    for image_id in range(len(valid_dataset)):
        if image_id in pred_boxes:
            if len(pred_boxes[image_id]) > 0:
                result = {
                    'scores': torch.as_tensor(pred_probs[image_id]),
                    'labels': torch.as_tensor(pred_classes[image_id]),
                    'boxes': torch.as_tensor(pred_boxes[image_id])
                }
                res = {image_id: result}
                coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
    print("COCO metrics summary: AP: {:.3f}, AP50: {:.3f}, AR: {:.3f}".format(
        stats[0], stats[1], stats[8]))

    if restrict_to_classes is not None:
        valid_dataset.coco_eval_n_classes = num_classes_backup
    return stats[0], stats[1], stats[8]


def alternative_clustering(preds):
    assert preds.ndim == 2 and preds.shape[0] == preds.shape[1]
    n = preds.shape[0]
    sort_idx = np.argsort(-preds.flatten())
    clusters = np.arange(n)
    already_done = set()
    for y, x in zip(sort_idx // n, sort_idx % n):
        if (y, x) in already_done or (x, y) in already_done:
            continue
        else:
            already_done.add((y, x))

        if preds[y, x] < 1:
            # Nothing more to cluster, stop
            break
        if clusters[y] == clusters[x]:
            # Already same clusters, nothing to do
            continue

        # Otherwise, check if we can merge
        mask_x = clusters == clusters[x]
        mask_y = clusters == clusters[y]

        # This block is just for debugging
        s1 = set(np.where(mask_x)[0])
        s2 = set(np.where(mask_y)[0])
        assert not s1.intersection(s2)
        assert set(clusters[mask_x]) == {clusters[x]}
        assert set(clusters[mask_y]) == {clusters[y]}

        sub_preds = preds[np.ix_(mask_x, mask_y)]

        # Check if the average pair votes for more than against:
        if sub_preds.mean() >= 1:
            clusters[mask_x] = clusters[y]
    return clusters


def evaluate_one_head(head_preds_clustering, head_preds_clustering_weak,
                      head_adjacency_matrix, is_use_connected_components=True,
                      is_inference=False):
    # weak connections cannot be greater than 1 even after we added transpose
    STRONG_THRESHOLD = 1.3
    passing_mask = head_preds_clustering > STRONG_THRESHOLD

    # ## DEBUG: uncomment to use label as a prediction to debug
    # head_preds_clustering = head_adjacency_matrix + head_adjacency_matrix.T
    # passing_mask = head_preds_clustering > 1

    # nothing to evaluate, adjacency matrix is empty and prediction is empty
    if not np.any(passing_mask) and (is_inference is True or not np.any(head_adjacency_matrix)):
        return passing_mask, head_preds_clustering, 0, 0, 1

    if is_use_connected_components:
        # labels = alternative_clustering(head_preds_clustering)
        graph = csr_matrix(passing_mask)
        _, labels = connected_components(csgraph=graph,
                                         directed=False,
                                         return_labels=True)
        passing_mask = TrainTablesDataset.get_adjacency_matrix(
            labels + 1, passing_mask).astype(bool)

    # keep only weak connections, remove strong connections
    head_preds_clustering_weak = head_preds_clustering_weak.copy()
    head_preds_clustering_weak[passing_mask] = 0
    WEAK_THRESHOLD = 0.5
    passing_mask_weak = head_preds_clustering_weak >= WEAK_THRESHOLD

    # ## DEBUG: uncomment to use label as a prediction to debug
    # head_preds_clustering_weak = head_adjacency_matrix.astype(int)
    # head_preds_clustering_weak[passing_mask] = 0
    # passing_mask_weak = head_preds_clustering_weak == 1

    # for each strong component get all the weak connections and keep them depending on the threshold
    rows_to_select = set(range(len(passing_mask)))
    while rows_to_select:
        row_idx = rows_to_select.pop()
        current_row_passing_mask = passing_mask[row_idx]

        if np.any(current_row_passing_mask):
            # remove other rows as we already have connected components (all rows in connected components should be the same)
            if is_use_connected_components:
                rows_to_select = rows_to_select.difference(
                    np.where(current_row_passing_mask)[0])
            else:
                previous_passing_mask = passing_mask[:row_idx]
                if (previous_passing_mask == current_row_passing_mask[None]
                    ).all(axis=1).any():
                    continue

            current_weak_passing_mask = passing_mask_weak[
                current_row_passing_mask]
            if np.any(current_weak_passing_mask):
                # remove weak connections if average is below 0.5
                passing_mask_weak[current_row_passing_mask] = np.mean(
                    current_weak_passing_mask, axis=0) >= 0.5

    predictions = passing_mask | passing_mask_weak
    if is_inference is False:
        correct_preds = np.sum(predictions & head_adjacency_matrix)
        number_of_ones = np.sum(predictions | head_adjacency_matrix)

        if number_of_ones == 0:
            accuracy = 1
        else:
            accuracy = correct_preds / number_of_ones
    else:
        correct_preds = 0
        number_of_ones = 0
        accuracy = 0

    return passing_mask, passing_mask_weak, correct_preds, number_of_ones, accuracy


def rescale_word_boxes_to_image_size(word_boxes, image_size):
    w, h = image_size.reshape(-1)
    old_shape = word_boxes.shape
    word_boxes = word_boxes.reshape(len(word_boxes), -1, 2) * np.array(
        [[[w / float(MAX_DOC_SIZE), h / float(MAX_DOC_SIZE)]]])
    return np.round(word_boxes.reshape(old_shape))


def construct_predicted_rect(selected_word_boxes, selected_word_boxes_weak,
                             mask_idx):
    if selected_word_boxes.ndim == 3:
        selected_word_boxes = np.concatenate(
            [selected_word_boxes.min(axis=1),
             selected_word_boxes.max(axis=1)],
            axis=1)
        selected_word_boxes_weak = np.concatenate([
            selected_word_boxes_weak.min(axis=1),
            selected_word_boxes_weak.max(axis=1)
        ],
                                                  axis=1)

    if selected_word_boxes_weak.shape[0] == 0 or mask_idx not in [1, 2, 3]:
        return np.asarray([
            np.amin(selected_word_boxes[:, 0]),
            np.amin(selected_word_boxes[:, 1]),
            np.amax(selected_word_boxes[:, 2]),
            np.amax(selected_word_boxes[:, 3])
        ])

    if mask_idx == 1:
        # Columns; add spanning cells on the y direction
        xmin = np.amin(selected_word_boxes[:, 0])
        ymin = min(
            np.amin(selected_word_boxes[:, 1]),
            np.amin(selected_word_boxes_weak[:, 1]),
        )
        xmax = np.amax(selected_word_boxes[:, 2])
        ymax = max(
            np.amax(selected_word_boxes[:, 3]),
            np.amax(selected_word_boxes_weak[:, 3]),
        )
        return np.array([xmin, ymin, xmax, ymax])

    xmin = min(
        np.amin(selected_word_boxes[:, 0]),
        np.amin(selected_word_boxes_weak[:, 0]),
    )
    ymin = np.amin(selected_word_boxes[:, 1])
    xmax = max(
        np.amax(selected_word_boxes[:, 2]),
        np.amax(selected_word_boxes_weak[:, 2]),
    )
    ymax = np.amax(selected_word_boxes[:, 3])
    return np.array([xmin, ymin, xmax, ymax])


def create_imagedraw(file_path, word_boxes, target, accuracies_per_head,
                     perspective_transform, is_augment_in_eval=False,
                     crop_size=[]):
    val_image = Image.open(file_path)
    if len(crop_size) == 4:
        val_image = val_image.crop(tuple(crop_size))
    width, height = val_image.size

    if word_boxes.ndim > 2 and is_augment_in_eval:
        val_image = val_image.resize((MAX_DOC_SIZE, MAX_DOC_SIZE))
        perspective_transform = perspective_transform.cpu().numpy()[0]
        image_skewed = cv2.warpPerspective(
            np.asarray(val_image), perspective_transform,
            (int(MAX_DOC_SIZE), int(MAX_DOC_SIZE)))
        val_image = Image.fromarray(np.uint8(image_skewed))
        val_image = val_image.resize((width, height))

    draw = ImageDraw.Draw(val_image)
    # visualize word boxes
    for box in word_boxes:
        if box.ndim == 1:
            draw.rectangle(box.tolist(), outline='lightgrey')
        else:
            draw.polygon([tuple(x) for x in box], outline='lightgrey')

    # visualize GT boxes
    #           [ table            column        row          header       row header   spanning cell]
    colors_gt = [
        'palevioletred', 'lightgreen', 'lightblue', 'palegoldenrod',
        'lightpink', 'rosybrown'
    ]
    for box, label in zip(target['boxes'].numpy()[0],
                          target['labels'].numpy()[0]):
        if accuracies_per_head[label] != 1:
            cc = label + 1
            if not isinstance(box[0], np.ndarray):
                box = [box[0] - cc, box[1] - cc, box[2] - cc, box[3] - cc]
                draw.rectangle(box, outline=colors_gt[label], width=2)
            else:
                draw.polygon([tuple(x) for x in box], outline=colors_gt[label])

    return val_image, draw


def draw_rectangles(word_boxes, pred_boxes, pred_classes, target,
                    accuracies_per_head, perspective_transform, output_dir, is_augment_in_eval=False, crop_size=[]):

    file_path = target['img_path'][0]
    file_name = (os.path.basename(file_path)[:-4])

    # Get bounding boxes around clusters
    #             [ table  column   row     header    row header  spanning cell]
    colors_pred = ['red', 'green', 'blue', 'orange', 'pink', 'brown']

    table_class_map = {0: 'table',
                       1: 'column',
                       2: 'row',
                       3: 'header'
                      }

    val_image, draw = create_imagedraw(file_path, word_boxes, target,
                                       accuracies_per_head,
                                       perspective_transform,
                                       is_augment_in_eval=is_augment_in_eval,
                                       crop_size=crop_size)

    tables = {'tables': []}
    is_save_acc = True
    for mask_idx, predicted_table_rect in zip(pred_classes, pred_boxes):
        tables['tables'].append({'bbox': np.asarray(predicted_table_rect).reshape(2, 2).tolist(), 'class': table_class_map[mask_idx]})

        if accuracies_per_head[mask_idx] != 1:
            is_save_acc = True
            if not isinstance(predicted_table_rect[0], np.ndarray) and not isinstance(predicted_table_rect[0], list):
                draw.rectangle(predicted_table_rect,
                               outline=colors_pred[mask_idx],
                               width=2)
            else:
                draw.polygon([tuple(x) for x in predicted_table_rect],
                             outline=colors_pred[mask_idx])
    if is_save_acc:
        val_image.save(os.path.join(output_dir, file_name + '.png'))
        from evaluation.debug_output.save_and_plot import plot_tables
        plot_tables(tables, file_path, output_dir)


def get_predicted_rectangles(word_boxes, all_passing_masks,
                             all_passing_masks_weak, all_passing_masks_probs,
                             image_size, is_use_connected_components):
    word_boxes = rescale_word_boxes_to_image_size(word_boxes, image_size)

    pred_boxes = []
    pred_classes = []
    pred_probs = []

    for mask_idx, passing_mask in enumerate(all_passing_masks):
        # Using only sets, rather than dicts with None values, would be more elegant,
        # but alas we need to keep them sorted (if not using connected components),
        # and sets are not ordered.
        rows_to_select = dict.fromkeys(range(len(word_boxes)))

        while rows_to_select:
            row_idx = next(iter(rows_to_select))
            del rows_to_select[row_idx]
            current_row_passing_mask = passing_mask[row_idx]

            if not np.any(current_row_passing_mask):
                # no 1s in the passing mask row so continue
                continue

            if is_use_connected_components:
                # remove rows that are already connected with the current row (Works only if connected component is run earlier)
                to_delete = set(np.where(current_row_passing_mask)[0])
                rows_to_select = {
                    k: None
                    for k in rows_to_select if k not in to_delete
                }
            else:
                # if the same row already appeared in the `passing_mask` before, then there is no need to get bbox around this cluster
                # TODO: if connected components is not run then we should use some sort of NMS, but i think we will always run connected components
                previous_passing_mask = passing_mask[:row_idx]
                if (previous_passing_mask == current_row_passing_mask[None]
                    ).all(axis=1).any():
                    continue

            # get words that should be connected together and save the prediction box around them
            selected_word_boxes = word_boxes[current_row_passing_mask]
            selected_word_boxes_weak = word_boxes[
                all_passing_masks_weak[mask_idx][row_idx]]

            if selected_word_boxes.shape[0] > 0:
                predicted_table_rect = construct_predicted_rect(
                    selected_word_boxes, selected_word_boxes_weak, mask_idx)

                pred_boxes.append(predicted_table_rect.tolist())
                pred_classes.append(mask_idx)
                row_probs = all_passing_masks_probs[mask_idx][row_idx]
                avg_prob = np.mean(row_probs[current_row_passing_mask])
                pred_probs.append(avg_prob)

    return pred_boxes, pred_classes, pred_probs, word_boxes


def validate(model,
             valid_loader,
             output_dir,
             is_save=False,
             is_use_connected_components=True,
             is_augment_in_eval=False):
    correct_preds_total = 0
    num_of_ones_total = 0

    pred_boxes_total = {}
    pred_classes_total = {}
    pred_probs_total = {}

    for word_boxes, contents_idx, target, adjacency_matrices, _, mask, image_size, img_patches, perspective_transform, shadow_mask, crop_size, header_mask in tqdm(
            valid_loader, leave=False, desc='validation'):

        assert word_boxes.size(0) == 1,\
            f'Only batch_size=1, got {word_boxes.size(0)} is supported in validation'

        word_boxes = word_boxes[mask > 0].to(device)
        if img_patches.numel():
            img_patches = img_patches[mask > 0].unsqueeze(0)
        img_patches = img_patches.to(device)
        contents_idx = contents_idx[mask > 0].to(device)

        preds = model(word_boxes.unsqueeze(0),
                      contents_idx.unsqueeze(0),
                      None,
                      img_patches)
        preds = torch.cat(preds) # <number_of_heads, num_words, num_words>
        preds_clustering = torch.sigmoid(preds)

        # we still keep transpose, although we have also weak connections
        preds_clustering_weak = preds_clustering.cpu().numpy().copy()
        # add transpose to get strong connections
        preds_clustering = preds_clustering + preds_clustering.transpose(1, 2)
        preds_clustering = preds_clustering.cpu().numpy()

        num_clustering_heads = preds_clustering.shape[0]
        # <number_of_heads, num_words, num_words>
        adjacency_matrices = adjacency_matrices[0].bool().cpu().numpy()
        shadow_mask = shadow_mask.cpu().numpy()[0]  # scalar, 0 or 1
        header_mask = header_mask.cpu().numpy()[0]  # scalar, 0 or 1

        all_passing_masks = []
        all_passing_masks_weak = []
        all_passing_masks_probs = []
        accuracies_per_head = []

        for head_idx in range(num_clustering_heads):
            if head_idx > 0 and not shadow_mask:
                # continue because we have no labels for the remaining heads/classes
                continue

            # skip header for fintabnet and icdar2019 datasets
            if head_idx == 3 and not header_mask:
                continue

            passing_mask, passing_mask_weak, correct_preds, number_of_ones, accuracy = evaluate_one_head(
                preds_clustering[head_idx], preds_clustering_weak[head_idx],
                adjacency_matrices[head_idx], is_use_connected_components)

            all_passing_masks.append(passing_mask)
            all_passing_masks_weak.append(passing_mask_weak)
            all_passing_masks_probs.append(preds_clustering_weak[head_idx])
            accuracies_per_head.append(accuracy)
            correct_preds_total += correct_preds
            num_of_ones_total += number_of_ones

        pred_boxes, pred_classes, pred_probs, word_boxes = get_predicted_rectangles(
            word_boxes.cpu().numpy(), all_passing_masks, all_passing_masks_weak,
            all_passing_masks_probs, image_size.cpu().numpy(), is_use_connected_components)

        if is_save:
            draw_rectangles(word_boxes, pred_boxes, pred_classes, target,
                            accuracies_per_head, perspective_transform,
                            output_dir, is_augment_in_eval=is_augment_in_eval,
                            crop_size=crop_size.cpu().numpy()[0])

        pred_boxes_total[target['image_id'].item()] = pred_boxes
        pred_classes_total[target['image_id'].item()] = pred_classes
        pred_probs_total[target['image_id'].item()] = pred_probs

    accuracy = correct_preds_total / num_of_ones_total
    print('current accuracy', accuracy)
    return accuracy, pred_boxes_total, pred_classes_total, pred_probs_total


def get_data_paths_and_maps(task, use_dox_datasets):
    if task == 'both':
        dataset_paths = {
            '/data/chargrid_ocr/PubTables-1M/pubtables-1m/pubtables1m/PubTables1M-Structure-PASCAL-VOC': 10,
            '/data/chargrid_ocr/PubTables-1M/pubtables-1m/pubtables1m/PubTables1M-Detection-PASCAL-VOC': 10,
            '/data/chargrid_ocr/pubtabnet/pubtabnet': 10,
            '/data/chargrid_ocr/fintabnet/fintabnet': 10,
            '/data/chargrid_ocr/synthtabnet/sparse': 5,
            '/data/chargrid_ocr/synthtabnet/pubtabnet': 5,
            '/data/chargrid_ocr/synthtabnet/fintabnet': 5
        }

        dataset_paths_validation = {
            '/data/chargrid_ocr/PubTables-1M/pubtables-1m/pubtables1m/PubTables1M-Structure-PASCAL-VOC': 1,
            '/data/chargrid_ocr/PubTables-1M/pubtables-1m/pubtables1m/PubTables1M-Detection-PASCAL-VOC': 1,
            '/data/chargrid_ocr/pubtabnet/pubtabnet': 1,
            '/data/chargrid_ocr/fintabnet/fintabnet': 1,
            '/data/chargrid_ocr/synthtabnet/sparse': 1,
            '/data/chargrid_ocr/synthtabnet/pubtabnet': 1,
            '/data/chargrid_ocr/synthtabnet/fintabnet': 1,
            '/data/chargrid_ocr/icdar2019/ICDAR2019_cTDaR/recognition': 1
        }

        num_clustering_heads = 4
        class_map = {
            'table': 0,
            'table column': 1,
            'column': 1,
            'table row': 2,
            'row': 2,
            'table column header': 3,
            'table header': 3,
            'header': 3,
            'no object': 6
        }

        return dataset_paths, dataset_paths_validation, num_clustering_heads, class_map

    if task == 'detection':
        class_map = {'table': 0, 'table rotated': 0, 'no object': 2}
        dataset_paths = {
            '/data/chargrid_ocr/PubTables-1M/pubtables-1m/pubtables1m/PubTables1M-Detection-PASCAL-VOC': 20,
            '/data/chargrid_ocr/fintabnet/fintabnet': 10,
            '/data/chargrid_ocr/pubtabnet/pubtabnet': 1,
            '/data/chargrid_ocr/synthtabnet/pubtabnet': 1,
            '/data/chargrid_ocr/synthtabnet/fintabnet': 1
        }

        dataset_paths_validation = {
            '/data/chargrid_ocr/PubTables-1M/pubtables-1m/pubtables1m/PubTables1M-Detection-PASCAL-VOC': 1,
            '/data/chargrid_ocr/fintabnet/fintabnet': 1,
            '/data/chargrid_ocr/icdar2019/ICDAR2019_cTDaR/recognition': 1,
            '/data/chargrid_ocr/icdar2019/ICDAR2019_cTDaR/detection': 1
        }

        num_clustering_heads = 1
        return dataset_paths, dataset_paths_validation, num_clustering_heads, class_map

    raise ValueError('Only training both or detection tasks is currently supported')


def load_weights(model, checkpoint_path):
    print('loading weights from checkpoint')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # delete not needed weights

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # load float16 quantized version of the model
        model.load_state_dict(checkpoint, strict=False)
        model = model.float()
    print('Number of parameters in the model:', sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def run_all_validations(model,
                        valid_loader,
                        output_dir,
                        is_use_4_points,
                        prev_best_score=0.0,
                        is_run_4_5_classes=False,
                        is_debug_plot=False,
                        is_augment_in_eval=False):
    model.eval()
    accuracy, pred_boxes, pred_classes, pred_probs = validate(
        model, valid_loader, output_dir, is_save=is_debug_plot, is_augment_in_eval=is_augment_in_eval)
    if not is_use_4_points or (is_use_4_points and not is_augment_in_eval):
        if is_run_4_5_classes:
            validate_coco(valid_loader.dataset,
                          pred_boxes,
                          pred_classes,
                          pred_probs,
                          restrict_to_classes=4)
            validate_coco(valid_loader.dataset,
                          pred_boxes,
                          pred_classes,
                          pred_probs,
                          restrict_to_classes=5)

        average_precision, average_precision_50, average_recall = validate_coco(
            valid_loader.dataset, pred_boxes, pred_classes, pred_probs)
        score = (accuracy + average_precision) / 2.0
    else:
        score = accuracy
        average_precision = 'none'
        average_precision_50 = 'none'
        average_recall = 'none'

    best_score = max(score, prev_best_score)

    out = f'accuracy: {accuracy}, average precision: {average_precision}, average precision IoU:50: {average_precision_50}, \
    average recall: {average_recall}, best: {best_score}'

    with open(f'{output_dir}/log_train.txt', 'a') as log:
        log.write(out + '\n')

    return best_score


def save_model(model, optimizer, output_dir):
    print('saving model')
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, f'{output_dir}/model_and_optimizer_fp32.pth')
    model.half()
    torch.save(model.state_dict(), f'{output_dir}/model_fp16.pth')
    model.float()


def compute_cost(preds_clustering, adjacency_matrices, mask, shadow_mask, header_mask):
    cost = torch.tensor(0.0, device=preds_clustering[0].device)

    for head_idx in range(len(preds_clustering)):
        if head_idx > 0:
            # no loss to calculate further as heads are masked (loss only for the table detection), we can break
            if not torch.sum(shadow_mask):
                break
            mask = mask * torch.unsqueeze(shadow_mask, 1)

        if head_idx == 3:
            if not torch.sum(header_mask):
                break
            mask = mask * torch.unsqueeze(header_mask, 1)

        mask_clustering = torch.unsqueeze(mask, 1)
        mask_clustering = torch.mul(mask_clustering,
                                    mask_clustering.transpose(1, 2))

        cost_tmp = torch.nn.functional.binary_cross_entropy_with_logits(
            preds_clustering[head_idx],
            adjacency_matrices[:, head_idx, :, :],
            reduction='none')
        cost_tmp = torch.mul(cost_tmp, mask_clustering)
        cost_tmp = torch.sum(cost_tmp) / torch.sum(mask_clustering)

        cost += cost_tmp

    return cost


def collate_fn_pad(batch):
    '''Padds batch with elements of variable length'''
    if len(batch) > 1:
        num_of_words = [len(b[0]) for b in batch]
        max_num_of_words = max(num_of_words)

        for b_idx, b in enumerate(batch):
            batch[b_idx] = list(b)  # it was a tuple
            for idx, el in enumerate(b):
                if idx in [0, 1]:
                    # word_boxes, contents_idx
                    if len(el) > 0:
                        batch[b_idx][idx] = TrainTablesDataset.pad_to_max_words(
                            el, max_num_of_words)
                elif idx == 7:
                    # img_patches
                    if len(el) > 0:
                        batch[b_idx][idx] = TrainTablesDataset.pad_to_max_words(
                            el, max_num_of_words * IMAGE_PATCHES_CHANNELS)
                elif idx == 3:
                    # adjacency_matrices
                    new_adj_matrices = []
                    for adjacency_matrix in el:
                        adj_matrix = np.pad(
                            adjacency_matrix,
                            ((0, max_num_of_words - num_of_words[b_idx]),
                             (0, max_num_of_words - num_of_words[b_idx])))
                        new_adj_matrices.append(adj_matrix)
                    batch[b_idx][idx] = np.asarray(new_adj_matrices)
                elif idx == 5:
                    # mask
                    batch[b_idx][idx] = np.zeros(max_num_of_words)
                    batch[b_idx][idx][:num_of_words[b_idx]] = 1
                elif idx == 10:
                    # set crop_size to empty array, it is used only in validation when batch_size = 1
                    batch[b_idx][idx] = np.asarray([])

    return default_collate(batch)


def main(output_dir,
         checkpoint_path=None,
         task='both',
         learning_rate=0.0001,
         eval_set='val',
         ocr_labels_folder='ocr_gt',
         is_use_4_points=False,
         is_use_image_patches=False,
         use_dox_datasets=False,
         use_cell_pointers=True,
         is_augment_in_eval=False,
         use_content_emb=True):

    dataset_paths, dataset_paths_validation, num_clustering_heads, class_map = get_data_paths_and_maps(
        task, use_dox_datasets)

    model = TransformerEncoderTable(num_clustering_heads=num_clustering_heads,
                                    is_use_image_patches=is_use_image_patches,
                                    is_use_4_points=is_use_4_points,
                                    max_doc_size=MAX_DOC_SIZE,
                                    image_patches_channels=IMAGE_PATCHES_CHANNELS,
                                    is_sum_embeddings=True,
                                    use_content_emb=use_content_emb)
    os.makedirs(output_dir, exist_ok=True)
    print('Number of model parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    model.cuda(0)
    train_dataset = TrainTablesDataset(dataset_paths,
                                       class_map=class_map,
                                       num_clustering_heads=num_clustering_heads,
                                       ocr_labels_folder=ocr_labels_folder,
                                       is_use_4_points=is_use_4_points,
                                       is_use_image_patches=is_use_image_patches,
                                       is_one_model=task == 'both',
                                       use_cell_pointers=use_cell_pointers)

    valid_dataset = CocoValidDataset(dataset_paths_validation,
                                     eval_set,
                                     class_map=class_map,
                                     num_clustering_heads=num_clustering_heads,
                                     ocr_labels_folder=ocr_labels_folder,
                                     is_use_4_points=is_use_4_points,
                                     is_use_image_patches=is_use_image_patches,
                                     is_one_model=task == 'both',
                                     use_cell_pointers=use_cell_pointers,
                                     is_augment_in_eval=is_augment_in_eval)

    print('num_clustering_heads', num_clustering_heads)

    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=8,
                              drop_last=True,
                              collate_fn=collate_fn_pad)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=8,
                              drop_last=False,
                              collate_fn=collate_fn_pad)

    if checkpoint_path:
        load_weights(model, checkpoint_path)
        model.to(device)
        best_score = run_all_validations(model,
                                         valid_loader,
                                         output_dir,
                                         is_use_4_points=is_use_4_points,
                                         is_run_4_5_classes=False,
                                         is_debug_plot=True,
                                         is_augment_in_eval=is_augment_in_eval)

    train_iterator = iter(train_loader)

    EPOCHS = 200
    STEPS_PER_EPOCH = 5000
    best_score = 0.0

    for epoch in trange(EPOCHS):
        model.train()
        progress_bar = trange(STEPS_PER_EPOCH, leave=False, desc='Train')

        for _ in progress_bar:
            try:
                word_boxes, contents_idx, _, adjacency_matrices, _, mask, _, img_patches, _, shadow_mask, _, header_mask = next(
                    train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                word_boxes, contents_idx, _, adjacency_matrices, _, mask, _, img_patches, _, shadow_mask, _, header_mask = next(
                    train_iterator)

            word_boxes = word_boxes.to(device)
            img_patches = img_patches.to(device)
            contents_idx = contents_idx.to(device)
            mask = mask.to(device).float()
            shadow_mask = shadow_mask.to(device).float()
            header_mask = header_mask.to(device).float()
            adjacency_matrices = adjacency_matrices.to(device)

            preds_clustering = model(word_boxes, contents_idx, mask,
                                     img_patches)

            cost = compute_cost(preds_clustering, adjacency_matrices, mask,
                    shadow_mask, header_mask)

            progress_bar.set_description(f'Train loss {cost:.6f}')

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            model.zero_grad()

        # To plot debug images every 5 epochs use this parameter:
        is_debug_plot = epoch % 5 == 0 and epoch > 0
        new_best_score = run_all_validations(model,
                                             valid_loader,
                                             output_dir,
                                             prev_best_score=best_score,
                                             is_use_4_points=is_use_4_points,
                                             is_run_4_5_classes=False,
                                             is_debug_plot=False,
                                             is_augment_in_eval=is_augment_in_eval)
        if new_best_score > best_score:
            best_score = new_best_score
            save_model(model, optimizer, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output_dir',
                        default='/data/tmp/line_grouping')
    parser.add_argument('-c', '--checkpoint_path', default=None)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--eval_set', default='val', choices=['val', 'test'])
    parser.add_argument('--ocr_labels_folder',
                        default='ocr_gt',
                        choices=['ocr_gt', 'ocr'])
    parser.add_argument('-t',
                        '--task',
                        default='both',
                        choices=['detection', 'recognition', 'both'])
    parser.add_argument('--is_use_4_points', action='store_true')
    parser.add_argument('--is_augment_in_eval', action='store_true')
    parser.add_argument('--is_use_image_patches', action='store_true')
    parser.add_argument('--use_dox_datasets', action='store_true')
    parser.add_argument('--no_cell_pointers', action='store_true')
    parser.add_argument('--no_content_emb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('====================')
    pprint(vars(args))
    print('====================')
    main(args.output_dir,
         checkpoint_path=args.checkpoint_path,
         task=args.task,
         learning_rate=args.learning_rate,
         eval_set=args.eval_set,
         ocr_labels_folder=args.ocr_labels_folder,
         is_use_4_points=args.is_use_4_points,
         is_use_image_patches=args.is_use_image_patches,
         use_dox_datasets=args.use_dox_datasets,
         use_cell_pointers=not args.no_cell_pointers,
         is_augment_in_eval=args.is_augment_in_eval,
         use_content_emb=not args.no_content_emb)
