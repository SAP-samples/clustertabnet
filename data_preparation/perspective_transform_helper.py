import numpy as np
import cv2


class PerspectiveTransformHelper:

    @staticmethod
    def get_perspective_transform(image_size, four_points):
        width, height = image_size
        target_points = np.asarray([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        return cv2.getPerspectiveTransform(four_points.astype(np.float32), target_points.astype(np.float32))

    @staticmethod
    def transform_line_boxes(line_boxes, perspective_transform):
        assert line_boxes.ndim == 3 and line_boxes.shape[1] == 4 and line_boxes.shape[2] == 2, \
            f'Got line boxes of shape {line_boxes.shape}'
        assert perspective_transform.shape == (3, 3), f'Got perspective_transform of shape {perspective_transform.shape}'

        extended_line_boxes = np.concatenate([line_boxes, np.ones_like(line_boxes[:, :, :1])], axis=2)
        transformed_line_boxes = np.matmul(perspective_transform, np.transpose(extended_line_boxes, [0, 2, 1]))
        transformed_line_boxes = np.transpose(transformed_line_boxes, [0, 2, 1])
        return transformed_line_boxes[:, :, :2] / transformed_line_boxes[:, :, 2:]
