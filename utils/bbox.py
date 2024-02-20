import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from shapely.geometry import Polygon


def _ioa_or_iou_batch_not_rotated(b1, b2, is_iou, max_ioa=False):
    assert b1.ndim == 2 and b2.ndim == 2 and b1.shape[1] == 4 and b2.shape[1] == 4

    b1 = np.array(b1)
    b2 = np.array(b2)

    b1 = b1[:, :4].reshape(-1, 1, 4)
    b2 = b2[:, :4].reshape(1, -1, 4)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(b1[:, :, 0], b2[:, :, 0])
    yA = np.maximum(b1[:, :, 1], b2[:, :, 1])
    xB = np.minimum(b1[:, :, 2], b2[:, :, 2])
    yB = np.minimum(b1[:, :, 3], b2[:, :, 3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    boxAArea = (b1[:, :, 2] - b1[:, :, 0]) * (b1[:, :, 3] - b1[:, :, 1])

    if is_iou:
        boxBArea = (b2[:, :, 2] - b2[:, :, 0]) * (b2[:, :, 3] - b2[:, :, 1])
        denominator = boxAArea + boxBArea - interArea
    elif max_ioa:
        boxBArea = (b2[:, :, 2] - b2[:, :, 0]) * (b2[:, :, 3] - b2[:, :, 1])
        denominator = np.minimum(boxAArea, boxBArea)
    else:
        denominator = boxAArea

    with np.errstate(divide='ignore', invalid='ignore'):
        out = interArea / denominator

    if np.any(denominator.flatten() == 0):
        out[~np.isfinite(out)] = 0

    return out


def compute_intersection_batch_parallel(b1, b2):
    """
    Given two arrays of non-rotated boxes, returns the intersection matrix. This is slightly faster than IoA or IoU...
    :param b1: array-like, of shape (n, 2, 2) or (n, 4)
    :param b2: array-like, of shape (m, 2, 2) or (m, 4)
    :return: array of shape (n, m)
    """
    b1 = np.asarray(b1)
    b2 = np.asarray(b2)
    n = b1.shape[0]
    m = b2.shape[0]

    assert b1.size == n * 4 and b2.size == m * 4, 'Expecting arrays of shape (*, 4) or (*, 2, 2), instead got {} and ' \
                                                  '{}'.format(b1.shape, b2.shape)

    b1 = b1[:, :4].reshape(-1, 1, 4)
    b2 = b2[:, :4].reshape(1, -1, 4)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(b1[:, :, 0], b2[:, :, 0])
    yA = np.maximum(b1[:, :, 1], b2[:, :, 1])
    xB = np.minimum(b1[:, :, 2], b2[:, :, 2])
    yB = np.minimum(b1[:, :, 3], b2[:, :, 3])

    # compute the area of intersection rectangle
    return np.maximum(0, xB - xA) * np.maximum(0, yB - yA)


def bounding_boxes_of_rotated_boxes(b):
    """
    Given a list of possibly rotated bounding boxes, returns their non-rotated bounding boxes (identical if input was
    not rotated, larger otherwise). These are given as (x1, y1, x2, y2) coords.
    :param b: (N, 4, 2)
    :return: (N, 4)
    """
    if len(b) == 0:
        return np.empty((0, 4))

    assert b.shape[1:] == (4, 2), 'Expecting a (*, 4, 2) input, instead got {}'.format(b.shape)
    x1, y1 = b.min(axis=1).T
    x2, y2 = b.max(axis=1).T

    return np.array([x1, y1, x2, y2]).T


def estimate_min_angle(b):
    """
    Given a list of rotated bounding boxes, estimates one of the two angles along which the rectangle develops.
    It always picks the angle that is closest to 0 degrees (among the 4 possible ones)
    :param b: 3D array of shape (N, 4, 2). Points must be sequentially ordered (clockwise or counter-clockwise)
    :return: 2D array of shape (N, 2); for each box, the two angles.
    """
    # Any difference between two consecutive edges will do; since it's a rectangle, all the other ones are the same up
    # to summing some pi/2.
    delta = b[:, 1] - b[:, 0]

    # Estimate one of the angles below the edge of the rectangle. It will have a pretty random orientation.
    # It is a 1d vector of shape (N, ).
    one_angle = np.arctan2(delta[:, 0], delta[:, 1])

    # We want to have it between -pi/4 and pi/4, so we translate, take mod, and translate back.
    return (one_angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


def _compute_ioa_or_iou_rotated(b1, b2, is_iou, max_ioa=False):
    """
    Given two boxes (of which at least one rotated), computes the IoA or IoU using the accurate - but slower - shapely
    method.
    :param b1: 2D array of shape (4, 2) with the coordinates of the box
    :param b2: 2D array of shape (4, 2) with the coordinates of the box
    :return: A scalar, either the IoA or the IoU.
    """
    a = Polygon(b1)
    b = Polygon(b2)

    try:
        area_int = a.intersection(b).area
    except Exception:
        return 0.0

    if area_int == 0.0:
        # Avoid issues if a.area == 0...
        return 0.0

    if not is_iou:
        if max_ioa:
            return area_int / np.minimum(a.area, b.area)
        else:
            return area_int / a.area

    # Taking intersection is the actually slowest part, so this is twice as fast as using a.union(b)
    if (a.area + b.area - area_int) == 0:
        return 0

    return area_int / (a.area + b.area - area_int)


def _boxes_rotated_more_than(b, degree_threshold=5):
    """
    Given a list of boxes, check which ones are rotated by more than the given threshold of degrees
    :param b: array (N, 4, 2). Points must be sequentially ordered (clockwise or counter-clockwise)
    :param degree_threshold: scalar (number of degrees)
    :return: 1d-array of length N and type bool.
    """
    min_angle = estimate_min_angle(b) * 180 / np.pi
    return np.abs(min_angle) > degree_threshold


def _ioa_or_iou_batch_rotated(b1, b2, is_iou, degree_threshold=5, max_ioa=False):
    """
    Internal function to compute the IoA or IoU between two lists of (possibly rotated) boxes, specified with coords of
    all 4 corners.
    :param b1: (N, 4, 2). Points must be sequentially ordered (clockwise or counter-clockwise)
    :param b2: (M, 4, 2). Points must be sequentially ordered (clockwise or counter-clockwise)
    :param is_iou: bool
    :param degree_threshold: How many degrees are still considered "horizontal" (if so, exclude from the lengthier
        exact computation for small rotation)
    :return: (N, M)
    """
    assert b1.shape[1:] == (4, 2) and b2.shape[1:] == (4, 2)

    # First compute the IoA / IoU between the (non-rotated) bounding boxes. If these don't overlap, no need to waste
    # more computations!
    upper_bound = _ioa_or_iou_batch_not_rotated(bounding_boxes_of_rotated_boxes(b1),
                                                bounding_boxes_of_rotated_boxes(b2), is_iou, max_ioa)

    # We compute the intersection with the more accurate method only if one of the two bounding boxes is rotated by more
    # than "degree_threhsold", to save time.
    one_is_rotated = np.logical_or(_boxes_rotated_more_than(b1, degree_threshold).reshape(-1, 1),
                                   _boxes_rotated_more_than(b2, degree_threshold).reshape(1, -1))

    # The exact computation can only be done one pair of rectangles at a time with a for loop, so we need a where to get
    # the indices to recompute; these are the rotated ones which have any possibility of intersecting at all.
    to_recompute = np.where(np.logical_and(one_is_rotated, upper_bound > 0))

    result = upper_bound.copy()

    for i, j in zip(*to_recompute):
        result[i, j] = _compute_ioa_or_iou_rotated(b1[i], b2[j], is_iou, max_ioa)

    return result


def ensure_4_vertices(b):
    """
    Given a list of boxes, ensures to put it in 4 vertices (N, 4, 2) format, if necessary by repeating.
    :param b: Either a 2D array of shape (n, 4), or a 3D of shape (n, 2, 2) or (n, 4, 2) if we want to deal with
        rotated boxes.
    :return: An array of shape (n, 4, 2).
    """
    if b.ndim == 3 and b.shape[1] == 4 and b.shape[2] == 2:
        return b

    assert b.size == len(b) * 4

    x1, y1, x2, y2 = b.reshape(-1, 4).T

    # Staking them together creates a (4, 2, N) array...
    result = np.array([
        [x1, y1],
        [x1, y2],
        [x2, y2],
        [x2, y1]
    ])

    # ... therefore we have to transpose
    return result.transpose([2, 0, 1])


def _compute_ioa_or_iou_batch(b1, b2, is_iou, degree_threshold=5, max_ioa=False):
    """
    Unified function to compute either IoA or IoU without repeating the nearly identical code...
    :param b1:
    :param b2:
    :param is_iou:
    :return:
    """
    if min(len(b1), len(b2)) == 0:
        return np.empty((len(b1), len(b2)), dtype=np.float32)

    assert b1.ndim in [2, 3] and b2.ndim in [2, 3]

    n = len(b1)
    m = len(b2)
    if b1.size == 4 * n and b2.size == 4 * m:
        return _ioa_or_iou_batch_not_rotated(b1.reshape(-1, 4), b2.reshape(-1, 4), is_iou, max_ioa)

    # From here on, at least one of the two must be rotated. We call _ensure_4_vertices to align the format to
    # (*, 4, 2) for both b1 and b2 (at least one of the two must already be)
    assert (b1.ndim == 3 and b1.shape[1:] == (4, 2)) or (b2.ndim == 3 and b2.shape[1:] == (4, 2)), \
        'Was expecting one of the bboxes to be of shape (*, 4, 2) but failed. b1 has shape {} which is not 4 * {}.' \
        'b2 has shape {} which is not 4 * {}. And neither is ({}, 4, 2) or ({}, 4, 2), respectively'.format(
            b1.shape, n, b2.shape, m, n, m)

    return _ioa_or_iou_batch_rotated(ensure_4_vertices(b1), ensure_4_vertices(b2), is_iou, degree_threshold, max_ioa)


def compute_ioa_batch(b1, b2, degree_threshold=5, max_ioa=False):
    """
    Given two sets of bounding boxes (say of length n and m), returns a rectangular n x m matrix with the pairwise
    Intersection over Area (of b1).
    It uses numpy, O(n*m) time.
    :param b1: Either a 2D array of shape (n, 4), or a 3D of shape (n, 2, 2) or (n, 4, 2) if we want to deal with
        rotated boxes.
    :param b2: Either a 2D array of shape (m, 4), or a 3D of shape (m, 2, 2) or (m, 4, 2) if we want to deal with
        rotated boxes.
    :return: A 2D numpy array of shape (n, m)
    """
    return _compute_ioa_or_iou_batch(np.array(b1), np.array(b2), False, degree_threshold, max_ioa)


def compute_iou_batch(b1, b2, degree_threshold=5):
    """
    Given two sets of bounding boxes (say of length n and m), returns a rectangular n x m matrix with the pairwise
    Intersection over Union.
    It uses numpy, O(n*m) time.
    :param b1: Either a 2D array of shape (n, 4), or a 3D of shape (n, 2, 2) or (n, 4, 2) if we want to deal with
        rotated boxes.
    :param b2: Either a 2D array of shape (m, 4), or a 3D of shape (m, 2, 2) or (m, 4, 2) if we want to deal with
        rotated boxes.
    :param degree_threshold: num of degrees below which we don't use real rotated angles, but only a non-rotated
        bounding box, for speed sake
    :return: A 2D numpy array of shape (n, m)
    """
    return _compute_ioa_or_iou_batch(np.array(b1), np.array(b2), True, degree_threshold=degree_threshold)


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    Taken from https://stackoverflow.com/questions/13542855/python-help-to-implement-an-algorithm-to-find-the-minimum-area-rectangle-for-gi
    :param points: an nx2 matrix of coordinates
    :return: a 4x2 matrix of coordinates
    """
    if len(points) == 0:
        print('Warning: empty collection passed to minimum_bounding_rectangle, returning also empty.')
        return np.empty((0, 2))

    pi2 = np.pi / 2.

    points = np.asarray(points)

    # get the convex hull for the points
    try:
        hull_points = points[ConvexHull(points).vertices]
    except QhullError:
        # This happens if we have a "flat" rectangle (e.g. all x are the same). In that case, return a trivial rectangle
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        return np.asarray([
            [x1, y1],
            [x1, y2],
            [x2, y2],
            [x2, y1]
        ])

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # [
    # [cos(t), -sin(t)],
    # [sin(t), cos(t)]
    # ]
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def horizontal_rectangle_from_end_points_and_height(x_left, x_right, y_left, y_right, height_above, height_below):
    alpha = np.arctan2(y_right - y_left, x_right - x_left)
    line_box = [
        [x_right + height_below * np.sin(alpha), y_right - height_below * np.cos(alpha)],  # Bottom-right
        [x_left + height_below * np.sin(alpha), y_left - height_below * np.cos(alpha)],  # Bottom-left
        [x_left - height_above * np.sin(alpha), y_left + height_above * np.cos(alpha)],  # Top-left
        [x_right - height_above * np.sin(alpha), y_right + height_above * np.cos(alpha)],  # Top-right
    ]
    return line_box


def points_belong_to_rotated_rectangles(points, rectangles, degree_threshold=1):
    """
    For each point, returns whether it belongs to each rectangle.
    This is done by taking the scalar product between the center and the normal of each edge, and checking that it is
    positive - which means that the point is in the "inner" side of that edge.
    However, if the rectangles are not *really* rotated, it takes the bounding boxes invokes
    points_belong_to_not_rotated_rectangles, which is faster.
    :param points: array of shape (N, 2), format x, y
    :param rectangles: array of shape (M, 4, 2) (coordinates of the four edges)
    :return: numpy array of shape (N, M) and type bool.
    """

    # We first split the rectangles in "really" rotated and "not really" ones. For the latter, we use the faster method.
    is_rotated = _boxes_rotated_more_than(rectangles, degree_threshold)

    result = np.empty((len(points), len(rectangles)), dtype=bool)
    result[:, is_rotated] = points_belong_to_seriously_rotated_rectangles(points, rectangles[is_rotated])
    not_really_rotated = rectangles[~is_rotated]
    bounding_rotated = np.stack([not_really_rotated.min(axis=1), not_really_rotated.max(axis=1)], axis=1)
    result[:, ~is_rotated] = points_belong_to_not_rotated_rectangles(points, bounding_rotated)

    return result


def points_belong_to_not_rotated_rectangles(points, rectangles):
    """
    For each point, returns whether it belongs to each rectangle.
    This is done simply by checking if the coordinates are between the extremes.
    :param points: array of shape (N, 2), format x, y
    :param rectangles: array of shape (M, 4), format x1, y1, x2, y2
    :return: numpy array of shape (N, M) and type bool.
    """
    rectangles = rectangles.reshape(1, -1, 4)
    cond_left = points[:, [0]] >= rectangles[:, :, 0]
    cond_top = points[:, [1]] >= rectangles[:, :, 1]
    cond_right = points[:, [0]] <= rectangles[:, :, 2]
    cond_bottom = points[:, [1]] <= rectangles[:, :, 3]

    return cond_left & cond_top & cond_right & cond_bottom


def points_belong_to_rectangles(points, rectangles, degree_threshold=1):
    """
    Given a list of points and a list of (possibly rotated) rectangles, computes the boolean matrix of whether each
    point is inside each rectangle.
    (Should actually work for any convex quadrilaters, not just rectangles)
    Warning: not assured to work consistently if the points are allowed to lie exactly on the edges - but it's fine if
    they are really floats, as in the current application scenario.
    :param points: numpy array of shape (N, 2), format x, y
    :param rectangles: numpy array of shape either (M, 4) or (M, 2, 2), or (M, 4, 2).
    :return: numpy array of shape (N, M) and type bool.
    """
    assert points.ndim == 2 and points.shape[1] == 2, 'Expecting points to be of shape (N, 2), instead got {}'.format(
        points.shape)
    if rectangles.ndim == 2:
        assert rectangles.shape[1] == 4, 'Got 2d rectangles but instead of shape (M, 4) got {}'.format(rectangles.shape)
    else:
        assert rectangles.ndim == 3, 'Expecting rectangles to be 2d or 3d, got instead {}'.format(rectangles.shape)
        assert rectangles.shape[1] in [2, 4] and rectangles.shape[2] == 2, \
            'Expecting rectangles to be one of (M, 4), (M, 2, 2), or (M, 4, 2), got instead {}'.format(rectangles.shape)

    if rectangles.ndim == 3 and rectangles.shape[1] == 4:
        return points_belong_to_rotated_rectangles(points, rectangles, degree_threshold)

    return points_belong_to_not_rotated_rectangles(points, rectangles.reshape(-1, 4))


def anticlockwise_vertices_from_top_left(rectangle):
    # First sort by x
    p1, p2, p3, p4 = rectangle[np.argsort(rectangle[:, 0])]
    top_left, bottom_left = sorted([p1, p2], key=lambda x: x[1])
    top_right, bottom_right = sorted([p3, p4], key=lambda x: x[1])
    return top_left, bottom_left, bottom_right, top_right


def anticlockwise_vertices_from_top_left_vertical(rectangle):
    p1, p2, p3, p4 = rectangle[np.argsort(rectangle[:, 1])]
    top_left, top_right = sorted([p1, p2], key=lambda x: x[0])
    bottom_left, bottom_right = sorted([p3, p4], key=lambda x: x[0])
    return top_left, bottom_left, bottom_right, top_right


def clockwise_vertices_from_top_left(rectangle):
    top_left, bottom_left, bottom_right, top_right = anticlockwise_vertices_from_top_left(rectangle)
    return top_left, top_right, bottom_right, bottom_left


def clockwise_vertices_from_top_left_vertical(rectangle):
    top_left, bottom_left, bottom_right, top_right = anticlockwise_vertices_from_top_left_vertical(rectangle)
    return top_left, top_right, bottom_right, bottom_left


def optimized_minimum_bounding_rectangle(box, rotation):
    if rotation in [0, 180]:
        top_left, bottom_left, bottom_right, top_right = anticlockwise_vertices_from_top_left(box)
        x_center_left = (top_left[0] + bottom_left[0]) / 2.0
        x_center_right = (top_right[0] + bottom_right[0]) / 2.0
        y_center_left = (top_left[1] + bottom_left[1]) / 2.0
        y_center_right = (top_right[1] + bottom_right[1]) / 2.0

        height = (np.abs(bottom_left[1] - top_left[1]) + np.abs(bottom_right[1] - top_right[1])) / 4
        bounding_box = horizontal_rectangle_from_end_points_and_height(x_center_left, x_center_right, y_center_left,
                                                                       y_center_right, height, height)
    else:
        top_left, bottom_left, bottom_right, top_right = anticlockwise_vertices_from_top_left_vertical(box)
        x_center_top = (top_left[0] + top_right[0]) / 2.0
        x_center_bottom = (bottom_left[0] + bottom_right[0]) / 2.0
        y_center_top = (top_left[1] + top_right[1]) / 2.0
        y_center_bottom = (bottom_left[1] + bottom_right[1]) / 2.0

        width = (np.abs(bottom_right[0] - bottom_left[0]) + np.abs(top_right[0] - top_left[0])) / 4
        bounding_box = horizontal_rectangle_from_end_points_and_height(y_center_top, y_center_bottom, x_center_top,
                                                                       x_center_bottom, width, width)
        bounding_box = [xy[::-1] for xy in bounding_box]
    return bounding_box
