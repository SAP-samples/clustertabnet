import re
import unicodedata
import unidecode
import numpy as np


DEFAULT_STRIP_PUNCTUATIONS = ",.?!\"|[](){}~@#$%^&*-_+="


# copied from charmer project
def charmer_word_normalization(word: str, strip_punctuations: str = DEFAULT_STRIP_PUNCTUATIONS):
    """Normalizes a word to increase the chances of finding it in the word_map.
    The function normalizes a word in three steps:
        (1) lower case
        (2) remove leading and trailing punctuations
        (3) map all digits to 1
    Args:
        word (str): The unnormalized word.
        strip_punctuations (str, optional): A string of punctuations to strip. Defaults to ",.?!"|{}~@#$%^&*-_+=".
    Returns:
        str: The normalized word.
    """
    # (1) normalize unicode characters (e.g converts "３１３８０" to "31380")
    word = unicodedata.normalize('NFKC', word)
    # (3) try to remove punctuations on both sides of a word if there is something left
    word = word.strip(strip_punctuations) or word

    word = unidecode.unidecode(word)
    word = word.replace(' ', '')
    # (4) map all numbers to 1 as the signature
    word = re.sub(r'\d', '1', word)
    # (4) map all chars [a-z] to a
    word = re.sub(r'[a-z]', 'a', word)
    # (4) map all chars [A-Z] to A
    word = re.sub(r'[A-Z]', 'A', word)
    # replace any non-alphabetic character with a '.'
    word = re.sub(r'[^0-9a-zA-Z]', '.', word)
    return word


def get_angle(id, boxes):
    diff = boxes[:, id + 1, :] - boxes[:, id, :]
    y_over_x = diff[:, 1] / np.where(diff[:, 0] == 0, 1e-10, diff[:, 0])
    return np.arctan(y_over_x)


def get_side_length(id, boxes):
    if boxes.shape[1] == 4:
        # boxes has 4 vertices
        diff = boxes[:, id + 1, :] - boxes[:, id, :]
        return np.linalg.norm(diff, axis=1)

    # boxes has 2 vertices; return difference in x or y depending on id.
    return boxes[:, 1, id] - boxes[:, 0, id]


def get_width_height_min_angle(boxes):
    if boxes.shape[1] == 4:
        # Get angles of rotation from the box. Since they're rectangles, we just need to compute 2 angles between
        # consecutive sides, the others will have the same arctan.
        angles = [get_angle(0, boxes), get_angle(1, boxes)]
        first_angle_is_smaller = np.abs(angles[0]) < np.abs(angles[1])
        min_angle = np.where(first_angle_is_smaller, angles[0], angles[1])
    else:
        assert boxes.shape[1] == 2, 'Expecting either 4 or 2 vertices (second shape dim), got shape {}'.format(
            boxes.shape)
        # In this case, the box must be horizontally aligned.
        min_angle = 0.0
        first_angle_is_smaller = np.full(len(boxes), True, bool)

    # Get word width or height
    sides = np.array([get_side_length(0, boxes), get_side_length(1, boxes)])
    width = np.where(first_angle_is_smaller, sides[0], sides[1])
    height = np.where(first_angle_is_smaller, sides[1], sides[0])
    return width, height, min_angle
