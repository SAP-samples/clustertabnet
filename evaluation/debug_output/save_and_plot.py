import os
from io import BytesIO
import numpy as np

from PIL import Image, ImageDraw, ImageOps
import cv2


EXIF_ORIENTATION_KEY = 0x0112


def rotate_based_on_exif(img):
    """
    Rotate image based on EXIF tag from Image
    :param img: PIL Image object
    :return: rotated PIL Imagee
    """
    try:
        if img.getexif() is not None and EXIF_ORIENTATION_KEY in img.getexif().keys():
            exif = {k: v for k, v in img.getexif().items()}
            if exif[EXIF_ORIENTATION_KEY] != 1:
                img = ImageOps.exif_transpose(img)
    except Exception as e:
        print(e)
    return img


def get_image_and_mask(image, image_filename):
    if image is not None:
        image = Image.fromarray(np.uint8(np.squeeze(image, axis=2) * 255)).convert('RGBA')
    else:
        if isinstance(image_filename, str):
            image = Image.open(image_filename).convert('RGBA')
        else:
            # process image without dumping on disc
            image = Image.open(BytesIO(image_filename)).convert('RGBA')
    image = rotate_based_on_exif(image)
    mask = Image.new('RGBA', image.size, color=(255, 255, 255, 0))
    return image, mask


def thumbnail(image, max_size=1500):
    if image.width > image.height:
        new_size = (max_size, int(max_size * image.height / image.width))
    else:
        new_size = (int(max_size * image.width / image.height), max_size)
    return image.resize(new_size)


def save_thumbnail_of_composite(image, mask, image_filename, output_dir, extension):
    result = Image.alpha_composite(image, mask).convert('RGB')
    if output_dir:
        basename = str(os.path.basename(image_filename))
        basename = basename[::-1].split('.', 1)[-1][::-1]
        # Resize to save disk space; at most 1500 pixels per side.
        thumbnail(result).save(os.path.join(output_dir, basename) + extension)
    else:
        return result


def draw_box(bbox, fill, outline, draw, width=1):
    points = tuple(x for y in bbox for x in y)
    if len(points) == 4:
        draw.rectangle(points, fill, outline, width=width)
    else:
        draw.polygon(points, fill, outline)
        draw.line(points + points[:2], fill=outline, width=5)


def plot_tables(out_dict, image_filename, output_dir, image=None, extension='.table.jpg'):
    image, mask = get_image_and_mask(image, image_filename)
    draw = ImageDraw.Draw(mask, 'RGBA')
    table_colors = {'table': [0, [255, 0, 0]],
                    'column': [5, [0, 128, 0]],
                    'row': [0, [0, 0, 255]],
                    'header': [7, [255, 165, 0]]
                   }

    if out_dict.get('tables'):
        transparency = 60
        for table_idx, table in enumerate(out_dict['tables']):
            color_shift, color = table_colors[table['class']]

            draw_box(table['bbox'], tuple(color + [transparency]), tuple(color + [255]), draw, width=1)
            if table_idx < len(out_dict['tables'])-1 and table['class'] != out_dict['tables'][table_idx+1]['class'] \
                and table['class'] == 'column':
                transparency -= 40
                image = Image.alpha_composite(image, mask)

    if out_dict.get('tables') is not None:
        return save_thumbnail_of_composite(image, mask, image_filename, output_dir, extension)
