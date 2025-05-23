"""
This python file is used to extract the annotations of each image, store in txt
"""

import json


def extract_annotations_to_txt(json_path, image_root):
    """
    extract image paths and correspondent annotations
    :param json_path: the json file stores the annotations
    :param image_root: the directory stores all images
    :return: a list in which the keys are image_root, the values are lists containing all 5 annotations for the
    image
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = []
    # for item in data['images']:
    for i in range(100):
        item = data['images'][i]
        image_id = item['id']
        file_name = f"{image_root}/{item['file_name']}"

        # get all captions of the image
        captions = [
            anno['caption'].replace("\n", " ")
            for anno in data['annotations']
            if anno['image_id'] == image_id
        ]
        # concate the captions
        captions_str = "\t".join(captions)

        annotations.append(f"{file_name}\t{captions_str}")

    return annotations


def save_to_txt(annotations, output_path):
    """
    Save the captions extracted from json to txt, in 'image_root\tcaption' strings format
    :param annotations: extracted 'image_root\tcaption' strings, a list
    :param output_path: txt file path
    :return: none
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in annotations:
            f.write(entry + "\n")

