"""
This file is used to parse the train.json, extract the captions and image_paths
"""
import json


def extract_annotations_to_txt(json_path, image_root):
    """
    extract image paths and correspondent annotations
    :param json_path: the json file stores the annotations
    :param image_root: the directory stores all images
    :return: a list in which the keys are image_paths, the values are lists containing all 5 annotations for the image
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    annotations = []
    for item in data['images']:
        image_id = item['id']
        file_name = f"{image_root}/{item['file_name']}"

        # get all captions of the image
        captions = [
            anno['caption'].replace("\n", " ")
            for anno in data['annotations']
            if anno['image_id'] == image_id
        ]

        # group the captions
        for caption in captions:
            annotations.append(f"{file_name}\t{caption}")

    return annotations


def save_to_txt(annotations, output_path):
    """
    Save the captions extracted from json to txt, in 'image_path\tcaption' strings format
    :param annotations: extracted 'image_path\tcaption' strings, a list
    :param output_path: txt file path
    :return: none
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in annotations:
            f.write(entry + "\n")