'''
modified from: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
GitHub: https://github.com/tensorflow/models
For evaluating SSD-ResNet50 on COCO2017_1000, TF_version=2.0
'''
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import pickle

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from tqdm import tqdm


PICK_LIST = []
BAN_LIST = []

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"
    model_dir_str = str(model_dir)
    model = tf.saved_model.load(model_dir_str)
    model = model.signatures['serving_default']

    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        
    return output_dict

def show_inference(model, dataset_dir, output_dir, folder, image_name):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_name_noext = os.path.splitext(image_name)[0]
    image_path = os.path.join(dataset_dir, folder, image_name)
    image_np = np.array(Image.open(image_path).convert('RGB'))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    with open(os.path.join(output_dir, folder, image_name_noext + '.pkl'), 'wb') as f:
        pickle.dump(output_dict, f)

if __name__ == "__main__":
    dataset_dir = "/data1/yantao/DR_datasets/COCO2017_1000"
    output_dir = '/home/yantao/output_det_dir'
    if os.path.exists(output_dir):
        raise
    os.mkdir(output_dir)
    
    test_folders = []
    for temp_folder in os.listdir(dataset_dir):
        if not os.path.isdir(os.path.join(dataset_dir, temp_folder)):
            continue 
        if temp_folder == 'imagenet_val_5000' or temp_folder == '.git' or temp_folder == '_annotations' or temp_folder == '_segmentations':
            continue 
        if len(PICK_LIST) != 0 and temp_folder not in PICK_LIST:
            continue
        if len(BAN_LIST) != 0 and temp_folder in BAN_LIST:
            continue
        test_folders.append(temp_folder)

    model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    detection_model = load_model(model_name)

    for folder in tqdm(test_folders):
        print(folder)
        os.mkdir(os.path.join(output_dir, folder))
        images_name = os.listdir(os.path.join(dataset_dir, folder))
        for image_name in tqdm(images_name):
            show_inference(detection_model, dataset_dir, output_dir, folder, image_name)