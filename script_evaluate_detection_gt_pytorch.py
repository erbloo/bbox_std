import os
import sys
import shutil
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import torchvision
import torch
import pickle
from pycocotools.coco import COCO
import datetime

from utils.image_utils import load_image, save_image, save_bbox_img
from utils.mAP import save_detection_to_file, calculate_mAP_from_files
from utils.torch_utils import numpy_to_variable, variable_to_numpy, convert_torch_det_output
from utils.VOC2012_1000.annotation_loader import load_annotations as load_voc_annotations
from utils.COCO2017_1000.annotation_loader import load_annotations as load_coco_annotations


# {voc_idx : coco_idx}
with open('utils/VOC_AND_COCO91_CLASSES.pkl', 'rb') as f:
    VOC_AND_COCO91_CLASSES = pickle.load(f)
with open('utils/VOC_AND_COCO80_CLASSES.pkl', 'rb') as f:
    VOC_AND_COCO80_CLASSES = pickle.load(f)


PICK_LIST = []
BAN_LIST = []

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for generating adversarial examples.')
    parser.add_argument('test_model', help='Model for testing AEs.', type=str)
    parser.add_argument('dataset_type', choices=['coco', 'voc'], help='Dataset for testing AEs.', type=str)
    parser.add_argument('--dataset-dir', help='Dataset folder path.', default='/home/yantao/workspace/datasets/imagenet5000', type=str)

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    if args.dataset_type == 'voc':
        gt_dir = os.path.join(args.dataset_dir, '_annotations')
    elif args.dataset_type == 'coco':
        gt_loader = COCO(os.path.join(args.dataset_dir, 'instances_val2017.json'))

    if args.test_model == 'fasterrcnn':
        test_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
        img_size = (416, 416)
    elif args.test_model == 'maskrcnn':
        test_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
        img_size = (416, 416)
    elif args.test_model == 'keypointrcnn':
        test_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).cuda().eval()
        img_size = (416, 416)
    else:
        raise ValueError('Invalid test_model {0}'.format(args.test_model))

    test_folders = []
    for temp_folder in os.listdir(args.dataset_dir):
        if not os.path.isdir(os.path.join(args.dataset_dir, temp_folder)):
            continue 
        if temp_folder == 'imagenet_val_5000' or temp_folder == '.git' or temp_folder == '_annotations' or temp_folder == '_segmentations':
            continue 
        if len(PICK_LIST) != 0 and temp_folder not in PICK_LIST:
            continue
        if len(BAN_LIST) != 0 and temp_folder in BAN_LIST:
            continue
        test_folders.append(temp_folder)

    result_dict = {}
    for curt_folder in tqdm(test_folders):
        print('Folder : {0}'.format(curt_folder))
        currentDT = datetime.datetime.now()
        result_dir = 'temp_dect_results_{0}_{1}'.format(currentDT.strftime("%Y_%m_%d_%H_%M_%S"), currentDT.microsecond)
        if os.path.exists(result_dir):
            raise
        os.mkdir(result_dir)
        os.mkdir(os.path.join(result_dir, 'gt'))
        os.mkdir(os.path.join(result_dir, 'pd'))
        for adv_name in tqdm(os.listdir(os.path.join(args.dataset_dir, curt_folder))):
            temp_image_name_noext = os.path.splitext(adv_name)[0]
            if args.dataset_type == 'voc':
                gt_path = os.path.join(gt_dir, temp_image_name_noext + '.xml')

            if curt_folder == 'ori':
                adv_img_path = os.path.join(args.dataset_dir, curt_folder, temp_image_name_noext + '.jpg')
            else:
                adv_img_path = os.path.join(args.dataset_dir, curt_folder, temp_image_name_noext + '.png')

            if not os.path.exists(adv_img_path):
                print('File {0} not found.'.format(adv_name))
                continue
            
            if args.dataset_type == 'voc':
                gt_out = load_voc_annotations(gt_path, img_size)
                gt_out['classes'] = gt_out['classes'].astype(np.int)
            elif args.dataset_type == 'coco':
                gt_out = load_coco_annotations(temp_image_name_noext, img_size, gt_loader)
            
            if args.test_model == 'keypointrcnn':
                gt_out = only_person(gt_out, args)
            
            image_adv_np = load_image(data_format='channels_first', shape=img_size, bounds=(0, 1), abs_path=True, fpath=adv_img_path)
            Image.fromarray(np.transpose(image_adv_np * 255., (1, 2, 0)).astype(np.uint8)).save(os.path.join(result_dir, 'temp_adv.png'))
            image_adv_var = numpy_to_variable(image_adv_np)
            with torch.no_grad():
                pd_out = test_model(image_adv_var)
            pd_out = convert_torch_det_output(pd_out, cs_th=0.5)[0]

            if args.dataset_type == 'voc':
                pd_out = _transfer_label_to_voc(pd_out, args)

            bbox_list = pd_out['boxes']
            for idx, temp_bbox in enumerate(bbox_list):
                pd_out['boxes'][idx] = [temp_bbox[1], temp_bbox[0], temp_bbox[3], temp_bbox[2]]


            save_detection_to_file(gt_out, os.path.join(result_dir, 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
            save_detection_to_file(pd_out, os.path.join(result_dir, 'pd', temp_image_name_noext + '.txt'), 'detection')
            
            if pd_out:
                save_bbox_img(os.path.join(result_dir, 'temp_adv.png'), pd_out['boxes'], out_file=os.path.join(result_dir, 'temp_adv_box.png'))
            else:
                save_bbox_img(os.path.join(result_dir, 'temp_adv.png'), [], out_file=os.path.join(result_dir, 'temp_adv_box.png'))
            

        mAP_score = calculate_mAP_from_files(os.path.join(result_dir, 'gt'), os.path.join(result_dir, 'pd'))

        shutil.rmtree(result_dir)
        print(curt_folder, ' : ', mAP_score)
        result_dict[curt_folder] = 'mAP: {0:.04f}'.format(mAP_score)

        with open('temp_det_results_gt_{0}_{1}.json'.format(args.test_model, args.dataset_type), 'w') as fout:
            json.dump(result_dict, fout, indent=2)

def _transfer_label_to_voc(pd_out, args):
    voc_and_coco_classes = VOC_AND_COCO91_CLASSES

    ret = {
        'classes' : [],
        'scores' : [],
        'boxes' : []
    }
    for key in pd_out.keys():
        if key not in ret.keys():
            ret[key] = pd_out[key]
    classes_list = pd_out['classes']
    scores_list = pd_out['scores']
    boxes_list = pd_out['boxes']
    for idx, temp_class in enumerate(classes_list):
        for key, val in voc_and_coco_classes.items():
            if int(temp_class) == val:
                ret['classes'].append(int(key))
                ret['scores'].append(scores_list[idx])
                ret['boxes'].append(boxes_list[idx])
    return ret

def only_person(gt_out, args):
    if args.dataset_type == 'voc':
        target_label = 14
    elif args.dataset_type == 'coco':
        target_label = 1

    ret = {
        'classes' : [],
        'scores' : [],
        'boxes' : []
    }
    for key in gt_out.keys():
        if key not in ret.keys():
            ret[key] = gt_out[key]
    classes_list = gt_out['classes']
    scores_list = gt_out['scores']
    boxes_list = gt_out['boxes']
    for idx, temp_class in enumerate(classes_list):
        if temp_class == target_label:
            ret['classes'].append(temp_class)
            ret['scores'].append(scores_list[idx])
            ret['boxes'].append(boxes_list[idx])
    return ret


if __name__ == '__main__':
    main()
