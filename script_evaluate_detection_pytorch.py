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
import datetime

from utils.image_utils import load_image, save_image, save_bbox_img
from utils.mAP import save_detection_to_file, calculate_mAP_from_files
from utils.torch_utils import numpy_to_variable, variable_to_numpy, convert_torch_det_output


PICK_LIST = []
BAN_LIST = []

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Script for generating adversarial examples.')
    parser.add_argument('test_model', help='Model for testing AEs.', type=str)
    parser.add_argument('--dataset-dir', help='Dataset folder path.', default='/home/yantao/workspace/datasets/imagenet5000', type=str)

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args_dic = vars(args)

    with open('utils/labels.txt','r') as inf:
        args_dic['imagenet_dict'] = eval(inf.read())

    input_dir = os.path.join(args.dataset_dir, 'ori')

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
        if temp_folder == 'imagenet_val_5000' or temp_folder == 'ori' or temp_folder == '.git' or temp_folder == '_annotations' or temp_folder == '_segmentations':
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
        is_missing = False
        for image_name in tqdm(os.listdir(input_dir)):
            temp_image_name_noext = os.path.splitext(image_name)[0]
            ori_img_path = os.path.join(input_dir, image_name)
            adv_img_path = os.path.join(args.dataset_dir, curt_folder, image_name)
            adv_img_path = os.path.splitext(adv_img_path)[0] + '.png'
            if not os.path.exists(adv_img_path):
                if not is_missing:
                    is_missing = True
                    print('File {0} not found.'.format(image_name))
                    continue
            
            image_ori_np = load_image(data_format='channels_first', shape=img_size, bounds=(0, 1), abs_path=True, fpath=ori_img_path)
            Image.fromarray(np.transpose(image_ori_np * 255., (1, 2, 0)).astype(np.uint8)).save(os.path.join(result_dir, 'temp_ori.png'))
            image_ori_var = numpy_to_variable(image_ori_np)
            with torch.no_grad():
                gt_out = test_model(image_ori_var)
            gt_out = convert_torch_det_output(gt_out, cs_th=0.3)[0]

            image_adv_np = load_image(data_format='channels_first', shape=img_size, bounds=(0, 1), abs_path=True, fpath=adv_img_path)
            Image.fromarray(np.transpose(image_adv_np * 255., (1, 2, 0)).astype(np.uint8)).save(os.path.join(result_dir, 'temp_adv.png'))
            image_adv_var = numpy_to_variable(image_adv_np)
            with torch.no_grad():
                pd_out = test_model(image_adv_var)
            pd_out = convert_torch_det_output(pd_out, cs_th=0.3)[0]

            save_detection_to_file(gt_out, os.path.join(result_dir, 'gt', temp_image_name_noext + '.txt'), 'ground_truth')
            save_detection_to_file(pd_out, os.path.join(result_dir, 'pd', temp_image_name_noext + '.txt'), 'detection')
            
            if gt_out:
                save_bbox_img(os.path.join(result_dir, 'temp_ori.png'), gt_out['boxes'], out_file=os.path.join(result_dir, 'temp_ori_box.png'))
            else:
                save_bbox_img(os.path.join(result_dir, 'temp_ori.png'), [], out_file=os.path.join(result_dir, 'temp_ori_box.png'))
            if pd_out:
                save_bbox_img(os.path.join(result_dir, 'temp_adv.png'), pd_out['boxes'], out_file=os.path.join(result_dir, 'temp_adv_box.png'))
            else:
                save_bbox_img(os.path.join(result_dir, 'temp_adv.png'), [], out_file=os.path.join(result_dir, 'temp_adv_box.png'))
            

        mAP_score = calculate_mAP_from_files(os.path.join(result_dir, 'gt'), os.path.join(result_dir, 'pd'))
        shutil.rmtree(result_dir)
        print(curt_folder, ' : ', mAP_score)
        result_dict[curt_folder] = 'mAP: {0:.04f}'.format(mAP_score)

        with open('temp_det_results_{0}.json'.format(args.test_model), 'w') as fout:
            json.dump(result_dict, fout, indent=2)


if __name__ == '__main__':
    main()
