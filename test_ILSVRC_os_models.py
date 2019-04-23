import torchvision.models as models
import torch
from PIL import Image
from image_utils import load_image, save_image
from torch_utils import numpy_to_variable, variable_to_numpy
from image_utils import numpy_to_bytes
from models.vgg import Vgg16
from models.resnet import Resnet152
from attacks.dispersion import DispersionAttack_opt, DispersionAttack
from attacks.mifgsm import MomentumIteratorAttack
from api_utils import detect_label_numpy
import numpy as np
import torchvision
from api_utils import detect_label_file
from tqdm import tqdm
import os
import pdb

# Resnet152 [4, 5, 6, 7]
# Vgg16 [2, 7, 14, 21, 28]

with open('labels.txt','r') as inf:
    imagenet_dict = eval(inf.read())

dataset_dir = "/home/yantao/datasets/ILSVRC/Data/DET/test"
write_image_dir='./out/DispersionAttack_opt_out'
images_name = os.listdir(dataset_dir)

model = Vgg16()
internal = [i for i in range(29)]
test_model = torchvision.models.inception_v3(pretrained='imagenet').cuda().eval()
#attack = DispersionAttack(model, epsilon=16./255, step_size=1./255, steps=2000, test_api=True)
attack = DispersionAttack_opt(model, epsilon=16./255, steps=2000, is_test_model=True)

total_samples = len(images_name)
success_attacks = 0
for idx, temp_image_name in enumerate(tqdm(images_name)):
    print('idx: ', idx)
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    image = numpy_to_variable(image_np)

    adv = image

    pred_nat = test_model(adv).detach().cpu().numpy()
    gt_label = np.argmax(pred_nat)

    pred_cls = imagenet_dict[gt_label]
    print(pred_cls)

    adv, info_dict = attack(image, 
                            attack_layer_idx=14, 
                            internal=internal, 
                            test_steps=500, 
                            gt_label=gt_label,
                            test_model=test_model)
    if bool(info_dict):
        output_label = info_dict['det_label']
        output_cls = imagenet_dict[output_label]
        print(output_cls)
        if(gt_label != output_label):
            success_attacks += 1
    else:
        print("Attack failed.")
    adv_np = variable_to_numpy(adv)
    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)
    print(" ")
    print('success_attacks: ', success_attacks)

    

print('attack success rate: ', float(success_attacks) / float(total_samples))


'''
model = torchvision.models.vgg16(pretrained=True).cuda()
attack = MomentumIteratorAttack(model, decay_factor=1.0, epsilon=16./255, steps=2000, step_size=1./255, random_start=False)

total_samples = 100
success_attacks = 0
for idx, temp_image_name in enumerate(images_name):
    print('idx: ', idx)

    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)

    Image.fromarray(np.transpose((image_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/ori.jpg')
    google_label = detect_label_file('./out/ori.jpg')
    if len(google_label) > 0:
        pred_cls = google_label[0].description
    else:
        pred_cls = None
    print(pred_cls)

    image_torch_nchw = torch.from_numpy(np.expand_dims(image_np, axis=0)).float()
    pred_nat = model(image_torch_nchw.cuda()).detach().cpu().numpy()
    label = np.argmax(pred_nat)
    label_tensor = torch.tensor(np.array([label]))
    adv = attack(image_torch_nchw, label_tensor)
    adv_np = variable_to_numpy(adv)
    Image.fromarray(np.transpose((adv_np * 255).astype(np.uint8), (1, 2, 0))).save('./out/temp.jpg')
    google_label = detect_label_file('./out/temp.jpg')
    if len(google_label) > 0:
        output_cls = google_label[0].description
    else:
        output_cls = None
    print(output_cls)
    print(" ")

    adv_np = variable_to_numpy(adv)
    linf = int(np.max(abs(image_np - adv_np)) * 255)
    print('linf: ', linf)
    l1 = np.mean(abs(image_np - adv_np)) * 255
    print('l1: ', l1)
    l2 = np.sqrt(np.mean(np.multiply((image_np * 255 - adv_np * 255), (image_np * 255 - adv_np * 255))))
    print('l2: ', l2)
    print(" ")

    if output_cls != pred_cls:
        success_attacks += 1

print('attack success rate: ', float(success_attacks) / float(total_samples))
'''