#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python script_generate_adversarial_imagenet5000.py --adv-method dr -tm inception_v3 --inc3-attacklayer 4 --step-size 4 --steps 100