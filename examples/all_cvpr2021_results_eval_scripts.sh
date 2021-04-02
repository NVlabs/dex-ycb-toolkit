#!/bin/bash

# COCO
python examples/evaluate_coco.py \
  --name s0_test \
  --res_file results/cvpr2021_results/coco_maskrcnn_s0_test.json
python examples/evaluate_coco.py \
  --name s1_test \
  --res_file results/cvpr2021_results/coco_maskrcnn_s1_test.json
python examples/evaluate_coco.py \
  --name s2_test \
  --res_file results/cvpr2021_results/coco_maskrcnn_s2_test.json
python examples/evaluate_coco.py \
  --name s3_test \
  --res_file results/cvpr2021_results/coco_maskrcnn_s3_test.json

# BOP
python examples/evaluate_bop.py \
  --name s0_test \
  --res_file results/cvpr2021_results/bop_posecnn_s0_test.csv
python examples/evaluate_bop.py \
  --name s1_test \
  --res_file results/cvpr2021_results/bop_posecnn_s1_test.csv
python examples/evaluate_bop.py \
  --name s2_test \
  --res_file results/cvpr2021_results/bop_posecnn_s2_test.csv
python examples/evaluate_bop.py \
  --name s3_test \
  --res_file results/cvpr2021_results/bop_posecnn_s3_test.csv
python examples/evaluate_bop.py \
  --name s0_test \
  --res_file results/cvpr2021_results/bop_posecnn_s0_test_refined.csv
python examples/evaluate_bop.py \
  --name s1_test \
  --res_file results/cvpr2021_results/bop_posecnn_s1_test_refined.csv
python examples/evaluate_bop.py \
  --name s2_test \
  --res_file results/cvpr2021_results/bop_posecnn_s2_test_refined.csv
python examples/evaluate_bop.py \
  --name s3_test \
  --res_file results/cvpr2021_results/bop_posecnn_s3_test_refined.csv
python examples/evaluate_bop.py \
  --name s0_test \
  --res_file results/cvpr2021_results/bop_deepim_s0_test_COLOR.csv
python examples/evaluate_bop.py \
  --name s1_test \
  --res_file results/cvpr2021_results/bop_deepim_s1_test_COLOR.csv
python examples/evaluate_bop.py \
  --name s2_test \
  --res_file results/cvpr2021_results/bop_deepim_s2_test_COLOR.csv
python examples/evaluate_bop.py \
  --name s3_test \
  --res_file results/cvpr2021_results/bop_deepim_s3_test_COLOR.csv
python examples/evaluate_bop.py \
  --name s0_test \
  --res_file results/cvpr2021_results/bop_deepim_s0_test_RGBD.csv
python examples/evaluate_bop.py \
  --name s1_test \
  --res_file results/cvpr2021_results/bop_deepim_s1_test_RGBD.csv
python examples/evaluate_bop.py \
  --name s2_test \
  --res_file results/cvpr2021_results/bop_deepim_s2_test_RGBD.csv
python examples/evaluate_bop.py \
  --name s3_test \
  --res_file results/cvpr2021_results/bop_deepim_s3_test_RGBD.csv
python examples/evaluate_bop.py \
  --name s0_test \
  --res_file results/cvpr2021_results/bop_poserbpf_s0_test_rgb.csv
python examples/evaluate_bop.py \
  --name s1_test \
  --res_file results/cvpr2021_results/bop_poserbpf_s1_test_rgb.csv
python examples/evaluate_bop.py \
  --name s2_test \
  --res_file results/cvpr2021_results/bop_poserbpf_s2_test_rgb.csv
python examples/evaluate_bop.py \
  --name s3_test \
  --res_file results/cvpr2021_results/bop_poserbpf_s3_test_rgb.csv
python examples/evaluate_bop.py \
  --name s0_test \
  --res_file results/cvpr2021_results/bop_poserbpf_s0_test_rgbd.csv
python examples/evaluate_bop.py \
  --name s1_test \
  --res_file results/cvpr2021_results/bop_poserbpf_s1_test_rgbd.csv
python examples/evaluate_bop.py \
  --name s2_test \
  --res_file results/cvpr2021_results/bop_poserbpf_s2_test_rgbd.csv
python examples/evaluate_bop.py \
  --name s3_test \
  --res_file results/cvpr2021_results/bop_poserbpf_s3_test_rgbd.csv
python examples/evaluate_bop.py \
  --name s0_test \
  --res_file results/cvpr2021_results/bop_cosypose_s0_test.csv
python examples/evaluate_bop.py \
  --name s1_test \
  --res_file results/cvpr2021_results/bop_cosypose_s1_test.csv
python examples/evaluate_bop.py \
  --name s2_test \
  --res_file results/cvpr2021_results/bop_cosypose_s2_test.csv
python examples/evaluate_bop.py \
  --name s3_test \
  --res_file results/cvpr2021_results/bop_cosypose_s3_test.csv
python examples/evaluate_bop.py \
  --name s1_test \
  --res_file results/cvpr2021_results/bop_dope_s1_test.csv

# HPE
python examples/evaluate_hpe.py \
  --name s0_test \
  --res_file results/cvpr2021_results/hpe_spurr_hrnet_s0_test.txt
python examples/evaluate_hpe.py \
  --name s1_test \
  --res_file results/cvpr2021_results/hpe_spurr_hrnet_s1_test.txt
python examples/evaluate_hpe.py \
  --name s2_test \
  --res_file results/cvpr2021_results/hpe_spurr_hrnet_s2_test.txt
python examples/evaluate_hpe.py \
  --name s3_test \
  --res_file results/cvpr2021_results/hpe_spurr_hrnet_s3_test.txt
python examples/evaluate_hpe.py \
  --name s0_test \
  --res_file results/cvpr2021_results/hpe_spurr_resnet50_s0_test.txt
python examples/evaluate_hpe.py \
  --name s1_test \
  --res_file results/cvpr2021_results/hpe_spurr_resnet50_s1_test.txt
python examples/evaluate_hpe.py \
  --name s2_test \
  --res_file results/cvpr2021_results/hpe_spurr_resnet50_s2_test.txt
python examples/evaluate_hpe.py \
  --name s3_test \
  --res_file results/cvpr2021_results/hpe_spurr_resnet50_s3_test.txt
python examples/evaluate_hpe.py \
  --name s0_test \
  --res_file results/cvpr2021_results/hpe_a2j_s0_test.txt
python examples/evaluate_hpe.py \
  --name s1_test \
  --res_file results/cvpr2021_results/hpe_a2j_s1_test.txt
python examples/evaluate_hpe.py \
  --name s2_test \
  --res_file results/cvpr2021_results/hpe_a2j_s2_test.txt
python examples/evaluate_hpe.py \
  --name s3_test \
  --res_file results/cvpr2021_results/hpe_a2j_s3_test.txt

# Grasp
python examples/evaluate_grasp.py \
  --name s0_test \
  --bop_res_file results/cvpr2021_results/bop_posecnn_s0_test.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s0_test.json
python examples/evaluate_grasp.py \
  --name s1_test \
  --bop_res_file results/cvpr2021_results/bop_posecnn_s1_test.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s1_test.json
python examples/evaluate_grasp.py \
  --name s2_test \
  --bop_res_file results/cvpr2021_results/bop_posecnn_s2_test.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s2_test.json
python examples/evaluate_grasp.py \
  --name s3_test \
  --bop_res_file results/cvpr2021_results/bop_posecnn_s3_test.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s3_test.json
python examples/evaluate_grasp.py \
  --name s0_test \
  --bop_res_file results/cvpr2021_results/bop_posecnn_s0_test_refined.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s0_test.json
python examples/evaluate_grasp.py \
  --name s1_test \
  --bop_res_file results/cvpr2021_results/bop_posecnn_s1_test_refined.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s1_test.json
python examples/evaluate_grasp.py \
  --name s2_test \
  --bop_res_file results/cvpr2021_results/bop_posecnn_s2_test_refined.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s2_test.json
python examples/evaluate_grasp.py \
  --name s3_test \
  --bop_res_file results/cvpr2021_results/bop_posecnn_s3_test_refined.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s3_test.json
python examples/evaluate_grasp.py \
  --name s0_test \
  --bop_res_file results/cvpr2021_results/bop_deepim_s0_test_COLOR.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s0_test.json
python examples/evaluate_grasp.py \
  --name s1_test \
  --bop_res_file results/cvpr2021_results/bop_deepim_s1_test_COLOR.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s1_test.json
python examples/evaluate_grasp.py \
  --name s2_test \
  --bop_res_file results/cvpr2021_results/bop_deepim_s2_test_COLOR.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s2_test.json
python examples/evaluate_grasp.py \
  --name s3_test \
  --bop_res_file results/cvpr2021_results/bop_deepim_s3_test_COLOR.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s3_test.json
python examples/evaluate_grasp.py \
  --name s0_test \
  --bop_res_file results/cvpr2021_results/bop_deepim_s0_test_RGBD.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s0_test.json
python examples/evaluate_grasp.py \
  --name s1_test \
  --bop_res_file results/cvpr2021_results/bop_deepim_s1_test_RGBD.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s1_test.json
python examples/evaluate_grasp.py \
  --name s2_test \
  --bop_res_file results/cvpr2021_results/bop_deepim_s2_test_RGBD.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s2_test.json
python examples/evaluate_grasp.py \
  --name s3_test \
  --bop_res_file results/cvpr2021_results/bop_deepim_s3_test_RGBD.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s3_test.json
python examples/evaluate_grasp.py \
  --name s0_test \
  --bop_res_file results/cvpr2021_results/bop_poserbpf_s0_test_rgb.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s0_test.json
python examples/evaluate_grasp.py \
  --name s1_test \
  --bop_res_file results/cvpr2021_results/bop_poserbpf_s1_test_rgb.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s1_test.json
python examples/evaluate_grasp.py \
  --name s2_test \
  --bop_res_file results/cvpr2021_results/bop_poserbpf_s2_test_rgb.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s2_test.json
python examples/evaluate_grasp.py \
  --name s3_test \
  --bop_res_file results/cvpr2021_results/bop_poserbpf_s3_test_rgb.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s3_test.json
python examples/evaluate_grasp.py \
  --name s0_test \
  --bop_res_file results/cvpr2021_results/bop_poserbpf_s0_test_rgbd.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s0_test.json
python examples/evaluate_grasp.py \
  --name s1_test \
  --bop_res_file results/cvpr2021_results/bop_poserbpf_s1_test_rgbd.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s1_test.json
python examples/evaluate_grasp.py \
  --name s2_test \
  --bop_res_file results/cvpr2021_results/bop_poserbpf_s2_test_rgbd.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s2_test.json
python examples/evaluate_grasp.py \
  --name s3_test \
  --bop_res_file results/cvpr2021_results/bop_poserbpf_s3_test_rgbd.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s3_test.json
python examples/evaluate_grasp.py \
  --name s0_test \
  --bop_res_file results/cvpr2021_results/bop_cosypose_s0_test.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s0_test.json
python examples/evaluate_grasp.py \
  --name s1_test \
  --bop_res_file results/cvpr2021_results/bop_cosypose_s1_test.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s1_test.json
python examples/evaluate_grasp.py \
  --name s2_test \
  --bop_res_file results/cvpr2021_results/bop_cosypose_s2_test.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s2_test.json
python examples/evaluate_grasp.py \
  --name s3_test \
  --bop_res_file results/cvpr2021_results/bop_cosypose_s3_test.csv \
  --coco_res_file results/cvpr2021_results/coco_maskrcnn_s3_test.json
