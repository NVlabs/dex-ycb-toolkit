import os
import time
import numpy as np
import pycocotools.mask
import json
import copy
import itertools

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

from lib.factory import get_dataset

# TODO(ywchao): tune OKS following https://cocodataset.org/#keypoints-eval.
_KPT_OKS_SIGMAS = [0.05] * 21


class COCOEvaluator():

  def __init__(self, name):
    self._name = name

    self._dataset = get_dataset(self._name)

    self._class_names = self._dataset.ycb_classes + ['hand']

    self._anno_file = os.path.join(os.path.dirname(__file__), "..", "eval",
                                   "anno_coco_{}.json".format(self._name))
    if os.path.isfile(self._anno_file):
      print('Found COCO annnotation file.')
    else:
      print('Cannot find COCO annnotation file.')
      self._generate_anno_file()

  def _generate_anno_file(self):
    print('Generating COCO annotation file')
    s = time.time()

    images = []
    annotations = []
    cnt_ann = 0

    for i in range(len(self._dataset)):
      if (i + 1) in np.floor(np.linspace(0, len(self._dataset), 11))[1:]:
        print('{:3.0f}%  {:6d}/{:6d}'.format(100 * i / len(self._dataset), i,
                                             len(self._dataset)))

      sample = self._dataset[i]

      img = {
          'id': i,
          'width': self._dataset.w,
          'height': self._dataset.h,
      }
      images.append(img)

      label = np.load(sample['label_file'])

      for y in sample['ycb_ids'] + [255]:
        mask = label['seg'] == y
        if np.count_nonzero(mask) == 0:
          continue
        mask = np.asfortranarray(mask)
        rle = pycocotools.mask.encode(mask)
        segmentation = rle
        segmentation['counts'] = segmentation['counts'].decode('ascii')
        # https://github.com/cocodataset/cocoapi/issues/36
        area = pycocotools.mask.area(rle).item()
        bbox = pycocotools.mask.toBbox(rle).tolist()
        if y == 255:
          category_id = 22
          keypoints = label['joint_2d'].squeeze(0).tolist()
          keypoints = [[0.0, 0.0, 0] if x[0] == -1 and x[1] == -1 else x + [2]
                       for x in keypoints]
          keypoints = [y for x in keypoints for y in x]
          num_keypoints = 21
        else:
          category_id = y
          keypoints = [0] * 21 * 3
          num_keypoints = 0
        ann = {
            'id': cnt_ann + 1,
            'image_id': i,
            'category_id': category_id,
            'segmentation': segmentation,
            'area': area,
            'bbox': bbox,
            'iscrowd': 0,
            'keypoints': keypoints,
            'num_keypoints': num_keypoints,
        }
        annotations.append(ann)
        cnt_ann += 1

    categories = []

    for i, x in enumerate(self._class_names):
      if x == 'hand':
        supercategory = 'mano'
        keypoints = self._dataset.mano_joints
        skeleton = [[y + 1 for y in x] for x in self._dataset.mano_joint_connect
                   ]
      else:
        supercategory = 'ycb'
        keypoints = []
        skeleton = []
      cat = {
          'id': i + 1,
          'name': x,
          'supercategory': supercategory,
          'keypoints': keypoints,
          'skeleton': skeleton,
      }
      categories.append(cat)

    anno = {}
    anno['info'] = {}
    anno['images'] = images
    anno['annotations'] = annotations
    anno['categories'] = categories

    print('Saving to {}'.format(self._anno_file))

    os.makedirs(os.path.dirname(self._anno_file), exist_ok=True)

    with open(self._anno_file, 'w') as f:
      json.dump(anno, f)

    e = time.time()
    print('time: {:7.2f}'.format(e - s))

  # https://github.com/facebookresearch/detectron2/blob/492cf9c7bae22d7d528f7f58169fcd52a450a0ca/detectron2/evaluation/coco_evaluation.py#L252
  def _derive_coco_results(self, coco_eval, iou_type):
    metrics = {
        'bbox': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'segm': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'keypoints': ['AP', 'AP50', 'AP75', 'APm', 'APl'],
    }[iou_type]

    results = {
        metric: float(coco_eval.stats[idx] *
                      100 if coco_eval.stats[idx] >= 0 else "nan")
        for idx, metric in enumerate(metrics)
    }
    keys, values = tuple(zip(*results.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt='pipe',
        floatfmt='.3f',
        stralign='center',
        numalign='center',
    )
    print('Evaluation results for {}: \n'.format(iou_type) + table)
    if not np.isfinite(sum(results.values())):
      print('Some metrics cannot be computed and is shown as NaN.')

    precisions = coco_eval.eval["precision"]
    assert len(self._class_names) == precisions.shape[2]

    results_per_category = []
    for idx, name in enumerate(self._class_names):
      precision = precisions[:, :, idx, 0, -1]
      precision = precision[precision > -1]
      ap = np.mean(precision) if precision.size else float('nan')
      results_per_category.append(("{}".format(name), float(ap * 100)))

    n_cols = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(
        *[results_flatten[i::n_cols] for i in range(n_cols)])
    table = tabulate(
        results_2d,
        tablefmt='pipe',
        floatfmt='.3f',
        headers=['category', 'AP'] * (n_cols // 2),
        numalign='left',
    )
    print('Per-category {} AP: \n'.format(iou_type) + table)

    results.update({'AP-' + name: ap for name, ap in results_per_category})
    return results

  def evaluate(self, res_file, tasks=('bbox', 'segm', 'keypoints')):
    coco_gt = COCO(self._anno_file)
    coco_dt = coco_gt.loadRes(res_file)

    results = {}

    for task in tasks:
      # https://github.com/facebookresearch/detectron2/blob/492cf9c7bae22d7d528f7f58169fcd52a450a0ca/detectron2/evaluation/coco_evaluation.py#L506
      if task == 'segm':
        coco_dt_ = copy.deepcopy(coco_dt)
        for ann in coco_dt_.loadAnns(coco_dt_.getAnnIds()):
          ann.pop('bbox', None)
        coco_dt_ = coco_gt.loadRes(coco_dt_.dataset['annotations'])
      else:
        coco_dt_ = coco_dt

      coco_eval = COCOeval(coco_gt, coco_dt_, task)

      if task == 'keypoints':
        coco_eval.params.kpt_oks_sigmas = np.array(_KPT_OKS_SIGMAS)

      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()

      results[task] = self._derive_coco_results(coco_eval, task)

    return results