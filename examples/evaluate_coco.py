import os

from dex_ycb_toolkit.coco_eval import COCOEvaluator

name = 's0_test'
coco_eval = COCOEvaluator(name)

res_file = os.path.join("results", "example_results_coco_{}.json".format(name))

coco_eval.evaluate(res_file)
