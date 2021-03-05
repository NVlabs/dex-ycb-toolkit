import os

from dex_ycb_toolkit.bop_eval import BOPEvaluator

name = 's0_test'
bop_eval = BOPEvaluator(name)

res_file = os.path.join("eval", "example_results_bop_{}.csv".format(name))

bop_eval.evaluate(res_file, renderer_type='python')
