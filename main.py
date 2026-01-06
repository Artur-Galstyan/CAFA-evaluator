from cafaeval.evaluation import cafa_eval


config = {
    "obo_file": "",
    "pred_dir": "",
    "gt_file": "",
    "ia": "",
    "no_orphans": False,
    "norm": "cafa",
    "prop": "max",
    "max_terms": None,
    "th_step": 0.01,
    "n_cpu": 1,
}


df, df_best = cafa_eval(**config)
