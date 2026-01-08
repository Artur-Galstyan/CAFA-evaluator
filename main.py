from cafaeval.evaluation import cafa_eval
import time


config = {
    "obo_file": "go-basic.obo",
    "pred_dir": "preds",
    "gt_file": "train_terms.tsv",
    "ia": "IA.tsv",
    "no_orphans": False,
    "norm": "cafa",
    "prop": "max",
    "max_terms": None,
    "th_step": 0.01,
    "n_cpu": 12,
}


start_time = time.time()
print(f"{start_time=}")
df, df_best = cafa_eval(**config)
end_time = time.time()
print(f"{end_time-start_time=}")
