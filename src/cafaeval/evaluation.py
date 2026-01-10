import logging
import os

import numpy as np
import polars as pl
from scipy import sparse

from cafaeval.graph import Prediction, propagate
from cafaeval.parser import gt_parser, obo_parser, pred_parser

logging.getLogger(__name__).addHandler(logging.NullHandler())


# Return a mask for all the predictions (matrix) >= tau
def solidify_prediction(pred, tau):
    return pred >= tau


# computes the f metric for each precision and recall in the input arrays
def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def compute_s(ru, mi):
    return np.sqrt(ru**2 + mi**2)
    # return np.where(np.isnan(ru), mi, np.sqrt(ru + np.nan_to_num(mi)))


def compute_confusion_matrix(tau_arr, g, pred, n_gt, ic_arr=None):
    metrics = np.zeros((len(tau_arr), 6), dtype="float")

    g_csr = sparse.csr_matrix(g)
    pred_csr = sparse.csr_matrix(pred)

    if ic_arr is not None:
        ic_diag = sparse.diags(ic_arr)

    for i, tau in enumerate(tau_arr):
        p = pred_csr >= tau

        intersection = p.multiply(g_csr).astype(bool)
        mis = p > g_csr
        remaining = g_csr > p

        if ic_arr is not None:
            p_w = p @ ic_diag
            intersection_w = intersection @ ic_diag
            mis_w = mis @ ic_diag
            remaining_w = remaining @ ic_diag

            n_pred = np.asarray(p_w.sum(axis=1)).ravel()
            n_intersection = np.asarray(intersection_w.sum(axis=1)).ravel()

            metrics[i, 0] = (np.asarray(p_w.sum(axis=1)).ravel() > 0).sum()
            metrics[i, 1] = n_intersection.sum()
            metrics[i, 2] = mis_w.sum()
            metrics[i, 3] = remaining_w.sum()
        else:
            n_pred = np.asarray(p.sum(axis=1)).ravel()
            n_intersection = np.asarray(intersection.sum(axis=1)).ravel()

            metrics[i, 0] = (n_pred > 0).sum()
            metrics[i, 1] = n_intersection.sum()
            metrics[i, 2] = mis.sum()
            metrics[i, 3] = remaining.sum()

        metrics[i, 4] = np.divide(
            n_intersection,
            n_pred,
            out=np.zeros_like(n_intersection, dtype="float"),
            where=n_pred > 0,
        ).sum()

        metrics[i, 5] = np.divide(
            n_intersection,
            n_gt,
            out=np.zeros_like(n_gt, dtype="float"),
            where=n_gt > 0,
        ).sum()

    return metrics


def compute_metrics(pred, gt, tau_arr, toi, ic_arr=None, n_cpu=0):
    columns = ["n", "tp", "fp", "fn", "pr", "rc"]

    if len(toi) == gt.matrix.shape[1]:
        g = gt.matrix
        pred_sub = pred.matrix
    else:
        g = gt.matrix[:, toi]
        pred_sub = pred.matrix[:, toi]

    w = None if ic_arr is None else ic_arr[toi]

    if w is None:
        n_gt = (
            np.asarray(g.sum(axis=1)).ravel() if sparse.issparse(g) else g.sum(axis=1)
        )
    else:
        if sparse.issparse(g):
            n_gt = np.asarray((g @ sparse.diags(w)).sum(axis=1)).ravel()
        else:
            n_gt = (g * w).sum(axis=1)

    metrics = compute_confusion_matrix(tau_arr, g, pred_sub, n_gt, w)

    return pl.DataFrame(dict(zip(columns, metrics.T)))


def normalize(metrics, ns, tau_arr, ne, normalization):
    # Normalize columns
    new_cols = {}
    for column in metrics.columns:
        if column != "n":
            # By default normalize by gt
            denominator = ne
            # Otherwise normalize by pred
            if normalization == "pred" or (normalization == "cafa" and column == "pr"):
                denominator = metrics["n"].to_numpy()
            col_values = metrics[column].to_numpy()
            new_cols[column] = np.divide(
                col_values,
                denominator,
                out=np.zeros_like(col_values, dtype="float"),
                where=denominator > 0,
            )

    # Update normalized columns
    for column, values in new_cols.items():
        metrics = metrics.with_columns(pl.Series(column, values))

    # Add new columns
    n_values = metrics["n"].to_numpy()
    metrics = metrics.with_columns(
        [
            pl.Series("ns", [ns] * len(tau_arr)),
            pl.Series("tau", tau_arr),
            pl.Series("cov", n_values / ne),
            pl.col("fp").alias("mi"),
            pl.col("fn").alias("ru"),
        ]
    )

    pr_values = metrics["pr"].to_numpy()
    rc_values = metrics["rc"].to_numpy()
    ru_values = metrics["ru"].to_numpy()
    mi_values = metrics["mi"].to_numpy()
    tp_values = metrics["tp"].to_numpy()
    fp_values = metrics["fp"].to_numpy()
    fn_values = metrics["fn"].to_numpy()

    metrics = metrics.with_columns(
        [
            pl.Series("f", compute_f(pr_values, rc_values)),
            pl.Series("s", compute_s(ru_values, mi_values)),
        ]
    )

    # Micro-average, calculation is based on the average of the confusion matrices
    pr_micro = np.divide(
        tp_values,
        tp_values + fp_values,
        out=np.zeros_like(tp_values, dtype="float"),
        where=(tp_values + fp_values) > 0,
    )
    rc_micro = np.divide(
        tp_values,
        tp_values + fn_values,
        out=np.zeros_like(tp_values, dtype="float"),
        where=(tp_values + fn_values) > 0,
    )

    metrics = metrics.with_columns(
        [
            pl.Series("pr_micro", pr_micro),
            pl.Series("rc_micro", rc_micro),
            pl.Series("f_micro", compute_f(pr_micro, rc_micro)),
        ]
    )

    return metrics


def evaluate_prediction(
    prediction, gt, ontologies, tau_arr, normalization="cafa", n_cpu=0
):
    dfs = []
    dfs_w = []
    for ns in prediction:
        ne = np.full(len(tau_arr), gt[ns].matrix[:, ontologies[ns].toi].shape[0])
        dfs.append(
            normalize(
                compute_metrics(
                    prediction[ns], gt[ns], tau_arr, ontologies[ns].toi, None, n_cpu
                ),
                ns,
                tau_arr,
                ne,
                normalization,
            )
        )

        if ontologies[ns].ia is not None:
            ne = np.full(len(tau_arr), gt[ns].matrix[:, ontologies[ns].toi_ia].shape[0])
            dfs_w.append(
                normalize(
                    compute_metrics(
                        prediction[ns],
                        gt[ns],
                        tau_arr,
                        ontologies[ns].toi_ia,
                        ontologies[ns].ia,
                        n_cpu,
                    ),
                    ns,
                    tau_arr,
                    ne,
                    normalization,
                )
            )

    dfs = pl.concat(dfs)

    # Merge weighted and unweighted dataframes
    if dfs_w:
        dfs_w = pl.concat(dfs_w)
        dfs = dfs.join(dfs_w, on=["ns", "tau"], suffix="_w")

    return dfs


def cafa_eval(
    obo_file,
    pred_dir,
    gt_file,
    ia=None,
    no_orphans=False,
    norm="cafa",
    prop="max",
    max_terms=None,
    th_step=0.01,
    n_cpu=1,
):
    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(th_step, 1, th_step)

    # Parse the OBO file and creates a different graphs for each namespace
    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)

    # Parse ground truth file
    gt = gt_parser(gt_file, ontologies)

    # Set prediction files looking recursively in the prediction folder
    pred_folder = os.path.normpath(pred_dir) + "/"  # add the tailing "/"
    pred_files = []
    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    logging.debug("Prediction paths {}".format(pred_files))

    # Parse prediction files and perform evaluation
    dfs = []
    for file_name in pred_files:
        prediction = pred_parser(file_name, ontologies, gt, prop, max_terms)
        if not prediction:
            logging.warning("Prediction: {}, not evaluated".format(file_name))
        else:
            df_pred = evaluate_prediction(
                prediction, gt, ontologies, tau_arr, normalization=norm, n_cpu=n_cpu
            )
            df_pred = df_pred.with_columns(
                pl.lit(file_name.replace(pred_folder, "").replace("/", "_")).alias(
                    "filename"
                )
            )
            dfs.append(df_pred)
            logging.info("Prediction: {}, evaluated".format(file_name))

    # Concatenate all dataframes and save them
    df = None
    dfs_best = {}
    if dfs:
        df = pl.concat(dfs)

        # Remove rows with no coverage
        df = df.filter(pl.col("cov") > 0)

        # Calculate the best index for each namespace and each evaluation metric
        for metric, cols in [
            ("f", ["rc", "pr"]),
            ("f_w", ["rc_w", "pr_w"]),
            ("s", ["ru", "mi"]),
            ("f_micro", ["rc_micro", "pr_micro"]),
            ("f_micro_w", ["rc_micro_w", "pr_micro_w"]),
        ]:
            if metric in df.columns:
                # Find the best metric value per filename/ns group
                if metric in ["f", "f_w", "f_micro", "f_micro_w"]:
                    df_best = df.group_by(["filename", "ns"]).agg(
                        pl.all().sort_by(metric).last()
                    )
                else:
                    df_best = df.group_by(["filename", "ns"]).agg(
                        pl.all().sort_by(metric).first()
                    )

                # Calculate max coverage per filename/ns
                cov_col = "cov_w" if metric[-2:] == "_w" else "cov"
                cov_max = df.group_by(["filename", "ns"]).agg(
                    pl.col(cov_col).max().alias("cov_max")
                )
                df_best = df_best.join(cov_max, on=["filename", "ns"])

                dfs_best[metric] = df_best
    else:
        logging.info("No predictions evaluated")

    return df, dfs_best


def pred_df_parser(pred_df, ontologies, gts, prop_mode, max_terms=None):
    """
    Parse a prediction polars DataFrame and returns a dict of Prediction objects, one for each namespace.

    Args:
        pred_df: polars DataFrame with columns 'protein_id', 'term_id', 'score'
        ontologies: dict of ontology Graph objects
        gts: dict of GroundTruth objects
        prop_mode: propagation mode ('max', 'fill', etc.)
        max_terms: optional max number of terms per protein

    Returns:
        dict of Prediction objects keyed by namespace
    """
    ids = {}
    matrix = {}
    ns_dict = {}
    replaced = {}

    for ns in gts:
        matrix[ns] = np.zeros(gts[ns].matrix.shape, dtype="float")
        ids[ns] = {}
        for term in ontologies[ns].terms_dict:
            ns_dict[term] = ns
        for term in ontologies[ns].terms_dict_alt:
            ns_dict[term] = ns

    # Iterate over the DataFrame rows
    for row in pred_df.iter_rows(named=True):
        p_id = row["protein_id"]
        term_id = row["term_id"]
        prob = row["score"]

        ns = ns_dict.get(term_id)
        if ns in gts and p_id in gts[ns].ids:
            i = gts[ns].ids[p_id]
            term_ids = [term_id]
            if term_id in ontologies[ns].terms_dict_alt:
                term_ids = ontologies[ns].terms_dict_alt[term_id]
                replaced.setdefault(ns, 0)
                replaced[ns] += len(term_ids)
            for term_id in term_ids:
                if max_terms is None or np.count_nonzero(matrix[ns][i]) <= max_terms:
                    j = ontologies[ns].terms_dict.get(term_id)["index"]
                    ids[ns][p_id] = i
                    matrix[ns][i, j] = max(matrix[ns][i, j], float(prob))

    predictions = {}
    for ns in ids:
        if ids[ns]:
            logging.debug("pred matrix {} {} ".format(ns, matrix[ns]))
            propagate(matrix[ns], ontologies[ns], ontologies[ns].order, mode=prop_mode)
            logging.debug("pred matrix propagated {} {} ".format(ns, matrix[ns]))
            predictions[ns] = Prediction(ids[ns], matrix[ns], ns)
            logging.info(
                "Prediction from DataFrame: {}, proteins {}, annotations {}, replaced alt. ids {}".format(
                    ns, len(ids[ns]), np.count_nonzero(matrix[ns]), replaced.get(ns, 0)
                )
            )

    if not predictions:
        logging.warning("Empty prediction! Check format or overlap with ground truth")

    return predictions


def cafa_eval_df(
    obo_file,
    pred_df,
    gt_file,
    ia=None,
    no_orphans=False,
    norm="cafa",
    prop="max",
    max_terms=None,
    th_step=0.01,
    n_cpu=1,
    pred_name="prediction",
):
    """
    Evaluate predictions from a polars DataFrame instead of prediction files.

    Args:
        obo_file: path to the OBO ontology file
        pred_df: polars DataFrame with columns 'protein_id', 'term_id', 'score'
        gt_file: path to the ground truth file
        ia: path to information accretion file (optional)
        no_orphans: exclude orphan terms from evaluation
        norm: normalization mode ('cafa', 'pred', 'gt')
        prop: propagation mode ('max', 'fill')
        max_terms: max number of terms per protein (optional)
        th_step: threshold step size for tau values
        n_cpu: number of CPUs (unused, kept for API compatibility)
        pred_name: name to use for the prediction in output

    Returns:
        tuple of (df, dfs_best) where df is the full evaluation DataFrame
        and dfs_best is a dict of DataFrames with best metrics per filename/ns
    """
    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(th_step, 1, th_step)

    # Parse the OBO file and creates a different graphs for each namespace
    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)

    # Parse ground truth file
    gt = gt_parser(gt_file, ontologies)

    # Parse prediction DataFrame
    prediction = pred_df_parser(pred_df, ontologies, gt, prop, max_terms)

    df = None
    dfs_best = {}

    if not prediction:
        logging.warning("Prediction from DataFrame not evaluated")
        return df, dfs_best

    df_pred = evaluate_prediction(
        prediction, gt, ontologies, tau_arr, normalization=norm, n_cpu=n_cpu
    )
    df_pred = df_pred.with_columns(pl.lit(pred_name).alias("filename"))

    df = df_pred.filter(pl.col("cov") > 0)

    # Calculate the best index for each namespace and each evaluation metric
    for metric, cols in [
        ("f", ["rc", "pr"]),
        ("f_w", ["rc_w", "pr_w"]),
        ("s", ["ru", "mi"]),
        ("f_micro", ["rc_micro", "pr_micro"]),
        ("f_micro_w", ["rc_micro_w", "pr_micro_w"]),
    ]:
        if metric in df.columns:
            # Find the best metric value per filename/ns group
            if metric in ["f", "f_w", "f_micro", "f_micro_w"]:
                df_best = df.group_by(["filename", "ns"]).agg(
                    pl.all().sort_by(metric).last()
                )
            else:
                df_best = df.group_by(["filename", "ns"]).agg(
                    pl.all().sort_by(metric).first()
                )

            # Calculate max coverage per filename/ns
            cov_col = "cov_w" if metric[-2:] == "_w" else "cov"
            cov_max = df.group_by(["filename", "ns"]).agg(
                pl.col(cov_col).max().alias("cov_max")
            )
            df_best = df_best.join(cov_max, on=["filename", "ns"])

            dfs_best[metric] = df_best

    logging.info("Prediction from DataFrame: {}, evaluated".format(pred_name))

    return df, dfs_best


def cafa_score(
    obo_file,
    pred_df,
    gt_file,
    ia,
    no_orphans=False,
    norm="cafa",
    prop="max",
    max_terms=None,
    th_step=0.01,
):
    """
    Compute the CAFA evaluation score (single number) for predictions.

    This returns the average weighted F-max (f_w) across the three ontologies
    (BPO, MFO, CCO), which is the metric used by Kaggle for CAFA6.

    Args:
        obo_file: path to the OBO ontology file (go-basic.obo)
        pred_df: polars DataFrame with columns 'protein_id', 'term_id', 'score'
        gt_file: path to the ground truth file (train_terms.tsv format)
        ia: path to information accretion file (IA.tsv) - required for weighted score
        no_orphans: exclude orphan terms from evaluation
        norm: normalization mode ('cafa', 'pred', 'gt')
        prop: propagation mode ('max', 'fill')
        max_terms: max number of terms per protein (optional)
        th_step: threshold step size for tau values

    Returns:
        float: the CAFA score (average weighted F-max across ontologies)

    Example:
        >>> import polars as pl
        >>> from cafaeval.evaluation import cafa_score
        >>>
        >>> pred_df = pl.DataFrame({
        ...     "protein_id": ["P12345", "P12345", "P67890"],
        ...     "term_id": ["GO:0008150", "GO:0003674", "GO:0008150"],
        ...     "score": [0.9, 0.8, 0.7],
        ... })
        >>>
        >>> score = cafa_score(
        ...     obo_file="go-basic.obo",
        ...     pred_df=pred_df,
        ...     gt_file="train_terms.tsv",
        ...     ia="IA.tsv",
        ... )
        >>> print(f"CAFA Score: {score:.4f}")
    """
    df, dfs_best = cafa_eval_df(
        obo_file=obo_file,
        pred_df=pred_df,
        gt_file=gt_file,
        ia=ia,
        no_orphans=no_orphans,
        norm=norm,
        prop=prop,
        max_terms=max_terms,
        th_step=th_step,
    )

    if dfs_best is None or "f_w" not in dfs_best:
        logging.warning(
            "No weighted F-max scores available. Check if IA file is provided."
        )
        return 0.0

    df_best_fw = dfs_best["f_w"]

    # Get the f_w score for each namespace
    # Expected namespaces: biological_process, molecular_function, cellular_component
    ns_scores = df_best_fw.group_by("ns").agg(pl.col("f_w").max())

    # Calculate average across all namespaces present
    if ns_scores.height == 0:
        logging.warning("No namespace scores found.")
        return 0.0

    avg_score = ns_scores["f_w"].mean()

    return float(avg_score)


def gt_parser_df(gt_df, ontologies):
    """
    Parse ground truth from a polars DataFrame.

    Args:
        gt_df: polars DataFrame with columns 'protein_id' and 'term_id'
        ontologies: dict of ontology Graph objects

    Returns:
        dict of GroundTruth objects keyed by namespace
    """
    from cafaeval.graph import GroundTruth, propagate

    gt_dict = {}
    replaced = {}

    for row in gt_df.iter_rows(named=True):
        p_id = row["protein_id"]
        term_id = row["term_id"]

        for ns in ontologies:
            if term_id in ontologies[ns].terms_dict:
                gt_dict.setdefault(ns, {}).setdefault(p_id, []).append(term_id)
                break
            elif term_id in ontologies[ns].terms_dict_alt:
                for t_id in ontologies[ns].terms_dict_alt[term_id]:
                    gt_dict.setdefault(ns, {}).setdefault(p_id, []).append(t_id)
                    replaced.setdefault(ns, 0)
                    replaced[ns] += 1
                break

    gts = {}
    for ns in ontologies:
        if gt_dict.get(ns):
            matrix = np.zeros((len(gt_dict[ns]), ontologies[ns].idxs), dtype="bool")
            ids = {}
            for i, p_id in enumerate(gt_dict[ns]):
                ids[p_id] = i
                for term_id in gt_dict[ns][p_id]:
                    matrix[i, ontologies[ns].terms_dict[term_id]["index"]] = 1

            propagate(matrix, ontologies[ns], ontologies[ns].order, mode="max")
            gts[ns] = GroundTruth(ids, matrix, ns)
            logging.info(
                "Ground truth from DataFrame: {}, proteins {}, annotations {}, replaced alt. ids {}".format(
                    ns, len(ids), np.count_nonzero(matrix), replaced.get(ns, 0)
                )
            )

    return gts


def cafa_eval_dfs(
    obo_file,
    pred_df,
    gt_df,
    ia=None,
    no_orphans=False,
    norm="cafa",
    prop="max",
    max_terms=None,
    th_step=0.01,
    n_cpu=1,
    pred_name="prediction",
):
    """
    Evaluate predictions using polars DataFrames for both predictions and ground truth.

    This is useful when you have ground truth in a DataFrame format (like Kaggle's
    train_terms.tsv with columns EntryID, term, aspect).

    Args:
        obo_file: path to the OBO ontology file (go-basic.obo)
        pred_df: polars DataFrame with columns 'protein_id', 'term_id', 'score'
        gt_df: polars DataFrame with columns 'protein_id', 'term_id'
        ia: path to information accretion file (IA.tsv) - optional
        no_orphans: exclude orphan terms from evaluation
        norm: normalization mode ('cafa', 'pred', 'gt')
        prop: propagation mode ('max', 'fill')
        max_terms: max number of terms per protein (optional)
        th_step: threshold step size for tau values
        n_cpu: number of CPUs (unused, kept for API compatibility)
        pred_name: name to use for the prediction in output

    Returns:
        tuple of (df, dfs_best) where df is the full evaluation DataFrame
        and dfs_best is a dict of DataFrames with best metrics per filename/ns
    """
    tau_arr = np.arange(th_step, 1, th_step)

    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)
    gt = gt_parser_df(gt_df, ontologies)
    prediction = pred_df_parser(pred_df, ontologies, gt, prop, max_terms)

    df = None
    dfs_best = {}

    if not prediction:
        logging.warning("Prediction from DataFrame not evaluated")
        return df, dfs_best

    df_pred = evaluate_prediction(
        prediction, gt, ontologies, tau_arr, normalization=norm, n_cpu=n_cpu
    )
    df_pred = df_pred.with_columns(pl.lit(pred_name).alias("filename"))

    df = df_pred.filter(pl.col("cov") > 0)

    for metric, cols in [
        ("f", ["rc", "pr"]),
        ("f_w", ["rc_w", "pr_w"]),
        ("s", ["ru", "mi"]),
        ("f_micro", ["rc_micro", "pr_micro"]),
        ("f_micro_w", ["rc_micro_w", "pr_micro_w"]),
    ]:
        if metric in df.columns:
            if metric in ["f", "f_w", "f_micro", "f_micro_w"]:
                df_best = df.group_by(["filename", "ns"]).agg(
                    pl.all().sort_by(metric).last()
                )
            else:
                df_best = df.group_by(["filename", "ns"]).agg(
                    pl.all().sort_by(metric).first()
                )

            cov_col = "cov_w" if metric[-2:] == "_w" else "cov"
            cov_max = df.group_by(["filename", "ns"]).agg(
                pl.col(cov_col).max().alias("cov_max")
            )
            df_best = df_best.join(cov_max, on=["filename", "ns"])

            dfs_best[metric] = df_best

    logging.info("Prediction from DataFrame: {}, evaluated".format(pred_name))

    return df, dfs_best


def cafa_score_dfs(
    obo_file,
    pred_df,
    gt_df,
    ia,
    no_orphans=False,
    norm="cafa",
    prop="max",
    max_terms=None,
    th_step=0.01,
):
    """
    Compute the CAFA evaluation score using DataFrames for both predictions and ground truth.

    This is the recommended function for CAFA6/Kaggle evaluation when you have:
    - Predictions as a DataFrame
    - Ground truth as a DataFrame (e.g., train_terms.tsv from Kaggle)

    Args:
        obo_file: path to the OBO ontology file (go-basic.obo)
        pred_df: polars DataFrame with columns 'protein_id', 'term_id', 'score'
        gt_df: polars DataFrame with columns 'protein_id', 'term_id'
        ia: path to information accretion file (IA.tsv) - required for weighted score
        no_orphans: exclude orphan terms from evaluation
        norm: normalization mode ('cafa', 'pred', 'gt')
        prop: propagation mode ('max', 'fill')
        max_terms: max number of terms per protein (optional)
        th_step: threshold step size for tau values

    Returns:
        float: the CAFA score (average weighted F-max across ontologies)

    Example:
        >>> import polars as pl
        >>> from cafaeval.evaluation import cafa_score_dfs
        >>>
        >>> # Load Kaggle train_terms.tsv and rename columns
        >>> gt_df = pl.read_csv("train_terms.tsv", separator="\\t")
        >>> gt_df = gt_df.rename({"EntryID": "protein_id", "term": "term_id"})
        >>>
        >>> # Your predictions
        >>> pred_df = pl.DataFrame({
        ...     "protein_id": ["P12345", "P12345"],
        ...     "term_id": ["GO:0008150", "GO:0003674"],
        ...     "score": [0.9, 0.8],
        ... })
        >>>
        >>> score = cafa_score_dfs(
        ...     obo_file="go-basic.obo",
        ...     pred_df=pred_df,
        ...     gt_df=gt_df,
        ...     ia="IA.tsv",
        ... )
        >>> print(f"CAFA Score: {score:.4f}")
    """
    df, dfs_best = cafa_eval_dfs(
        obo_file=obo_file,
        pred_df=pred_df,
        gt_df=gt_df,
        ia=ia,
        no_orphans=no_orphans,
        norm=norm,
        prop=prop,
        max_terms=max_terms,
        th_step=th_step,
    )

    if dfs_best is None or "f_w" not in dfs_best:
        logging.warning(
            "No weighted F-max scores available. Check if IA file is provided."
        )
        return 0.0

    df_best_fw = dfs_best["f_w"]

    ns_scores = df_best_fw.group_by("ns").agg(pl.col("f_w").max())

    if ns_scores.height == 0:
        logging.warning("No namespace scores found.")
        return 0.0

    avg_score = ns_scores["f_w"].mean()

    return float(avg_score)


def write_results(df, dfs_best, out_dir="results", th_step=0.01):
    # Create output folder here in order to store the log file
    out_folder = os.path.normpath(out_dir) + "/"
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Set the number of decimals to write in the output files based on the threshold step size
    decimals = int(np.ceil(-np.log10(th_step))) + 1

    df.write_csv(
        "{}/evaluation_all.tsv".format(out_folder),
        separator="\t",
        float_precision=decimals,
    )

    for metric in dfs_best:
        dfs_best[metric].write_csv(
            "{}/evaluation_best_{}.tsv".format(out_folder, metric),
            separator="\t",
            float_precision=decimals,
        )
