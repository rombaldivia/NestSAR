import numpy as np


def scores(y, probabilities):
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Every scoring subset must contain both classes")
    if probabilities.shape != (len(y), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid model probabilities")
    pred = probabilities.argmax(1)
    recall = [float(np.mean(pred[y == c] == c)) for c in (0, 1)]
    return dict(balanced_accuracy=float(np.mean(recall)), accuracy=float(np.mean(pred == y)),
                recalls=recall, nll=float(-np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean()),
                samples=len(y), class_counts=np.bincount(y, minlength=2).tolist())


def paired_group_bootstrap(y, first, second, groups, seed, samples=2000):
    """Paired cluster bootstrap within ONE repeat; returns second-first in pp.

    Entire subject/setup groups, including camera views, are resampled together.
    This is a conditional descriptive interval, not a multiple-testing corrected
    significance test or an interval over independent repeated holdouts.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    unique = np.unique(groups)
    counts, a, b = [], [], []
    for g in unique:
        m = groups == g
        counts.append([np.sum(m & (y == c)) for c in (0, 1)])
        a.append([np.sum(m & (y == c) & (first == y)) for c in (0, 1)])
        b.append([np.sum(m & (y == c) & (second == y)) for c in (0, 1)])
    counts, a, b = map(np.asarray, (counts, a, b))
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(unique), size=(samples, len(unique)))
    denom = counts[draws].sum(1)
    good = np.all(denom > 0, axis=1)
    delta = 100*((b[draws].sum(1)[good]-a[draws].sum(1)[good])/denom[good]).mean(1)
    observed = 100*np.mean((b.sum(0)-a.sum(0))/counts.sum(0))
    ci = np.quantile(delta, [.025, .975]).tolist() if len(delta) >= 100 and len(unique) >= 2 else None
    return dict(delta_pp=float(observed), ci95_pp=ci, groups=len(unique), valid_bootstrap_draws=len(delta))


def summarize(records, seeds):
    arms = ("t16_mlp", "sequence16_gru", "sequence64_gru")
    by_repeat = []
    for seed in seeds:
        subset = [r for r in records if r["seed"] == seed]
        if not subset:
            continue
        row = {"seed": seed, "pairs": len(subset)}
        for arm in arms:
            row[arm] = 100*float(np.mean([r["arms"][arm]["final"]["balanced_accuracy"] for r in subset]))
        row["sequence64_minus_t16_pp"] = row[arms[2]]-row[arms[0]]
        row["sequence64_minus_sequence16_pp"] = row[arms[2]]-row[arms[1]]
        by_repeat.append(row)
    summary = {"by_repeat": by_repeat,
               "metric": "Macro of selected binary-pair balanced accuracies; NOT NTU120 top-1 accuracy",
               "uncertainty": "SD across overlapping grouped repeats; not an independent-repeat confidence interval"}
    for key in (*arms, "sequence64_minus_t16_pp", "sequence64_minus_sequence16_pp"):
        vals = [r[key] for r in by_repeat]
        summary[key] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals)>1 else None)
    return summary
