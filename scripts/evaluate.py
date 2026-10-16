"""Full evaluation: baseline, accuracy/calibration, robustness, novelty detection, triage, clustering, drift.

Writes reports/results.json and reports/figures/*.png, plus artifacts/triage_bank.npz used by the service.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (adjusted_rand_score, f1_score, normalized_mutual_info_score,  # noqa: E402
                             roc_auc_score, silhouette_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoint import data as D  # noqa: E402
from geoint import degrade, drift, novelty, triage  # noqa: E402
from geoint.calibrate import ece, probs, risk_coverage  # noqa: E402
from geoint.features import chip_features  # noqa: E402
from geoint.infer import load_net, run  # noqa: E402

REP = D.ROOT / "reports"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEVS = [1, 2, 3, 4, 5]


def fpr_at_95tpr(s_known, s_novel):
    thr = np.quantile(s_novel, 0.05)  # 95% of novel chips score above thr
    return float((s_known >= thr).mean())


def main():
    t_start = time.time()
    d = D.load()
    yk = {s: D.to_known_index(d[f"y_{s}"]) for s in ("train", "validation", "test")}
    km = {s: yk[s] >= 0 for s in yk}
    x_tr, y_tr = d["x_train"][km["train"]], yk["train"][km["train"]]
    x_va, y_va = d["x_validation"][km["validation"]], yk["validation"][km["validation"]]
    x_te, y_te_all = d["x_test"], yk["test"]
    x_tk, y_tk = x_te[km["test"]], y_te_all[km["test"]]
    x_tn = x_te[~km["test"]]
    res = {"classes": {"known": D.KNOWN_CLASSES, "novel": D.NOVEL_CLASSES},
           "sizes": {"train_known": int(len(x_tr)), "val_known": int(len(x_va)),
                     "test_known": int(len(x_tk)), "test_novel": int(len(x_tn))}}

    # degraded copies of the known test set, shared by baseline and CNNs
    degraded = {(k, s): degrade.apply(x_tk, k, s, seed=s) for k in degrade.DEGRADATIONS for s in SEVS}

    # ---------- 1. scikit-learn baseline ----------
    print("baseline...", flush=True)
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, random_state=0)
    hgb.fit(chip_features(x_tr), y_tr)
    base = {"test_acc": float((hgb.predict(chip_features(x_tk)) == y_tk).mean())}
    base["test_macro_f1"] = float(f1_score(y_tk, hgb.predict(chip_features(x_tk)), average="macro"))
    base["degraded_sev3_acc"] = {k: float((hgb.predict(chip_features(degraded[(k, 3)])) == y_tk).mean())
                                 for k in degrade.DEGRADATIONS}
    res["baseline_hgb"] = base
    print(json.dumps(base), flush=True)

    bank_out = {}
    for variant in ("clean", "robust"):
        print(f"== {variant}", flush=True)
        net = load_net(variant)
        L_tr, Z_tr = run(net, x_tr)
        L_va, Z_va = run(net, x_va)
        L_tk, Z_tk = run(net, x_tk)
        L_tn, Z_tn = run(net, x_tn)
        R = {}

        # ---------- 2. accuracy + calibration ----------
        maha = novelty.Mahalanobis().fit(Z_tr, y_tr)
        pol = triage.fit_policy(L_va, Z_va, y_va, maha)
        p_raw, p_cal = probs(L_tk), probs(L_tk, pol.temperature)
        R["test_acc"] = float((L_tk.argmax(1) == y_tk).mean())
        R["test_macro_f1"] = float(f1_score(y_tk, L_tk.argmax(1), average="macro"))
        R["ece_raw"], R["ece_temp_scaled"] = ece(p_raw, y_tk), ece(p_cal, y_tk)
        R["policy"] = pol.to_dict()

        # ---------- 3. robustness ----------
        rob, deg_cache = {}, {}
        for (k, s), xd in degraded.items():
            Ld, Zd = run(net, xd)
            deg_cache[(k, s)] = (Ld, Zd)
            rob.setdefault(k, {})[s] = float((Ld.argmax(1) == y_tk).mean())
        R["robustness_acc"] = rob
        R["mean_degraded_acc"] = float(np.mean([v for k in rob for v in rob[k].values()]))

        # ---------- 4. novelty detection (known test vs held-out classes) ----------
        knn = novelty.KNN().fit(Z_tr, k=10)
        scores = {
            "msp": (novelty.msp(L_tk), novelty.msp(L_tn)),
            "energy": (novelty.energy(L_tk), novelty.energy(L_tn)),
            "mahalanobis": (maha.score(Z_tk), maha.score(Z_tn)),
            "knn": (knn.score(Z_tk), knn.score(Z_tn)),
        }
        rmaha = novelty.RelativeMahalanobis().fit(Z_tr, y_tr)
        scores["relative_mahalanobis"] = (rmaha.score(Z_tk), rmaha.score(Z_tn))
        R["novelty"] = {}
        for name, (sk, sn) in scores.items():
            yb = np.r_[np.zeros(len(sk)), np.ones(len(sn))]
            R["novelty"][name] = {"auroc": float(roc_auc_score(yb, np.r_[sk, sn])), "fpr_at_95tpr": fpr_at_95tpr(sk, sn)}
        # per novel class
        y_tn_all = d["y_test"][~km["test"]]
        R["novelty_by_class_mahalanobis"] = {}
        for c in D.NOVEL_CLASSES:
            m = y_tn_all == D.ALL_CLASSES.index(c)
            sk, sn = scores["mahalanobis"][0], scores["mahalanobis"][1][m]
            R["novelty_by_class_mahalanobis"][c] = float(roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(sn))], np.r_[sk, sn]))

        # ---------- 5. triage on the mixed test feed ----------
        L_mix, Z_mix = np.r_[L_tk, L_tn], np.r_[Z_tk, Z_tn]
        y_mix = np.r_[y_tk, -np.ones(len(L_tn), dtype=int)]
        r_mix = triage.route(L_mix, Z_mix, pol, maha)
        R["triage_clean_feed"] = triage.summarize(r_mix, y_mix)
        # no-abstention reference: auto-label everything with argmax
        # operating-point trade-off: looser novelty threshold catches more new content but sends more known chips to analysts
        R["triage_tradeoff"] = {}
        for fpr in (0.02, 0.05, 0.10, 0.20):
            pol_f = triage.fit_policy(L_va, Z_va, y_va, maha, novelty_fpr=fpr)
            R["triage_tradeoff"][str(fpr)] = triage.summarize(triage.route(L_mix, Z_mix, pol_f, maha), y_mix)
        R["triage_no_abstention_error"] = float(((L_mix.argmax(1) != y_mix) | (y_mix < 0)).mean())
        R["triage_degraded_feed"] = {}
        for k in degrade.DEGRADATIONS:
            Ld, Zd = deg_cache[(k, 3)]
            xn_d = degrade.apply(x_tn, k, 3, seed=3)
            Lnd, Znd = run(net, xn_d)
            rd = triage.route(np.r_[Ld, Lnd], np.r_[Zd, Znd], pol, maha)
            R["triage_degraded_feed"][k] = triage.summarize(rd, y_mix)

        # ---------- 6. unsupervised discovery among chips flagged novel ----------
        flagged = r_mix["reason"] == "novel"
        true_all = np.r_[d["y_test"][km["test"]], y_tn_all][flagged]
        Zf = PCA(32, random_state=0).fit_transform(Z_mix[flagged])
        sil = {kk: float(silhouette_score(Zf, KMeans(kk, n_init=10, random_state=0).fit_predict(Zf))) for kk in range(2, 7)}
        k_best = max(sil, key=sil.get)
        lab = KMeans(k_best, n_init=10, random_state=0).fit_predict(Zf)
        clusters = []
        for c in range(k_best):
            m = lab == c
            vals, cnt = np.unique(true_all[m], return_counts=True)
            clusters.append({"size": int(m.sum()), "majority": D.ALL_CLASSES[vals[cnt.argmax()]],
                             "purity": float(cnt.max() / m.sum())})
        R["novel_clustering"] = {"n_flagged": int(flagged.sum()),
                                 "frac_flagged_truly_novel": float(np.isin(true_all, [D.ALL_CLASSES.index(c) for c in D.NOVEL_CLASSES]).mean()),
                                 "silhouette_by_k": sil, "k": k_best,
                                 "nmi": float(normalized_mutual_info_score(true_all, lab)),
                                 "ari": float(adjusted_rand_score(true_all, lab)), "clusters": clusters}

        # ---------- 7. drift monitoring ----------
        rng = np.random.default_rng(0)
        pca = PCA(32, random_state=0).fit(Z_va)
        ref_idx = rng.choice(len(Z_va), 400, replace=False)
        ref = pca.transform(Z_va[ref_idx])
        gamma = drift.median_gamma(ref)
        ref_pred = L_va[ref_idx].argmax(1)
        scen = {
            "no_shift": lambda: Z_tk[rng.choice(len(Z_tk), 200, replace=False)],
            "haze_sev2": lambda: deg_cache[("haze", 2)][1][rng.choice(len(Z_tk), 200, replace=False)],
            "jpeg_sev3": lambda: deg_cache[("jpeg", 3)][1][rng.choice(len(Z_tk), 200, replace=False)],
            "novel_20pct": lambda: np.r_[Z_tk[rng.choice(len(Z_tk), 160, replace=False)], Z_tn[rng.choice(len(Z_tn), 40, replace=False)]],
            "prior_shift_water_forest": lambda: Z_tk[rng.choice(np.where(np.isin(y_tk, [D.KNOWN_CLASSES.index("SeaLake"), D.KNOWN_CLASSES.index("Forest")]))[0], 200)],
        }
        R["drift"] = {}
        for name, draw in scen.items():
            pv = [drift.mmd_test(ref, pca.transform(draw()), n_perm=100, seed=i, gamma=gamma)["p_value"] for i in range(20)]
            R["drift"][name] = {"alarm_rate_at_0.05": float(np.mean(np.asarray(pv) < 0.05)), "median_p": float(np.median(pv))}

        res[variant] = R
        print(json.dumps({k: R[k] for k in ("test_acc", "ece_raw", "ece_temp_scaled", "mean_degraded_acc", "novelty", "triage_clean_feed")}, indent=1), flush=True)

        # ---------- figures ----------
        cov_c, risk_c = risk_coverage(p_cal.max(1), p_cal.argmax(1) == y_tk)
        bank_out[variant] = {"cov": cov_c, "risk": risk_c, "rob": rob, "maha": scores["mahalanobis"]}
        if variant == "robust":
            # bank for the service / MCP server
            sub = rng.choice(len(Z_tr), 4000, replace=False)
            np.savez_compressed(D.ART / "triage_bank.npz", z=Z_tr[sub].astype(np.float32), y=y_tr[sub],
                                mu=maha.mu.astype(np.float32), prec=maha.prec.astype(np.float32),
                                pca_mean=pca.mean_, pca_comp=pca.components_, ref=ref, gamma=gamma,
                                ref_pred=ref_pred)
            (D.ART / "policy.json").write_text(json.dumps({**pol.to_dict(), "classes": D.KNOWN_CLASSES}, indent=2))

    # robustness figure
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
    for ax, k in zip(axes, degrade.DEGRADATIONS):
        for v, sty in (("clean", "--"), ("robust", "-")):
            ax.plot(SEVS, [bank_out[v]["rob"][k][s] for s in SEVS], sty, marker="o", label=v)
        ax.axhline(res["baseline_hgb"]["test_acc"], color="gray", lw=0.8, ls=":")
        ax.set_title(k); ax.set_xlabel("severity")
    axes[0].set_ylabel("accuracy (known classes)"); axes[0].legend()
    fig.tight_layout(); fig.savefig(FIG / "robustness.png", dpi=130); plt.close(fig)

    # risk-coverage figure
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for v in ("clean", "robust"):
        ax.plot(bank_out[v]["cov"], bank_out[v]["risk"] * 100, label=v)
    ax.set_xlabel("coverage (fraction auto-accepted)"); ax.set_ylabel("error among accepted (%)")
    ax.set_ylim(0, 6); ax.legend(); ax.set_title("Selective prediction, known classes")
    fig.tight_layout(); fig.savefig(FIG / "risk_coverage.png", dpi=130); plt.close(fig)

    # novelty score histogram
    fig, ax = plt.subplots(figsize=(5, 3.6))
    sk, sn = bank_out["robust"]["maha"]
    bins = np.linspace(0, np.quantile(np.r_[sk, sn], 0.995), 60)
    ax.hist(sk, bins, alpha=0.6, density=True, label="known test")
    ax.hist(sn, bins, alpha=0.6, density=True, label="Highway + Industrial (never trained)")
    ax.axvline(res["robust"]["policy"]["novelty_threshold"], color="k", ls="--", lw=1, label="threshold (fit on val)")
    ax.set_xlabel("Mahalanobis distance"); ax.legend(fontsize=7); ax.set_title("Novelty score, robust model")
    fig.tight_layout(); fig.savefig(FIG / "novelty_hist.png", dpi=130); plt.close(fig)

    res["runtime_sec"] = time.time() - t_start
    (REP / "results.json").write_text(json.dumps(res, indent=2))
    print(f"done in {res['runtime_sec']:.0f}s")


if __name__ == "__main__":
    main()
