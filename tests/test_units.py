import numpy as np
import pytest

from geoint import calibrate, data as D, degrade, drift, novelty
from geoint.features import chip_features


def test_known_novel_split():
    assert set(D.KNOWN_CLASSES).isdisjoint(D.NOVEL_CLASSES)
    assert len(D.KNOWN_CLASSES) + len(D.NOVEL_CLASSES) == 10
    y = np.arange(10)
    k = D.to_known_index(y)
    assert (k[D.known_mask(y)] == np.arange(8)).all()
    assert (k[~D.known_mask(y)] == -1).all()


@pytest.mark.parametrize("kind", list(degrade.DEGRADATIONS))
def test_degradations_keep_shape_and_get_worse(kind):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (4, 64, 64, 3), dtype=np.uint8)
    errs = []
    for s in (1, 5):
        y = degrade.apply(x, kind, s)
        assert y.shape == x.shape and y.dtype == np.uint8
        errs.append(np.abs(y.astype(float) - x).mean())
    assert errs[1] > errs[0]


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 5
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 1.5
    hot = logits * 4  # overconfident copy
    t = calibrate.fit_temperature(hot, y)
    assert t > 1.5  # overconfident logits need cooling
    assert calibrate.ece(calibrate.probs(hot, t), y) < calibrate.ece(calibrate.probs(hot), y)


def test_risk_threshold_respects_target():
    rng = np.random.default_rng(1)
    conf = rng.random(5000)
    correct = rng.random(5000) < conf  # higher confidence -> more often right
    thr = calibrate.threshold_for_risk(conf, correct, 0.1)
    acc = correct[conf >= thr]
    assert 1 - acc.mean() <= 0.1 + 1e-9


def test_mahalanobis_flags_shifted_cluster():
    rng = np.random.default_rng(0)
    z = np.r_[rng.normal(0, 1, (500, 8)), rng.normal(5, 1, (500, 8))]
    y = np.r_[np.zeros(500, int), np.ones(500, int)]
    m = novelty.Mahalanobis().fit(z, y)
    inlier = m.score(rng.normal(0, 1, (100, 8)))
    outlier = m.score(rng.normal([-6] * 8, 1, (100, 8)))
    assert outlier.min() > np.quantile(inlier, 0.95)


def test_mmd_detects_shift_not_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (200, 4))
    same = drift.mmd_test(ref, rng.normal(0, 1, (150, 4)), n_perm=100)
    shifted = drift.mmd_test(ref, rng.normal(0.7, 1, (150, 4)), n_perm=100)
    assert same["p_value"] > 0.05 and shifted["p_value"] < 0.02


def test_psi_zero_on_identical():
    p = np.array([0, 1, 2, 2, 1, 0])
    assert drift.psi(p, p, 3) == pytest.approx(0, abs=1e-9)


def test_features_finite():
    x = np.random.default_rng(0).integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    f = chip_features(x)
    assert f.shape[0] == 3 and np.isfinite(f).all()
