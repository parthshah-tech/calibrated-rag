import numpy as np

from eval.stats import bootstrap_ci, paired_bootstrap, percentile


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    x = np.random.default_rng(1).normal(0.5, 0.1, 200)
    a = bootstrap_ci(x)
    assert a == bootstrap_ci(x)
    assert a[1] < a[0] < a[2]


def test_interval_narrows_with_more_data():
    rng = np.random.default_rng(2)
    small = bootstrap_ci(rng.normal(0, 1, 30))
    large = bootstrap_ci(rng.normal(0, 1, 3000))
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_paired_bootstrap_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(3)
    base = rng.uniform(0, 1, 300)
    better = base + 0.1 + rng.normal(0, 0.02, 300)
    assert paired_bootstrap(better, base)["significant"]
    same = paired_bootstrap(base, base)
    assert same["mean_diff"] == 0 and not same["significant"]


def test_empty_inputs():
    assert np.isnan(bootstrap_ci([])[0])
    assert np.isnan(percentile([], 50))
    assert percentile([1, 2, 3, 4], 50) == 2.5
