import numpy as np

from sa_asr.metrics import bootstrap_wer, score, utterance_errors


def test_utterance_errors_add_up_to_corpus_wer():
    refs = ["dumela rra", "ke a leboga thata", "", "go siame"]
    hyps = ["dumela", "ke leboga thata thata", "extra words", ""]
    e, w = utterance_errors(refs, hyps)
    assert w.tolist() == [2, 4, 0, 2]
    assert e[2] == 0 and e[3] == 2
    assert abs(e.sum() / w.sum() - score(refs, hyps)["wer"]) < 1e-9


def test_bootstrap_point_estimate_is_mean_of_set_wers():
    words = np.full(50, 10.0)
    a1, b1 = np.full(50, 2.0), np.full(50, 1.0)  # set 1: 20% vs 10%
    a2, b2 = np.full(50, 4.0), np.full(50, 4.0)  # set 2: 40% vs 40%
    r = bootstrap_wer([(words, a1, b1), (words, a2, b2)], n_boot=200)
    assert np.allclose(r["wer"], [30.0, 25.0])
    assert np.isclose(r["diff"][0], -5.0)
    # constant errors: every resample gives the same WER, so the interval is a point
    assert np.isclose(r["diff_low"][0], -5.0) and np.isclose(r["diff_high"][0], -5.0)
    assert r["p_better"][0] == 1.0


def test_bootstrap_noise_gives_an_interval_containing_zero_for_equal_models():
    rng = np.random.default_rng(1)
    words = rng.integers(3, 20, 100).astype(float)
    a = rng.binomial(words.astype(int), 0.25).astype(float)
    b = rng.binomial(words.astype(int), 0.25).astype(float)
    r = bootstrap_wer([(words, a, b)], n_boot=1000)
    assert r["low"][0] < r["wer"][0] < r["high"][0]
    assert r["diff_low"][0] < 0 < r["diff_high"][0] or abs(r["diff"][0]) > 3
    # a real 10-point gap on 100 utterances is clearly outside the noise
    c = rng.binomial(words.astype(int), 0.15).astype(float)
    r2 = bootstrap_wer([(words, a, c)], n_boot=1000)
    assert r2["diff_high"][0] < 0


def test_bootstrap_single_model():
    words = np.full(10, 5.0)
    r = bootstrap_wer([(words, np.ones(10))], n_boot=50)
    assert np.isclose(r["wer"][0], 20.0) and r["diff"] == []
