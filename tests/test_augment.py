import numpy as np
import soundfile as sf

from sa_asr.audio import SAMPLE_RATE, add_real_noise, augment, noise_files, reverberate, speed_perturb

SR = SAMPLE_RATE


def tone(seconds=3.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def test_speed_changes_length_by_the_factor_and_never_passes_the_limit():
    rng = np.random.default_rng(0)
    w = tone()
    out = speed_perturb(w, rng, factors=(1.1,))
    assert abs(len(out) - len(w) / 1.1) <= 2
    long = tone(29.5)
    assert len(speed_perturb(long, rng, factors=(0.9,))) == len(long)  # would pass 30 s: left alone


def test_reverb_keeps_length_and_level_but_changes_the_signal():
    w = tone()
    out = reverberate(w, np.random.default_rng(1))
    assert out.shape == w.shape and not np.allclose(out, w)
    assert abs(np.abs(out).max() - np.abs(w).max()) < 1e-4


def test_real_noise_is_mixed_at_the_requested_snr(tmp_path):
    sf.write(tmp_path / "n.wav", np.random.RandomState(0).randn(SR).astype(np.float32) * 0.1, SR)
    files = noise_files(tmp_path)
    w = tone()
    out = add_real_noise(w, np.random.default_rng(2), files, snr_db=(10.0, 10.0))
    snr = 10 * np.log10(np.mean(w**2) / np.mean((out - w) ** 2))
    assert abs(snr - 10) < 0.5


def test_packet_loss_silences_1_to_5_percent_in_whole_frames():
    from sa_asr.audio import packet_loss

    w = tone(10.0) + 0.5  # never zero on its own, so every zero sample is a dropped one
    for seed in range(5):
        out = packet_loss(w, np.random.default_rng(seed))
        assert out.shape == w.shape and w.min() > 0  # the input is untouched
        lost = np.mean(out == 0)
        assert 0.005 < lost <= 0.08  # 1-5% of frames, plus a burst's overshoot and overlaps
        frame = SR * 20 // 1000
        zero = (out == 0).reshape(-1, frame)
        assert np.all(zero.all(axis=1) | ~zero.any(axis=1))  # whole 20 ms frames, never part of one


def test_augment_with_everything_off_returns_the_input():
    w = tone()
    assert augment(w, np.random.default_rng(0)) is w
