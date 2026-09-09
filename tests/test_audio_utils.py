import numpy as np
import pytest

from src.audio_utils import chunk_audio, load_audio, trim_audio


def test_load_audio_returns_mono_signal_at_requested_rate(sine_wave_audio_file, cfg):
    signal, sr = load_audio(str(sine_wave_audio_file), sample_rate=cfg.audio.sample_rate)

    assert sr == cfg.audio.sample_rate
    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert len(signal) > 0


def test_load_audio_resamples_to_a_different_rate(sine_wave_audio_file, cfg):
    target_sr = cfg.audio.sample_rate // 2
    _signal, sr = load_audio(str(sine_wave_audio_file), sample_rate=target_sr)

    assert sr == target_sr


def test_trim_audio_returns_requested_slice():
    signal = np.arange(100)
    trimmed = trim_audio(signal, 10, 20)

    assert len(trimmed) == 10
    np.testing.assert_array_equal(trimmed, np.arange(10, 20))


def test_trim_audio_clamps_out_of_range_bounds():
    signal = np.arange(10)

    assert len(trim_audio(signal, -5, 3)) == 3  # negative start clamps to 0
    assert len(trim_audio(signal, 5, 1000)) == 5  # end beyond signal clamps to len(signal)
    assert len(trim_audio(signal, 8, 3)) == 0  # end before start clamps to an empty slice


def test_chunk_audio_produces_the_expected_number_of_segments(sine_wave_signal):
    signal, sr = sine_wave_signal
    segment_duration, step_duration = 1.0, 0.25

    chunks = chunk_audio(signal, sr, segment_duration, step_duration)

    segment_samples = int(segment_duration * sr)
    step_samples = int(step_duration * sr)
    expected = int((len(signal) - segment_samples) / step_samples) + 1
    assert len(chunks) == expected


def test_chunk_audio_segments_have_correct_length_and_start_times(sine_wave_signal):
    signal, sr = sine_wave_signal
    segment_duration, step_duration = 1.0, 0.25

    chunks = chunk_audio(signal, sr, segment_duration, step_duration)

    segment_samples = int(segment_duration * sr)
    for i, (start_time, segment) in enumerate(chunks):
        assert len(segment) == segment_samples
        assert start_time == pytest.approx(i * step_duration, abs=1e-6)


def test_chunk_audio_returns_empty_list_for_signal_shorter_than_one_segment():
    sr = 16000
    signal = np.zeros(sr // 2)  # 0.5s, shorter than a 1.0s segment

    assert chunk_audio(signal, sr, segment_duration=1.0, step_duration=0.25) == []
