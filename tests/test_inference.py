import math

import pytest
import torch

from src.inference import (
    anomaly_confidence,
    infer_domain,
    load_model,
    reconstruction_error,
    score_audio_file,
    score_segments,
)


def test_load_model_from_synthetic_checkpoint_returns_eval_mode_model(synthetic_checkpoint_path, cfg, device):
    model = load_model(cfg, checkpoint_path=synthetic_checkpoint_path, device=device)

    assert not model.training  # load_model() must call .eval()


def test_reconstruction_error_is_finite_and_nonnegative(synthetic_model, device):
    segment_tensor = torch.randn(3, 224, 224)

    score = reconstruction_error(synthetic_model, segment_tensor, device)

    assert math.isfinite(score)
    assert score >= 0


@pytest.mark.parametrize(
    "name,expected_domain",
    [
        ("S1000R_2017_AndreRobert.m4a", "Garage"),
        ("S1000RR_2014_MaximilianHohmann.m4a", "Garage"),
        ("S1000R_2015_PhilipKehl_cold.m4a", "Garage"),
        ("random_youtube_clip.m4a", "YouTube"),
    ],
)
def test_infer_domain(name, expected_domain, cfg):
    assert infer_domain(name, cfg) == expected_domain


def test_score_segments_computes_schema_and_relative_score_correctly(cfg):
    baseline = cfg.anomaly.domain_baselines["YouTube"]
    raw_scores = [baseline * 0.5, baseline, baseline * 2.0]
    start_times = [0.0, 0.25, 0.5]

    result = score_segments(raw_scores, start_times, "YouTube", cfg)

    assert result.domain == "YouTube"
    assert result.baseline == baseline
    assert result.threshold == cfg.anomaly.rel_threshold
    assert len(result.segments) == 3
    assert [s.relative_score for s in result.segments] == pytest.approx([0.5, 1.0, 2.0], rel=1e-6)
    assert [s.start_time for s in result.segments] == start_times


def test_score_segments_flags_a_sustained_anomaly_above_threshold(cfg):
    # A single isolated spike can legitimately bleed into its neighbors once
    # smoothed (window_size), so use a sustained run - long enough that the
    # smoothing window can't dilute it - to test thresholding in isolation.
    baseline = cfg.anomaly.domain_baselines["YouTube"]
    raw_scores = [baseline * (cfg.anomaly.rel_threshold + 5)] * 6
    start_times = [i * 0.25 for i in range(6)]

    result = score_segments(raw_scores, start_times, "YouTube", cfg)

    assert result.is_anomalous is True
    assert all(s.is_anomalous for s in result.segments)
    assert all(s.confidence > 0.5 for s in result.segments)
    assert result.max_confidence > 0.5


def test_score_segments_all_healthy_scores_are_not_anomalous(cfg):
    baseline = cfg.anomaly.domain_baselines["Garage"]
    raw_scores = [baseline * 0.9, baseline * 1.0, baseline * 1.05]
    start_times = [0.0, 0.25, 0.5]

    result = score_segments(raw_scores, start_times, "Garage", cfg)

    assert result.is_anomalous is False
    assert all(not s.is_anomalous for s in result.segments)
    assert all(s.confidence < 0.5 for s in result.segments)
    assert result.max_confidence < 0.5


def test_anomaly_confidence_crosses_one_half_exactly_at_the_threshold():
    assert anomaly_confidence(smoothed_score=2.5, threshold=2.5, healthy_median_score=1.0) == pytest.approx(0.5)


def test_anomaly_confidence_reads_low_but_nonzero_at_the_healthy_median():
    confidence = anomaly_confidence(smoothed_score=1.0, threshold=2.5, healthy_median_score=1.0)

    assert confidence == pytest.approx(0.05, abs=1e-9)


def test_anomaly_confidence_is_bounded_and_monotonic_in_the_score():
    threshold, healthy_median_score = 2.5, 1.0
    scores = [-5.0, 0.0, 1.0, 2.0, 2.5, 3.0, 10.0, 1000.0]

    confidences = [anomaly_confidence(s, threshold, healthy_median_score) for s in scores]

    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert confidences == sorted(confidences)  # strictly monotonic increasing in smoothed_score


def test_anomaly_confidence_never_overflows_on_pathological_inputs():
    # threshold <= healthy_median_score would otherwise divide by a
    # non-positive spread - must not raise, and must still return in [0, 1].
    assert 0.0 <= anomaly_confidence(smoothed_score=1e6, threshold=1.0, healthy_median_score=1.0) <= 1.0
    assert 0.0 <= anomaly_confidence(smoothed_score=-1e6, threshold=1.0, healthy_median_score=1.0) <= 1.0


def test_score_audio_file_end_to_end_with_synthetic_checkpoint(sine_wave_audio_file, synthetic_model, cfg, device):
    result = score_audio_file(sine_wave_audio_file, synthetic_model, cfg, device=device)

    assert len(result.segments) > 0
    for segment in result.segments:
        assert math.isfinite(segment.raw_score)
        assert math.isfinite(segment.relative_score)
        assert math.isfinite(segment.smoothed_score)
        assert 0.0 <= segment.confidence <= 1.0
        assert isinstance(segment.is_anomalous, bool)


def test_score_audio_file_explicit_domain_skips_filename_inference(sine_wave_audio_file, synthetic_model, cfg, device):
    # "sine.wav" contains none of cfg.anomaly.garage_name_hints, so infer_domain()
    # would default to "YouTube" - passing domain="Garage" must override that.
    result = score_audio_file(sine_wave_audio_file, synthetic_model, cfg, device=device, domain="Garage")

    assert result.domain == "Garage"
    assert result.baseline == cfg.anomaly.domain_baselines["Garage"]
