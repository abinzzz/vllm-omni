# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks.mammoth_moda2_cache import CacheProbeWorkerExtension, _validate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def arms():
    feature = {"shape": [154, 16384], "dtype": "torch.bfloat16", "sha256": "correct"}
    record = {
        "case": "repeat_a_0",
        "image_sha256": "image-a",
        "prompt": "Describe the image.",
        "processor_kwargs": {},
        "token_ids": [1, 2, 3],
        "finish_reason": "length",
        "encoded_images": 1,
        "processor_hits": 0,
        "processor_lookups": 0,
        "worker": {
            "encoder_features": {"request-local-key": feature},
            "consumed_encoder_features": [{"key": "request-local-key", "feature": feature}],
        },
    }
    cached = deepcopy(record)
    cached.update(encoded_images=0, processor_hits=1, processor_lookups=1)
    cached["worker"] = {
        "encoder_features": {"content-key": deepcopy(feature)},
        "consumed_encoder_features": [{"key": "content-key", "feature": deepcopy(feature)}],
    }
    return [{"records": [record]}, {"records": [cached]}]


def _feature_check(arms):
    return next(check for check in _validate(arms) if check["check"] == "encoder_feature_equivalence")


def test_request_local_and_content_keys_can_refer_to_equal_features(arms):
    assert all(check["passed"] for check in _validate(arms))


def test_correct_resident_feature_does_not_mask_wrong_consumed_feature(arms):
    consumed = arms[1]["records"][0]["worker"]["consumed_encoder_features"][0]
    consumed["feature"] = {**consumed["feature"], "sha256": "wrong-deepstack-features"}
    assert not _feature_check(arms)["passed"]


def test_missing_cache_read_is_not_a_successful_hit(arms):
    arms[1]["records"][0]["worker"]["consumed_encoder_features"].clear()
    assert not _feature_check(arms)["passed"]


def test_consumption_order_must_match(arms):
    for arm in arms:
        consumed = arm["records"][0]["worker"]["consumed_encoder_features"]
        consumed.append({"key": "second", "feature": {**consumed[0]["feature"], "sha256": "second"}})
    arms[1]["records"][0]["worker"]["consumed_encoder_features"].reverse()
    assert not _feature_check(arms)["passed"]


def test_text_only_must_not_consume_resident_image_features(arms):
    for arm in arms:
        record = arm["records"][0]
        record.update(case="text_only", image_sha256=None, encoded_images=0, processor_hits=0, processor_lookups=0)
        record["worker"]["consumed_encoder_features"].clear()
    assert all(check["passed"] for check in _validate(arms))
    record = arms[1]["records"][0]
    record["worker"]["consumed_encoder_features"].append(
        {"key": "stale", "feature": next(iter(record["worker"]["encoder_features"].values()))}
    )
    assert not _feature_check(arms)["passed"]


@pytest.mark.parametrize("field,value", [("case", "first_a"), ("prompt", "Changed"), ("processor_kwargs", {"size": 1})])
def test_mismatched_requests_are_rejected(arms, field, value):
    arms[1]["records"][0][field] = value
    with pytest.raises(ValueError, match="same requests"):
        _validate(arms)


def test_latency_validation_needs_no_worker_probe(arms):
    for arm in arms:
        del arm["records"][0]["worker"]
    checks = _validate(arms, instrumented=False)
    assert {check["check"] for check in checks} == {"identical_generated_tokens", "identical_finish_reason"}
    assert all(check["passed"] for check in checks)


def test_probe_fingerprints_returned_tensor_and_clears_reads(monkeypatch):
    import torch

    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.accelerator, name, lambda: None)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda: 0)
    resident = torch.zeros(2, 4, dtype=torch.bfloat16)
    returned = torch.ones_like(resident)
    probe = CacheProbeWorkerExtension()
    probe.model_runner = SimpleNamespace(
        model=SimpleNamespace(embed_multimodal=lambda **kwargs: [resident]),
        encoder_cache={"image-a": resident},
        _cache_encoder_output=lambda *args, **kwargs: None,
        _get_encoder_output_from_cache=lambda key: returned,
    )
    probe.mammoth_cache_probe("start")
    assert probe.model_runner._get_encoder_output_from_cache("image-a") is returned
    snapshot = probe.mammoth_cache_probe()
    consumed = snapshot["consumed_encoder_features"]
    assert len(consumed) == 1
    assert consumed[0]["key"] == "image-a"
    assert consumed[0]["feature"]["sha256"] != snapshot["encoder_features"]["image-a"]["sha256"]
    assert probe.mammoth_cache_probe()["consumed_encoder_features"] == []
