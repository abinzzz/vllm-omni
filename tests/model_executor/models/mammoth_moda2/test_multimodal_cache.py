# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real processor cache tests; only model configuration/tokenizer files are needed.

Set MAMMOTH_MODA2_TEST_MODEL to a local Preview or Dev checkpoint directory.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.config.multimodal import ImageDummyOptions
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import batched_tensors_equal
from vllm.multimodal.processing import InputProcessingContext
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.config import OmniModelConfig
from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    MammothModa2ForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def context():
    model = os.environ.get("MAMMOTH_MODA2_TEST_MODEL")
    if not model:
        pytest.skip("Set MAMMOTH_MODA2_TEST_MODEL to a local checkpoint (weights are not needed)")
    assert Path(model, "config.json").is_file()
    config = OmniModelConfig(
        model=model,
        tokenizer=model,
        model_arch="MammothModa2ForConditionalGeneration",
        model_stage="ar",
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
    )
    config.multimodal_config.limit_per_prompt = {"image": ImageDummyOptions(count=2)}
    config.multimodal_config.mm_processor_cache_gb = 0.1
    return InputProcessingContext(config, tokenizer=cached_tokenizer_from_config(config))


@pytest.fixture
def processors(context):
    factories = MammothModa2ForConditionalGeneration._processor_factory
    cache = MultiModalProcessorOnlyCache(context.model_config)
    return factories.build_processor(context, cache=None), factories.build_processor(context, cache=cache), cache


def _image(seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (224, 224, 3), dtype=np.uint8))


def _process(processor, images, question="Describe the image.", kwargs=None, uuids=None, token_prompt=False):
    prompt = (
        "<|im_start|>user\n"
        + "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    if token_prompt:
        prompt = processor.info.get_tokenizer().encode(prompt)
    return processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs=kwargs or {},
        mm_uuid_items={"image": uuids} if uuids is not None else None,
    )


def _assert_equal(a, b):
    assert {k: v for k, v in a.items() if k not in ("prompt", "mm_kwargs")} == {
        k: v for k, v in b.items() if k not in ("prompt", "mm_kwargs")
    }
    assert batched_tensors_equal(a["mm_kwargs"].get_data(), b["mm_kwargs"].get_data())


@pytest.mark.parametrize("token_prompt", [False, True])
@pytest.mark.parametrize("explicit_uuid", [False, True])
def test_hit_miss_and_changed_text(processors, monkeypatch, token_prompt, explicit_uuid):
    baseline, cached, cache = processors
    image = _image(0)
    uuids = ["image-a"] if explicit_uuid else None
    image_processor_cls = type(cached.info.get_hf_processor().image_processor)
    original = image_processor_cls.preprocess
    calls = []

    def preprocess(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(image_processor_cls, "preprocess", preprocess)
    outputs = []
    for index, question in enumerate(["Describe the image.", "Describe the image.", "Which colors are visible?"]):
        expected = _process(baseline, [image], question, uuids=uuids, token_prompt=token_prompt)
        before = len(calls)
        actual = _process(cached, [image.copy()], question, uuids=uuids, token_prompt=token_prompt)
        _assert_equal(expected, actual)
        assert len(calls) - before == (1 if index == 0 else 0)
        outputs.append(actual)
    assert outputs[0]["mm_hashes"] == outputs[2]["mm_hashes"]
    assert outputs[0]["prompt_token_ids"] != outputs[2]["prompt_token_ids"]
    assert cache.make_stats().hits >= 2


@pytest.mark.parametrize("explicit_uuid", [False, True])
def test_changed_processing_options_invalidate_cache(processors, explicit_uuid):
    baseline, cached, _ = processors
    image = _image(0)
    uuids = ["image-a"] if explicit_uuid else None
    original = _process(cached, [image], uuids=uuids)
    kwargs = {"size": {"shortest_edge": 56 * 56, "longest_edge": 112 * 112}}
    expected = _process(baseline, [image], kwargs=kwargs, uuids=uuids)
    changed = _process(cached, [image], kwargs=kwargs, uuids=uuids)
    _assert_equal(expected, changed)
    _assert_equal(expected, _process(cached, [image], kwargs=kwargs, uuids=uuids))
    assert original["mm_hashes"] != changed["mm_hashes"]
    assert not torch.equal(
        original["mm_kwargs"].get_data()["image_grid_thw"], changed["mm_kwargs"].get_data()["image_grid_thw"]
    )


def test_different_images_partial_hits_and_reordering(processors):
    baseline, cached, _ = processors
    a, b = _image(0), _image(1)
    first = _process(cached, [a])
    second = _process(cached, [b])
    assert first["mm_hashes"] != second["mm_hashes"]
    assert not batched_tensors_equal(first["mm_kwargs"].get_data(), second["mm_kwargs"].get_data())
    c = _image(2)
    for images in ([a, c], [c, a], [a, a], [b, c]):
        _assert_equal(_process(baseline, images), _process(cached, images))


def test_text_only_preserves_output_without_image_cache_lookups(processors):
    baseline, cached, cache = processors
    _process(cached, [_image(0)])
    before = cache.make_stats()
    _assert_equal(_process(baseline, []), _process(cached, []))
    assert cache.make_stats() == before


def test_clear_and_capacity_eviction(processors, context):
    baseline, cached, cache = processors
    a, b = _image(0), _image(1)
    expected = _process(baseline, [a])
    _process(cached, [a])
    key = expected["mm_hashes"]["image"][0]
    assert cache.is_cached_item(key)
    item_bytes = cache._cache.currsize
    cache.clear_cache()
    assert not cache.is_cached_item(key)
    _assert_equal(expected, _process(cached, [a]))

    # Keep exactly one same-sized item; the second distinct image must evict A.
    old_capacity = context.model_config.multimodal_config.mm_processor_cache_gb
    context.model_config.multimodal_config.mm_processor_cache_gb = (item_bytes + 1) / 2**30
    try:
        small_cache = MultiModalProcessorOnlyCache(context.model_config)
    finally:
        context.model_config.multimodal_config.mm_processor_cache_gb = old_capacity
    small = MammothModa2ForConditionalGeneration._processor_factory.build_processor(context, cache=small_cache)
    _process(small, [a])
    assert small_cache.is_cached_item(key)
    _assert_equal(_process(baseline, [b]), _process(small, [b]))
    assert not small_cache.is_cached_item(key)
    _assert_equal(expected, _process(small, [a]))
