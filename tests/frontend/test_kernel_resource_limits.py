from types import SimpleNamespace

import numpy as np
import pytest

import metile
import metile.frontend.kernel as kernel_module
from metile.frontend.kernel import OutOfResources, _validate_pipeline_threadgroup
from metile.runtime.metal_device import MetalDevice


@metile.kernel
def _copy(source, destination, BLOCK: metile.constexpr):
    offsets = metile.arange(0, BLOCK)
    metile.store(destination + offsets, metile.load(source + offsets))


@pytest.mark.parametrize("threadgroup", [(256, 1, 1), (16, 8, 2)])
@pytest.mark.parametrize("limit", [128, 256, 512])
def test_pipeline_limit_checks_all_threadgroup_dimensions(monkeypatch, threadgroup, limit):
    pipeline = object()
    queries = []

    def pipeline_max_threads(compiled):
        queries.append(compiled)
        return limit

    device = SimpleNamespace(pipeline_max_threads=pipeline_max_threads)
    monkeypatch.setattr(MetalDevice, "get", lambda: device)
    function = SimpleNamespace(name="bounded", threadgroup_size=threadgroup)

    if limit < 256:
        with pytest.raises(OutOfResources, match=r"requires 256.*pipeline limit is 128"):
            _validate_pipeline_threadgroup(function, pipeline)
    else:
        _validate_pipeline_threadgroup(function, pipeline)

    assert queries == [pipeline]


@pytest.mark.parametrize("offline", [False, True])
def test_unsupported_pipeline_is_rejected_before_dispatch_or_caching(monkeypatch, offline):
    pipeline = object()
    queries = []

    def pipeline_max_threads(compiled):
        queries.append(compiled)
        return 128

    device = SimpleNamespace(
        has_metal_compiler=offline,
        compile_msl=lambda *_: pipeline,
        compile_msl_precompiled=lambda *_: (pipeline, True),
        pipeline_max_threads=pipeline_max_threads,
    )
    monkeypatch.setattr(MetalDevice, "get", lambda: device)
    monkeypatch.setattr(kernel_module, "_kernel_cache", {})
    source = metile.Buffer.__new__(metile.Buffer)
    source.dtype = np.dtype(np.float32)
    source._source_array = None
    launcher = _copy[(1,)]

    def unexpected_dispatch(*_):
        pytest.fail("unsupported pipeline reached dispatch")

    monkeypatch.setattr(launcher, "_dispatch", unexpected_dispatch)

    with pytest.raises(OutOfResources, match=r"requires 256.*pipeline limit is 128"):
        launcher(source, source, BLOCK=256)

    assert queries == [pipeline]
    assert not kernel_module._kernel_cache
