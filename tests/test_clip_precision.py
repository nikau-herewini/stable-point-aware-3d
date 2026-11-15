"""Run the real estimator methods with no torch or model downloads."""

import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "spar3d/models/image_estimator/clip_based_estimator.py"
)


class ImageTensor:
    def __init__(self, name="input", dtype="float32"):
        self.name = name
        self.dtype = dtype

    def flatten(self, *args):
        return self

    def permute(self, *args):
        return self

    def __getitem__(self, index):
        return ImageTensor("mask" if index[1].start == 3 else "rgb", self.dtype)

    def __mul__(self, other):
        return self

    def to(self, *, dtype):
        return ImageTensor(self.name, dtype)


@pytest.fixture(params=["cpu", "cuda"])
def precision_path(request, monkeypatch):
    device = request.param
    events = []
    features = Mock()
    visual = Mock(return_value=features)
    visual.conv1 = SimpleNamespace(weight=SimpleNamespace(dtype="float32"))
    visual.input_resolution = 336
    visual.output_dim = 768
    # CLIP keeps some parameters in float32 even with half-precision convolutions.
    parameter = SimpleNamespace(dtype="float32", requires_grad=True)

    def cast_float():
        events.append("float")
        visual.conv1.weight.dtype = "float32"

    model = SimpleNamespace(
        visual=visual,
        parameters=Mock(side_effect=lambda: iter([parameter])),
        float=Mock(side_effect=cast_float),
        eval=Mock(side_effect=lambda: events.append("eval")),
    )

    def load(*args, **kwargs):
        events.append("load")
        return model, None

    def convert_weights(loaded_model):
        assert loaded_model is model
        events.append("convert")
        visual.conv1.weight.dtype = "float16"

    alpha_clip = ModuleType("alpha_clip")
    alpha_clip.load = Mock(side_effect=load)
    alpha_clip.model = ModuleType("alpha_clip.model")
    alpha_clip.model.convert_weights = Mock(side_effect=convert_weights)
    monkeypatch.setitem(sys.modules, "alpha_clip", alpha_clip)
    monkeypatch.setitem(sys.modules, "alpha_clip.model", alpha_clip.model)
    monkeypatch.setenv("ALPHA_CLIP_PATH", "/fake/clip-cache")

    # Compile only the production methods. Do not import the heavyweight module.
    tree = ast.parse(SOURCE.read_text())
    estimator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ClipBasedHeadEstimator"
    )
    methods = [
        node for node in estimator.body
        if isinstance(node, ast.FunctionDef) and node.name in {"configure", "forward"}
    ]
    constants = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.startswith("OPENAI_DATASET_")
            for target in node.targets
        )
    ]
    namespace = {
        "os": os,
        "alpha_clip": alpha_clip,
        "get_device": lambda: device,
        "nn": SimpleNamespace(
            Linear=Mock(),
            Sequential=Mock(),
            ModuleList=list,
            ModuleDict=dict,
            functional=SimpleNamespace(
                interpolate=Mock(side_effect=lambda image, **kwargs: image)
            ),
        ),
        "Normalize": lambda *args, **kwargs: lambda image: image,
    }
    isolated = ast.Module(body=constants + methods, type_ignores=[])
    # Postpone torch/jaxtyping annotations without importing either dependency.
    import __future__

    exec(
        compile(isolated, str(SOURCE), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    instance = SimpleNamespace(
        cfg=SimpleNamespace(
            model="ViT-L/14@336px",
            heads=[SimpleNamespace(name="roughness", n_hidden_layers=1)],
            hidden_features=512,
            activation="relu",
        ),
        make_activation=Mock(),
    )
    return SimpleNamespace(
        device=device,
        dtype="float16" if device == "cuda" else "float32",
        instance=instance,
        model=model,
        parameter=parameter,
        alpha_clip=alpha_clip,
        methods=namespace,
        events=events,
        features=features,
    )


def test_configure_loads_on_cpu_and_selects_weight_precision(precision_path):
    case = precision_path
    case.methods["configure"](case.instance)

    case.alpha_clip.load.assert_called_once_with(
        "ViT-L/14@336px", device="cpu", download_root="/fake/clip-cache"
    )
    if case.device == "cuda":
        case.alpha_clip.model.convert_weights.assert_called_once_with(case.model)
        case.model.float.assert_not_called()
        assert case.events == ["load", "convert", "eval"]
    else:
        case.model.float.assert_called_once_with()
        case.alpha_clip.model.convert_weights.assert_not_called()
        assert case.events == ["load", "float", "eval"]
    assert case.model.visual.conv1.weight.dtype == case.dtype
    assert case.parameter.dtype == "float32"
    assert case.parameter.requires_grad is False
    case.model.eval.assert_called_once_with()
    assert case.instance.img_size == 336
    assert "roughness" in case.instance.heads


def test_forward_matches_image_and_mask_to_convolution_precision(precision_path):
    case = precision_path
    case.methods["configure"](case.instance)
    # Skip the output heads to isolate image/mask casting and CLIP execution.
    case.instance.cfg.heads = []
    case.model.parameters.reset_mock()

    assert case.methods["forward"](case.instance, ImageTensor()) == {}

    case.model.visual.assert_called_once()
    image, mask = case.model.visual.call_args.args
    assert image.name == "rgb"
    assert mask.name == "mask"
    assert image.dtype == mask.dtype == case.dtype
    # The first parameter is float32 on CUDA and must not determine input dtype.
    case.model.parameters.assert_not_called()
    case.features.float.assert_called_once_with()
