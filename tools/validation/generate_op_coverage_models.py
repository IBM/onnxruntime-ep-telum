#!/usr/bin/env python3
"""Generate deterministic ONNX models for Telum plugin op-coverage validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _save_model(path: Path, graph: onnx.GraphProto, opset: int) -> None:
    model = helper.make_model(
        graph,
        producer_name="telum-plugin-ep",
        opset_imports=[helper.make_operatorsetid("", opset)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(path))


def _f32(name: str, shape: Iterable[int], rng: np.random.Generator) -> onnx.TensorProto:
    arr = rng.standard_normal(tuple(shape), dtype=np.float32)
    return numpy_helper.from_array(arr, name=name)


def _i64(name: str, values: Iterable[int]) -> onnx.TensorProto:
    arr = np.asarray(list(values), dtype=np.int64)
    return numpy_helper.from_array(arr, name=name)


def _bool(name: str, shape: Iterable[int], rng: np.random.Generator) -> onnx.TensorProto:
    arr = (rng.integers(0, 2, size=tuple(shape), dtype=np.int32) != 0).astype(np.bool_)
    return numpy_helper.from_array(arr, name=name)


def build_backend_matmul_chain(out_dir: Path, opset: int, rng: np.random.Generator) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 64])
    inits = [
        _f32("W1", [64, 64], rng),
        _f32("W2", [64, 64], rng),
        _f32("B1", [64], rng),
        _f32("B2", [64], rng),
    ]
    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["MM1"]),
        helper.make_node("Add", ["MM1", "B1"], ["A1"]),
        helper.make_node("Relu", ["A1"], ["R1"]),
        helper.make_node("MatMul", ["R1", "W2"], ["MM2"]),
        helper.make_node("Add", ["MM2", "B2"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "backend_matmul_chain", [x], [y], initializer=inits)
    _save_model(out_dir / "backend_matmul_chain.onnx", graph, opset)


def build_backend_gemm_bias(out_dir: Path, opset: int, rng: np.random.Generator) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [8, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [8, 32])
    inits = [_f32("W", [16, 32], rng), _f32("C", [32], rng)]
    nodes = [helper.make_node("Gemm", ["X", "W", "C"], ["Y"], alpha=1.0, beta=1.0, transB=0)]
    graph = helper.make_graph(nodes, "backend_gemm_bias", [x], [y], initializer=inits)
    _save_model(out_dir / "backend_gemm_bias.onnx", graph, opset)


def build_backend_add_equal(out_dir: Path, opset: int) -> None:
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 8, 16])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 8, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8, 16])
    nodes = [helper.make_node("Add", ["A", "B"], ["Y"])]
    graph = helper.make_graph(nodes, "backend_add_equal", [a, b], [y])
    _save_model(out_dir / "backend_add_equal.onnx", graph, opset)


def build_backend_tanh_unary(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 8, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8, 16])
    nodes = [helper.make_node("Tanh", ["X"], ["Y"])]
    graph = helper.make_graph(nodes, "backend_tanh_unary", [x], [y])
    _save_model(out_dir / "backend_tanh_unary.onnx", graph, opset)


def build_backend_softmax_last_axis(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 8, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8, 16])
    nodes = [helper.make_node("Softmax", ["X"], ["Y"], axis=-1)]
    graph = helper.make_graph(nodes, "backend_softmax_last_axis", [x], [y])
    _save_model(out_dir / "backend_softmax_last_axis.onnx", graph, opset)


def build_backend_layernorm_last_axis(out_dir: Path, opset: int, rng: np.random.Generator) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 16])
    inits = [_f32("Scale", [16], rng), _f32("Bias", [16], rng)]
    nodes = [helper.make_node("LayerNormalization", ["X", "Scale", "Bias"], ["Y"], axis=-1, epsilon=1e-5)]
    graph = helper.make_graph(nodes, "backend_layernorm_last_axis", [x], [y], initializer=inits)
    _save_model(out_dir / "backend_layernorm_last_axis.onnx", graph, opset)


def build_cpu_reshape_static(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 3])
    inits = [_i64("shape", [2, 4, 3])]
    nodes = [helper.make_node("Reshape", ["X", "shape"], ["Y"])]
    graph = helper.make_graph(nodes, "cpu_reshape_static", [x], [y], initializer=inits)
    _save_model(out_dir / "cpu_reshape_static.onnx", graph, opset)


def build_cpu_transpose_perm(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2, 4])
    nodes = [helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])]
    graph = helper.make_graph(nodes, "cpu_transpose_perm", [x], [y])
    _save_model(out_dir / "cpu_transpose_perm.onnx", graph, opset)


def build_cpu_squeeze_axes(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 1, 4, 1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    inits = [_i64("axes", [1, 3])]
    nodes = [helper.make_node("Squeeze", ["X", "axes"], ["Y"])]
    graph = helper.make_graph(nodes, "cpu_squeeze_axes", [x], [y], initializer=inits)
    _save_model(out_dir / "cpu_squeeze_axes.onnx", graph, opset)


def build_cpu_unsqueeze_axes(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1, 4, 1])
    inits = [_i64("axes", [1, 3])]
    nodes = [helper.make_node("Unsqueeze", ["X", "axes"], ["Y"])]
    graph = helper.make_graph(nodes, "cpu_unsqueeze_axes", [x], [y], initializer=inits)
    _save_model(out_dir / "cpu_unsqueeze_axes.onnx", graph, opset)


def build_cpu_reduce_mean_axes(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 1, 8])
    inits = [_i64("axes", [1])]
    nodes = [helper.make_node("ReduceMean", ["X", "axes"], ["Y"], keepdims=1)]
    graph = helper.make_graph(nodes, "cpu_reduce_mean_axes", [x], [y], initializer=inits)
    _save_model(out_dir / "cpu_reduce_mean_axes.onnx", graph, opset)


def build_cpu_cast_to_fp16(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 4, 8])
    nodes = [helper.make_node("Cast", ["X"], ["Y"], to=TensorProto.FLOAT16)]
    graph = helper.make_graph(nodes, "cpu_cast_to_fp16", [x], [y])
    _save_model(out_dir / "cpu_cast_to_fp16.onnx", graph, opset)


def build_cpu_where_broadcast(out_dir: Path, opset: int, rng: np.random.Generator) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 8])
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 4, 8])
    inits = [_bool("cond", [1, 4, 1], rng)]
    nodes = [helper.make_node("Where", ["cond", "X", "Y"], ["Z"])]
    graph = helper.make_graph(nodes, "cpu_where_broadcast", [x, y], [z], initializer=inits)
    _save_model(out_dir / "cpu_where_broadcast.onnx", graph, opset)


def build_cpu_expand_static(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 8])
    inits = [_i64("shape", [2, 4, 8])]
    nodes = [helper.make_node("Expand", ["X", "shape"], ["Y"])]
    graph = helper.make_graph(nodes, "cpu_expand_static", [x], [y], initializer=inits)
    _save_model(out_dir / "cpu_expand_static.onnx", graph, opset)


def build_cpu_concat_axis1(out_dir: Path, opset: int) -> None:
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3, 4])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 2, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 5, 4])
    nodes = [helper.make_node("Concat", ["A", "B"], ["Y"], axis=1)]
    graph = helper.make_graph(nodes, "cpu_concat_axis1", [a, b], [y])
    _save_model(out_dir / "cpu_concat_axis1.onnx", graph, opset)


def build_cpu_gather_axis1(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 5, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 4])
    inits = [_i64("indices", [0, 3])]
    nodes = [helper.make_node("Gather", ["X", "indices"], ["Y"], axis=1)]
    graph = helper.make_graph(nodes, "cpu_gather_axis1", [x], [y], initializer=inits)
    _save_model(out_dir / "cpu_gather_axis1.onnx", graph, opset)


def build_cpu_slice_basic(out_dir: Path, opset: int) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 5, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])
    inits = [
        _i64("starts", [1, 0]),
        _i64("ends", [4, 8]),
        _i64("axes", [1, 2]),
        _i64("steps", [1, 2]),
    ]
    nodes = [helper.make_node("Slice", ["X", "starts", "ends", "axes", "steps"], ["Y"])]
    graph = helper.make_graph(nodes, "cpu_slice_basic", [x], [y], initializer=inits)
    _save_model(out_dir / "cpu_slice_basic.onnx", graph, opset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="bench_models/op_coverage",
        help="Output directory for generated ONNX models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    rng = np.random.default_rng(args.seed)

    builders = [
        build_backend_matmul_chain,
        build_backend_gemm_bias,
        build_backend_add_equal,
        build_backend_tanh_unary,
        build_backend_softmax_last_axis,
        build_backend_layernorm_last_axis,
        build_cpu_reshape_static,
        build_cpu_transpose_perm,
        build_cpu_squeeze_axes,
        build_cpu_unsqueeze_axes,
        build_cpu_reduce_mean_axes,
        build_cpu_cast_to_fp16,
        build_cpu_where_broadcast,
        build_cpu_expand_static,
        build_cpu_concat_axis1,
        build_cpu_gather_axis1,
        build_cpu_slice_basic,
    ]

    for builder in builders:
        if "rng" in builder.__code__.co_varnames:
            builder(out_dir, args.opset, rng)
        else:
            builder(out_dir, args.opset)

    print(f"Generated {len(builders)} models in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
