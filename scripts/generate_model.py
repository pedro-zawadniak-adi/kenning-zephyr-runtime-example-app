# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Script for generating random TFLite models.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import numpy as np
import json

if TYPE_CHECKING:
    from keras.src.engine.keras_tensor import KerasTensor

def split_z(value: int, n: int) -> np.ndarray:
    """
    Randomly partitions natural number into `n` segments.

    Parameters
    ----------
    value : int
        Value to split.
    n : int
        Number of segments.

    Returns
    -------
    np.ndarray
        One-dimensional array denoting partitions.
    """
    splits = np.random.choice(np.arange(1, value), size=n - 1,replace=False)
    return np.diff(np.concatenate(([0], np.sort(splits), [value])))


def sample_keras_dense(
    inp: list[list[int]] | list[KerasTensor],
    out: list[int] | None,
    n_layers: int,
    n_units: int,
    **kwargs,
) -> tuple[KerasTensor | list[KerasTensor] | None, list[KerasTensor]]:
    """
    Creates sample dense NN layers.

    Parameters
    ----------
    inp : list[list[int]] | list[KerasTensor]
        Input shape or preceding layer.
    out : list[int] | None
        Output shape if output layers are required.
    n_layers : int
        Number of hidden layers.
    n_units : int
        Total number of neurons in hidden layers.
    **kwargs
        Other unused parameters.

    Returns
    -------
    tuple[KerasTensor | list[KerasTensor] | None, list[KerasTensor]]
        Input and output layers.

    Raises
    ------
    ValueError
        Raised if failed to partition `n_units` into `n_layers`.
    """
    import keras
    from keras.src.engine.keras_tensor import KerasTensor

    if isinstance(inp, KerasTensor) or inp and isinstance(inp[0], KerasTensor):
        inputs = None
        x = inp
    elif len(inp) == 1:
        inputs = keras.Input(inp[0])
        x = inputs
    else:
        inputs = [keras.Input(i) for i in inp]
        x = keras.layers.Concatenate()(inputs)

    if n_layers < 1 or n_units < n_layers:
        raise ValueError("Require n_layers >= 1 and n_units >= n_layers")

    units_per_layer = split_z(n_units, n_layers)

    for units in units_per_layer:
        x = keras.layers.Dense(units, activation="relu")(x)

    if out:
        outputs = [keras.layers.Dense(o, activation="softmax")(x) for o in out]
    else:
        outputs = x

    return inputs, outputs

def sample_keras_conv(
    inp: list[list[int]],
    out: list[int] | None,
    n_blocks: int,
    n_filters: int,
    kernel_size: int,
    pool_size: int,
    batchnorm: bool,
    pool_after_block: bool,
    pooling_mode: Literal['global'] | Literal['flatten'],
    **kwargs,
) -> tuple[list[KerasTensor], list[KerasTensor]]:
    """

    Parameters
    ----------
    inp : list[list[int]]
        Input shape.
    out : list[int] | None
        Output shape if output layers are required.
    n_blocks : int
        Number of convolutional blocks.
    n_filters : int
        Total number of filters in convolutional layers.
    kernel_size : int
        Kernel size.
    pool_size : int
        Pooling size.
    batchnorm : bool
        Whether to use `BatchNormalization` layer after convolution.
    pool_after_block : bool
        Whether to use `MaxPool2D` layer after convolution.
    pooling_mode : Literal['global'] | Literal['flatten']
        Flattening method, either `global` for `GlobalAveragePooling2D` layer or `flatten` for `Flatten` layer.
    **kwargs
        Other unused parameters.

    Returns
    -------
    tuple[list[KerasTensor], list[KerasTensor]]
        Input and output layers.

    Raises
    ------
    ValueError
        Raised if failed to partition `n_filters` into `n_blocks` or if `pooling_mode` got unsupported value.
    """
    """Creates sample convolutional NN."""
    import keras

    if n_blocks < 1 or n_filters < n_blocks:
        raise ValueError("Require n_blocks >= 1 and n_filters >= n_blocks")

    filters_per_block = split_z(n_filters, n_blocks)
    splits = np.random.choice(np.arange(1, n_blocks), size=len(inp) - 1, replace=False)
    filters_per_input = np.split(filters_per_block, np.sort(splits))

    inputs = []
    head_outputs = []
    for i, filters in enumerate(filters_per_input):
        x = keras.Input(shape=inp[i])
        inputs.append(x)
        for filt in filters:
            x = keras.layers.Conv2D(
                int(filt),
                kernel_size=kernel_size,
                padding="same",
                activation=None,
                use_bias=not batchnorm,
            )(x)
            if batchnorm:
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation("relu")(x)
            if pool_after_block:
                x = keras.layers.MaxPool2D(pool_size=pool_size, padding="same")(x)

        if pooling_mode == "global":
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling_mode == "flatten":
            x = keras.layers.Flatten()(x)
        else:
            raise ValueError("pooling_mode must be 'global' or 'flatten'")

        head_outputs.append(x)

    if len(head_outputs) == 1:
        x = head_outputs[0]
    else:
        x = keras.layers.Concatenate()(head_outputs)

    if out:
        outputs = [keras.layers.Dense(o, activation="softmax")(x) for o in out]
    else:
        outputs = x

    return inputs, outputs


def create_io_spec(inputs: KerasTensor | list[KerasTensor], outputs: KerasTensor | list[KerasTensor]) -> dict:
    """
    Creates Kenning IO specification based on input and output tensors.

    Parameters
    ----------
    inputs : KerasTensor | list[KerasTensor]
        Input tensors.
    outputs : KerasTensor | list[KerasTensor]
        Output layers.

    Returns
    -------
    dict
        IO Specification.
    """
    io_spec = {}
    for io_name, io in (("input", inputs), ("output", outputs)):
        layers = io if isinstance(io, list) else [io]
        io_spec_layers = []
        for layer in layers:
            name = layer.name
            shape = [dim if dim is not None else 1 for dim in layer.shape]
            if isinstance(layer.dtype, str):
                dtype = layer.dtype
            else:
                dtype = layer.dtype.as_numpy_dtype.__name__

            io_spec_layers.append({
                "name": name,
                "shape": shape,
                "dtype": dtype,
            })
        io_spec[io_name] = io_spec_layers
    return io_spec

def keras_compile(inputs: KerasTensor | list[KerasTensor], outputs: KerasTensor | list[KerasTensor]) -> bytes:
    """
    Creates Keras from inputs and outputs, compiles it, and converts it to TFLite.

    Parameters
    ----------
    inputs : KerasTensor | list[KerasTensor]
        Input tensors.
    outputs : KerasTensor | list[KerasTensor]
        Output tensors.

    Returns
    -------
    bytes
        TFLite buffer.
    """
    import keras
    import tensorflow as tf

    model = keras.Model(inputs, outputs)
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model)
    return tflite_model.convert()


def csv_to_int_list(s: str) -> list[int]:
    """
    Argparse type for comma-separated integer values.

    Parameters
    ----------
    s : str
        Value provided from CLI

    Returns
    -------
    list[int]
        Parsed integers.
    """
    return [int(x) for x in s.split(",") if x != ""]

def add_io_args(parser: argparse.ArgumentParser):
    """
    Adds input/output shape parameters.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to modify.
    """
    parser.add_argument(
        "--input",
        dest="inp",
        type=csv_to_int_list,
        action="append",
        help="Multiple CSV, e.g. '--input 1,2,3 --input 4,5,6' is converted into [[1,2,3],[4,5,6]] input shape",
    )
    parser.add_argument(
        "--output",
        dest="out",
        type=csv_to_int_list,
        action="append",
        help="Single CSV, --output 2,3 is converted into [2,3] output shape",
    )


def add_dense_args(parser: argparse.ArgumentParser):
    """
    Adds parameters for dense NN block.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to modify.
    """
    parser.add_argument(
        "--n-layers",
        type=int,
        help="Number of dense layers",
        default=5,
    )
    parser.add_argument(
        "--n-units",
        type=int,
        help="Total number of neurons in dense layers",
        default=50,
    )

def add_conv_args(parser: argparse.ArgumentParser):
    """
    Adds parameters for convolutional NN.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to modify.
    """
    parser.add_argument(
        "--n-blocks",
        type=int,
        help="Number of convolutional layers",
        default=2
    )
    parser.add_argument(
       "--n-filters",
        type=int,
        help="Total number of filters",
        default=6,
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        help="Kernel size",
        default=3,
    )
    parser.add_argument(
        "--batchnorm",
        action=argparse.BooleanOptionalAction,
        help="Whether to use batch normalization after convolutional layer",
        default=True,
    )
    parser.add_argument(
        "--pool-after-block",
        action=argparse.BooleanOptionalAction,
        help="Whether to use max pooling after convolutional layer",
        default=True,
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        help="Pool size for max pooling",
        default=2,
    )
    parser.add_argument(
        "--pooling-mode",
        type=str,
        choices=["flatten", "global"],
        default="flatten",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tflite-model-path",
        type=Path,
        help="Path to TFLite model",
        required=True,
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        help="Whether to run sample input on the generated model",
        default=True,
    )

    subparsers = parser.add_subparsers(dest="command")

    dense_parser = subparsers.add_parser("dense", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_io_args(dense_parser)
    add_dense_args(dense_parser)

    conv_parser = subparsers.add_parser("conv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_io_args(conv_parser)
    add_conv_args(conv_parser)

    mixed_parser = subparsers.add_parser("mixed", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_io_args(mixed_parser)
    add_dense_args(mixed_parser)
    add_conv_args(mixed_parser)

    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import tensorflow as tf

    print("Creating TFlite model")
    if args.command == "dense":
        args.inp = args.inp or [[20]]
        args.out = args.out or [5]
        inputs, outputs = sample_keras_dense(**vars(args))
    elif args.command == "conv":
        args.inp = args.inp or [[12, 12, 1]]
        args.out = args.out or [5]
        inputs, outputs = sample_keras_conv(**vars(args))
    else:
        args.inp = args.inp or [[12, 12, 1]]
        args.out = args.out or [5]
        inputs, conv_outputs = sample_keras_conv(**vars(args) | {"out": None})
        _, outputs = sample_keras_dense(**vars(args) | {"inp": conv_outputs})

    io_spec = create_io_spec(inputs, outputs)
    buf = keras_compile(inputs, outputs)

    path: Path = args.tflite_model_path
    path.with_suffix(".tflite").write_bytes(buf)
    path.with_suffix(".tflite.json").write_text(json.dumps(io_spec))

    if args.check:
        print("Checking TFLite model")
        interpreter = tf.lite.Interpreter(model_path=str(args.tflite_model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()

        for input_detail in input_details:
            input_shape = input_detail['shape']
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            interpreter.set_tensor(input_detail['index'], input_data)

        interpreter.invoke()
