# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1121

from functools import partial
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Callable[..., Callable]
InitFn = Callable[[Any, Iterable[int], Any], Any]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.relu
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    kernel_init: InitFn = nn.initializers.kaiming_normal()
    bias_init: InitFn = nn.initializers.zeros
    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)

    @nn.compact
    def __call__(self, x):
        x = self.conv_cls(
            self.n_filters, self.kernel_size, self.strides, use_bias=not self.norm_cls, padding=self.padding, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        scale_init = nn.initializers.zeros if self.is_last else nn.initializers.ones
        mutable = self.is_mutable_collection("batch_stats")
        x = self.norm_cls(use_running_average=not mutable, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x


class PoolSize(nn.Module):
    kernel: int
    stride: int
    padding: Any

    @nn.compact
    def __call__(self, x):
        return nn.max_pool(x, window_shape=(self.kernel, self.kernel), strides=(self.stride, self.stride), padding=self.padding)


class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x):
        x = self.conv_block_cls(64, kernel_size=(7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)])(x)
        x = PoolSize(kernel=3, stride=2, padding=((1, 1), (1, 1)))(x)
        return x


class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = self.conv_block_cls(out_shape[-1], kernel_size=(1, 1), strides=self.strides, activation=lambda y: y)(x)
        return x


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)], strides=self.strides)(x)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)], is_last=True)(y)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Sequential(nn.Module):
    layers: Sequence[Union[nn.Module, Callable[[jnp.ndarray], jnp.ndarray]]]

    @nn.compact
    def __call__(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


def ResNet(
    block_cls: ModuleDef,
    stage_sizes: Sequence[int],
    momentum: float,
    n_classes: int,
    hidden_sizes: Sequence[int],
    conv_cls: ModuleDef = nn.Conv,
    conv_block_cls: ModuleDef = ConvBlock,
    stem_cls: ModuleDef = ResNetStem,
) -> Sequential:
    norm_cls = partial(nn.BatchNorm, momentum=momentum)
    conv_block_cls = partial(conv_block_cls, conv_cls=conv_cls, norm_cls=norm_cls)
    stem_cls = partial(stem_cls, conv_block_cls=conv_block_cls)
    block_cls = partial(block_cls, conv_block_cls=conv_block_cls)

    layers = [stem_cls()]
    for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
        for b in range(n_blocks):
            strides = (1, 1) if i == 0 or b != 0 else (2, 2)
            layers.append(block_cls(n_hidden=hsize, strides=strides))

    layers.append(partial(jnp.mean, axis=(1, 2)))
    layers.append(nn.Dense(n_classes))
    return Sequential(layers)


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], stem_cls=ResNetStem, block_cls=ResNetBlock, hidden_sizes=(64, 128, 256, 512))
