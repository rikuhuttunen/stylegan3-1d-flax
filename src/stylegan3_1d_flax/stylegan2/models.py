# Modified for 1D signals based on https://github.com/NVlabs/stylegan3
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from typing import Any, Callable


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def upfirdn1d(
        x: jax.Array,   # [NCW]
        f: jax.Array,   # [W]
        up: int = 1,
        down: int = 1,
        padding: tuple[int, int] = (0, 0),
        gain: float = 1.0) -> jax.Array:
    """Upsample, pad, crop, filter, and downsample. Each stage is optional.

    Modified from https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/upfirdn2d.py#L167
    """
    # Upsample by inserting zeros
    # Note that this shifts the signal, which may need to be accounted for in `padding`
    x = jax.lax.pad(x, 0.0, [(0, 0, 0), (0, 0, 0), (0, up - 1, up - 1)])

    # Pad or crop
    x = jax.lax.pad(x, 0.0, [(0, 0, 0), (0, 0, 0), (max(0, padding[0]), max(0, padding[1]), 0)])
    x = x[:, :, max(-padding[0], 0) : jnp.shape(x)[-1] - max(-padding[1], 0)]

    # Convolve with the filter
    # Note that if 'SAME' padding is desired, the padding needs to be
    # explicitly defined in the `padding` kwarg.
    n_channels = jnp.shape(x)[1]
    f /= jnp.sum(f)
    f *= gain
    f = f[jnp.newaxis, jnp.newaxis, :]
    f = jnp.tile(f, [n_channels, 1, 1])  # [OIW]
    dn = jax.lax.conv_dimension_numbers(x.shape, f.shape, ('NCW', 'OIW', 'NCW'))
    x = jax.lax.conv_general_dilated(
        x, f,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=dn,
        feature_group_count=n_channels
    )

    # Downsample by keeping every `down`th sample
    x = x[:, :, ::down]

    return x


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def conv1d_resample(
        x: jax.Array,                           # [NCW]
        w: jax.Array,                           # [OIW]
        f: jax.Array                ,           # [W]
        up: int = 1,
        down: int = 1,
        padding: tuple[int, int] = (0, 0),  # Left and right padding w.r.t. the upsampled signals
        feature_group_count: int = 1,
        batch_group_count: int = 1) -> jax.Array:
    """Convolution with optional filtering, upsampling, and downsampling.
    
    Modified from https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/conv2d_resample.py#L46
    """
    fw = jnp.shape(f)[0]
    pl, pr = padding

    # Adjust padding w.r.t up/downsampling
    if up > 1:
        pl += (fw + up - 1) // 2
        pr += (fw - up) // 2
    if down > 1:
        pl += (fw - down + 1) // 2
        pr += (fw - down) // 2

    x = upfirdn1d(x, f=(f if up > 1 else jnp.array([1.])), up=up, padding=(pl, pr), gain=up)

    dn = jax.lax.conv_dimension_numbers(x.shape, w.shape, ('NCW', 'OIW', 'NCW'))
    x = jax.lax.conv_general_dilated(
        x, w,
        window_strides=(1,),
        padding='VALID',
        dimension_numbers=dn,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count
    )

    if down > 1:
        x = upfirdn1d(x, f, down=down)

    return x


class Dense(nnx.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            use_bias: bool = True,
            activation: Callable | None = None,
            lr_multiplier: float = 1.0,
            bias_init: float = 0.0,
            rngs: nnx.Rngs):
        kernel_key = rngs.params()
        kernel_init = nnx.initializers.normal(stddev=1.0 / lr_multiplier)
        self.kernel = nnx.Param(kernel_init(kernel_key, (in_features, out_features)))

        if use_bias:
            self.bias = nnx.Param(jnp.full((out_features,), bias_init))
        else:
            self.bias = nnx.Param(None)

        self.activation = activation
        self.lr_multiplier = lr_multiplier
        self.weight_gain = np.float32(lr_multiplier / np.sqrt(in_features))

    def __call__(self, inputs: jax.Array) -> jax.Array:
        kernel = self.kernel.value
        kernel *= self.weight_gain
        bias = self.bias.value

        y = jax.lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ()))
        )

        if bias is not None:
            bias *= self.lr_multiplier
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        if self.activation is None:
            return y
        
        return self.activation(y)


class ResampleFilter(nnx.Variable): pass


class Conv1D(nnx.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: int,
            *,
            use_bias: bool = True,
            activation: Callable | None = None,
            up: int = 1,
            down: int = 1,
            resample_filter: tuple[float, ...] = (1, 3, 3, 1),
            rngs: nnx.Rngs):
        self.resample_filter = ResampleFilter(jnp.array(resample_filter))
        self.activation = activation
        self.up = up
        self.down = down
        self.kernel_size = kernel_size
        self.out_features = out_features

        kernel_key = rngs.params()
        kernel_init = nnx.initializers.normal(stddev=1.0)
        kernel_shape = (out_features, in_features, kernel_size)
        self.kernel = nnx.Param(kernel_init(kernel_key, kernel_shape))
        self.weight_gain = np.float32(1.0 / np.sqrt(in_features * kernel_size))

        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_features,)))
        else:
            self.bias = nnx.Param(None)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        kernel = self.kernel.value
        kernel *= self.weight_gain
        bias = self.bias.value

        y = conv1d_resample(
            inputs,
            kernel,
            f=jnp.array(self.resample_filter, dtype=jnp.float32),
            up=self.up,
            down=self.down,
            padding = (self.kernel_size // 2,) * 2
        )

        if bias is not None:
            bias = jnp.reshape(bias, (1, self.out_features, 1))
            y += bias

        if self.activation is None:
            return y
        
        return self.activation(y)


def scaled_lrelu(x, negative_slope=0.2, gain=np.sqrt(2)):
    return nnx.leaky_relu(x, negative_slope=negative_slope) * gain


class DiscriminatorBlock(nnx.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            kernel_size: int,
            *,
            input_channels: int | None = None,
            activation: Callable = partial(scaled_lrelu, negative_slope=0.2, gain=np.sqrt(2)),
            resample_filter: tuple[float, ...] = (1, 3, 3, 1),
            rngs: nnx.Rngs):
        self.input_channels = input_channels
        if input_channels is not None:
            self.from_input = Conv1D(
                in_features=input_channels,
                out_features=in_features,
                kernel_size=1,
                activation=activation,
                rngs=rngs
            )
        f = jnp.array(resample_filter, dtype=jnp.float32)
        self.conv_res = Conv1D(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1,
            use_bias=False,
            down=2,
            resample_filter=f,
            rngs=rngs
        )
        self.conv_0 = Conv1D(
            in_features=in_features,
            out_features=in_features,
            kernel_size=kernel_size,
            activation=activation,
            rngs=rngs
        )
        self.conv_1 = Conv1D(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            activation=activation,
            down=2,
            resample_filter=f,
            rngs=rngs
        )

    def __call__(self, x):
        if self.input_channels is not None:
            x = self.from_input(x)

        y = self.conv_res(x) * np.sqrt(0.5)
        x = self.conv_0(x)
        x = self.conv_1(x) * np.sqrt(0.5)
        x = x + y

        return x


class MiniBatchStd(nnx.Module):

    def __init__(
            self,
            group_size: int = 4,
            num_new_features: int = 1):
        self.group_size = group_size
        self.num_new_features = num_new_features

    def __call__(self, x: jax.Array) -> jax.Array:
        _, C, W = jnp.shape(x)
        G = self.group_size
        F = self.num_new_features
        c = C // F

        y = jnp.reshape(x, (G, -1, F, c, W))    # [GnFcW]
        y = y - jnp.mean(y, axis=0)             # [GnFcW]
        y = jnp.mean(jax.lax.square(y), axis=0)     # [nFcW]
        y = jnp.sqrt((y + 1e-8))                # [nFcW]
        y = jnp.mean(y, axis=[2, 3])            # [nF]
        y = jnp.reshape(y, (-1, F, 1))          # [nF1]
        y = jnp.tile(y, (G, 1, W))              # [NFW]
        x = jnp.concatenate([x, y], axis=1)     # [N(C+F)W]

        return x


class DiscriminatorEpilogue(nnx.Module):

    def __init__(
            self,
            *,
            in_features: int,
            kernel_size: int,
            input_width: int,
            activation: Callable = partial(scaled_lrelu, negative_slope=0.2, gain=np.sqrt(2)),
            mbstd_group_size: int | None = 4,
            mbstd_num_new_features: int = 1,
            rngs: nnx.Rngs):
        self.mbstd_group_size = mbstd_group_size
        if mbstd_group_size is not None:
            self.mbstd = MiniBatchStd(group_size=mbstd_group_size, num_new_features=mbstd_num_new_features)
        else:
            mbstd_num_new_features = 0

        self.conv = Conv1D(
            in_features=in_features + mbstd_num_new_features,
            out_features=in_features,
            kernel_size=kernel_size,
            activation=activation,
            rngs=rngs
        )
        self.dense = Dense(
            in_features=in_features*input_width,
            out_features=in_features,
            activation=activation,
            rngs=rngs
        )
        self.out = Dense(
            in_features=in_features,
            out_features=1,
            rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.mbstd_group_size is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        
        # Flatten before dense layers
        x = jnp.reshape(x, (jnp.shape(x)[0], -1))

        x = self.dense(x)
        x = self.out(x)

        return x


class Discriminator(nnx.Module):
    
    def __init__(
            self,
            *,
            input_channels: int,
            input_width: int,
            block_args: list[dict[str, Any]],
            mbstd_group_size: int | None = 4,
            mbstd_num_new_features: int = 1,
            activation: Callable = partial(scaled_lrelu, negative_slope=0.2, gain=np.sqrt(2)),
            resample_filter: tuple[float, ...] = (1, 3, 3, 1),
            rngs: nnx.Rngs):
        # self.blocks = []
        self.blocks = nnx.data([])
        for i in range(len(block_args)):
            in_features = block_args[i]['features']
            
            if i == len(block_args) - 1:
                out_features = in_features
            else:
                out_features = block_args[i+1]['features']
            
            _block = DiscriminatorBlock(
                in_features=in_features,
                out_features=out_features,
                kernel_size=block_args[i]['kernel_size'],
                input_channels=input_channels if i == 0 else None,
                activation=activation,
                resample_filter=resample_filter,
                rngs=rngs
            )
            self.blocks.append(_block)

        # Each block downsamples the inputs by 2
        last_block_output_width = input_width // 2**len(block_args)
        self.epilogue = DiscriminatorEpilogue(
            in_features=block_args[-1]['features'],
            kernel_size=3,
            input_width=last_block_output_width,
            activation=activation,
            mbstd_group_size=mbstd_group_size,
            mbstd_num_new_features=mbstd_num_new_features,
            rngs=rngs
        )

    def __call__(self, inputs):
        # Transpose [NWC] to [NCW]
        x = jnp.transpose(inputs, axes=[0, 2, 1])

        for block in self.blocks:
            x = block(x)

        x = self.epilogue(x)

        return x
