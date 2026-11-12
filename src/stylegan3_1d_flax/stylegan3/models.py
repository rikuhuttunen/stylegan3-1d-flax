# StyleGAN3 generator modified for 1D signals based on https://github.com/NVlabs/stylegan3
import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal

from flax import nnx
from functools import partial
from jax import lax
from typing import Callable, Sequence


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
    x = lax.pad(x, 0.0, [(0, 0, 0), (0, 0, 0), (0, up - 1, up - 1)])

    # Pad or crop
    x = lax.pad(x, 0.0, [(0, 0, 0), (0, 0, 0), (max(0, padding[0]), max(0, padding[1]), 0)])
    x = x[:, :, max(-padding[0], 0) : jnp.shape(x)[-1] - max(-padding[1], 0)]
 
    # Convolution for each sample and channel with the same 1D filter
    separate_conv = jax.vmap(
        jax.vmap(
            lambda _x, _f: jnp.convolve(_x, _f, mode='same'), in_axes=(0, None)
        ), in_axes=(0, None)
    )

    # Convolve with the filter
    f *= gain
    x = separate_conv(x, f)

    # Downsample by keeping every `down`th sample
    x = x[:, :, ::down]

    return x


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
def filtered_lrelu(
        x: jax.Array,  # [NCW]
        fu: jax.Array,  # [W_fu]
        fd: jax.Array,  # [W_fd]
        bias: jax.Array,  # [C]
        up: int = 1,
        down: int = 1,
        padding: tuple[int, int] = (0, 0),
        gain: float = np.sqrt(2),
        slope: float = 0.2,
        clamp: float | None = None) -> jax.Array:
    # Apply bias
    if bias is not None:
        x = x + jnp.reshape(bias, [1, -1, 1])

    # Upsample
    x = upfirdn1d(
        x=x,
        f=fu,
        up=up,
        padding=padding,
        gain=up
    )

    # Apply ReLU
    x = nnx.leaky_relu(x, negative_slope=slope) * gain
    
    # Apply clamp
    if clamp is not None:
        x = lax.clamp(-clamp, x, clamp)

    # Downsample
    x = upfirdn1d(
        x=x,
        f=fd,
        down=down
    )

    return x


@partial(jax.jit, static_argnums=(2, 3, 4))
def conv1d(
        x: jax.Array,  # [NCW]
        w: jax.Array,  # [OIW]
        padding: tuple[int, int] = (0, 0),
        feature_group_count: int = 1,
        batch_group_count: int = 1) -> jax.Array:
    dn = lax.conv_dimension_numbers(x.shape, w.shape, ('NCW', 'OIW', 'NCW'))
    x = lax.conv_general_dilated(
        x, w,
        window_strides=(1,),
        padding=[padding],
        dimension_numbers=dn,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count
    )

    return x


@partial(jax.jit, static_argnums=(3, 4))
def modulated_conv1d(
        x: jax.Array,  # [NCW]
        w: jax.Array,  # [OIW]
        s: jax.Array,  # [NI]
        demodulate: bool = True,
        padding: tuple[int, int] = (0, 0),
        input_gain: float | None = None) -> jax.Array:
    """Modulated conv from stylegan3.
    
    Modified from https://github.com/NVlabs/stylegan3/blob/main/training/networks_stylegan3.py#L25
    """
    batch_size, n_channels, _ = jnp.shape(x)
    _, in_channels, kw = jnp.shape(w)
    assert n_channels == in_channels

    if demodulate:
        w = w * lax.rsqrt(jnp.mean(lax.square(w), axis=[1,2], keepdims=True))
        s = s * lax.rsqrt(jnp.mean(lax.square(s)))
 
    w = w[jnp.newaxis]  # [NOIW]
    w = w * s[:, jnp.newaxis, :, jnp.newaxis]  # [NOIW]

    if demodulate:
        dcoefs = lax.rsqrt(jnp.sum(lax.square(w), axis=[2, 3]) + 1e-8)  # [NO]
        w = w * dcoefs[:, :, jnp.newaxis, jnp.newaxis]  # [NOIW]

    if input_gain is not None:
        input_gain = jnp.broadcast_to(jnp.array(input_gain), (batch_size, in_channels))  # [NI]
        w = w * jnp.expand_dims(input_gain, (1, 3))

    # Utilize grouped convolution to perform convolution separately for each sample
    x = jnp.reshape(x, (1, -1, jnp.shape(x)[-1]))
    w = jnp.reshape(w, (-1, in_channels, kw))

    x = conv1d(x, w, padding=padding, feature_group_count=batch_size)

    x = jnp.reshape(x, (batch_size, -1, jnp.shape(x)[-1]))

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
            bias_init: float | Sequence[float] = 0.0,
            weight_init: float = 1.0,
            rngs: nnx.Rngs):
        kernel_key = rngs.Params()
        kernel_init = nnx.initializers.normal(stddev=weight_init / lr_multiplier)
        self.kernel = nnx.Param(kernel_init(kernel_key, (in_features, out_features)))

        if use_bias:
            self.bias = nnx.Param(jnp.broadcast_to(np.array(bias_init), (out_features,)))
        else:
            self.bias = nnx.Param(None)

        self.activation = activation
        self.lr_multiplier = lr_multiplier
        self.weight_gain = np.float32(lr_multiplier / np.sqrt(in_features))

    def __call__(self, inputs: jax.Array) -> jax.Array:
        kernel = self.kernel.value
        kernel *= self.weight_gain
        bias = self.bias.value

        y = lax.dot_general(
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


class MappingNetwork(nnx.Module):

    def __init__(
            self,
            *,
            z_dim: int,
            w_dim: int,
            n_ws: int,
            n_layers: int,
            lr_multiplier: int,
            activation=partial(nnx.leaky_relu, negative_slope=0.2),
            rngs=nnx.Rngs):
        self.n_ws = n_ws
        self.layers = nnx.data([])
        for i in range(n_layers):
            _layer = Dense(
                in_features=z_dim if i == 0 else w_dim,
                out_features=w_dim,
                activation=activation,
                lr_multiplier=lr_multiplier,
                rngs=rngs
            )
            self.layers.append(_layer)

    def __call__(self, z: jax.Array) -> jax.Array:
        # Normalize input
        x = z * lax.rsqrt(jnp.mean(lax.square(z), axis=1, keepdims=True) + 1e-8)
        for layer in self.layers:
            x = layer(x) * np.sqrt(2)  # Multiply by sqrt(2) to match the default gain of stylegan2

        x = jnp.tile(x[:, jnp.newaxis], (1, self.n_ws, 1))

        return x

# Custom type for storing random fourier features.
# Maybe unnecessary complication.
class RFFVariable(nnx.Variable): pass  


class SynthesisInput(nnx.Module):
    
    def __init__(
            self,
            w_dim: int,
            channels: int,
            width: int,  # Output width in sampling points
            fs: float,
            bandwidth: float,
            rngs: nnx.Rngs):
        self.channels = channels
        self.width = width
        self.fs = fs

        # Draw random frequencies from [-bandwidth, bandwidth]
        self.freqs = RFFVariable(
            jax.random.uniform(rngs(), (channels,), minval=-1.0, maxval=1.0) * bandwidth
        )
        # Draw random phases from [-0.5, 0.5]
        self.phases = RFFVariable(
            jax.random.uniform(rngs(), (channels,), minval=-0.5, maxval=0.5)
        )
        # User-defined frequency scale and phase shift
        self.user_transform = RFFVariable(jnp.array([1.0, 0.0]))

        # Learnable weights from w to phase shift
        self.w_to_phase = Dense(
            in_features=w_dim,
            out_features=1,
            weight_init=0.0,
            bias_init=0.0,
            rngs=rngs
        )

        self.rff_weights = nnx.Param(
            jax.random.normal(rngs(), (channels, channels))
        )

        # Construct a sampling grid for the fourier features
        self.grid = RFFVariable(np.linspace(-0.5 * self.width / self.fs, 0.5 * self.width / self.fs, int(self.width)))

    def __call__(self, w: jax.Array) -> jax.Array:
        phases = self.phases.value
        freqs = self.freqs.value
        user_transform = self.user_transform.value

        # Map w to phase shift and apply to self.phases along with the user transform
        t = self.w_to_phase(w)
        phases = phases + (user_transform[1] + t[:, 0])[:, jnp.newaxis] * freqs[jnp.newaxis]

        # Clamp user transform to [-1, 1] to prevent out-of-band frequencies
        freqs = lax.clamp(-1.0, user_transform[0], 1.0) * freqs
        freqs = jnp.tile(freqs[jnp.newaxis], [jnp.shape(w)[0], 1])

        # Create the fourier features
        x = freqs[:, :, jnp.newaxis] * self.grid[jnp.newaxis, jnp.newaxis, :]
        x = x + phases[..., jnp.newaxis] * self.width
        x = jnp.sin(2 * jnp.pi * x)

        # Apply the learned linear mapping between the features
        norm_weights = self.rff_weights / np.sqrt(self.channels)
        x = jnp.transpose(x, [0, 2, 1]) @ norm_weights.transpose()

        # Return [NCW]
        return jnp.transpose(x, [0, 2, 1])


class EMA(nnx.Variable): pass


class SynthesisFilter(nnx.Variable): pass


class SynthesisLayer(nnx.Module):
    
    def __init__(
            self,
            *,
            w_dim: int,
            in_features: int,
            out_features: int,
            in_width: int,
            out_width: int,
            in_fs: float,
            out_fs: float,
            in_cutoff: float,
            out_cutoff: float,
            in_fh: float, out_fh: float,
            is_last_layer: bool,
            is_critically_sampled: bool,
            conv_ksize: int = 3,
            filter_size: int = 6,
            lrelu_upsampling: int = 2,
            conv_clamp: float | None = None,
            magnitude_ema_beta: float = 0.999,
            rngs: nnx.Rngs):
        self.in_features = in_features
        self.in_width = in_width
        self.out_width = out_width
        self.conv_ksize = conv_ksize
        self.conv_clamp = conv_clamp
        self.is_last_layer = is_last_layer
        self.magnitude_ema_beta = magnitude_ema_beta
        self.magnitude_ema = EMA(jnp.array(1.0))
        self.map_style = Dense(
            in_features=w_dim,
            out_features=in_features,
            bias_init=1.0,
            rngs=rngs
        )
        self.w_mod_conv = nnx.Param(jax.random.normal(rngs(), (out_features, in_features, conv_ksize)))
        
        tmp_fs = max(in_fs, out_fs) * (1.0 if self.is_last_layer else lrelu_upsampling)
        
        # Design upsampling filter
        self.up = int(np.rint(tmp_fs / in_fs))
        self.up_taps = filter_size * self.up if not is_last_layer else 1
        self.up_filter = SynthesisFilter(
            self.design_lowpass_filter(
                numtaps=self.up_taps,
                cutoff=in_cutoff,
                width=2 * in_fh,
                fs=tmp_fs
            )
        )
        # Design downsampling filter
        self.down = int(np.rint(tmp_fs / out_fs))
        self.down_taps = filter_size * self.down if not is_last_layer else 1
        self.down_filter = SynthesisFilter(
            self.design_lowpass_filter(
                numtaps=self.down_taps,
                cutoff=out_cutoff,
                width=2 * out_fh,
                fs=tmp_fs
            )
        )

        self.bias = nnx.Param(jnp.zeros((out_features)))

    def __call__(self, x: jax.Array, w: jax.Array, update_emas: bool = False) -> jax.Array:
        if update_emas:
            magnitude_cur = jnp.mean(jnp.square(x))
            self.magnitude_ema.value = magnitude_cur + self.magnitude_ema_beta * (self.magnitude_ema.value - magnitude_cur)

        # Compute styles
        styles = self.map_style(w)
        if self.is_last_layer:
            weight_gain = 1 / np.sqrt(self.in_features * self.conv_ksize)
            styles = styles * weight_gain

        # Modulated conv
        input_gain = lax.rsqrt(self.magnitude_ema.value)
        x = modulated_conv1d(
            x=x,
            w=self.w_mod_conv.value,
            s=styles,
            demodulate=(not self.is_last_layer),
            padding=(self.conv_ksize-1, self.conv_ksize-1),
            input_gain=input_gain
        )

        # Compute padding for filtered lrelu
        pad_total = (self.out_width - 1) * self.down + 1  # Desired output width before downsampling
        pad_total -= (self.in_width + self.conv_ksize - 1) * self.up  # Input size after upsampling
        # pad_total += self.up_taps + self.down_taps - 2  # Size reduction caused by the filters
        pad_low = (pad_total + self.up) // 2  # Shift sample locations similarly to the original stylegan3
        pad_high = pad_total - pad_low
        padding = (int(pad_low), int(pad_high))

        slope = 1.0 if self.is_last_layer else 0.2
        gain = 1.0 if self.is_last_layer else np.sqrt(2)

        x = filtered_lrelu(
            x=x,
            fu=self.up_filter.value,
            fd=self.down_filter.value,
            bias=self.bias.value,
            up=self.up,
            down=self.down,
            padding=padding,
            gain=gain,
            slope=slope,
            clamp=self.conv_clamp
        )

        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs):
        if numtaps == 1:
            return jnp.array([1], dtype=jnp.float32)
        
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)

        return jnp.array(f, dtype=jnp.float32)


class SynthesisNetwork(nnx.Module):

    def __init__(
            self,
            *,
            w_dim: int,
            width_sec: int,
            output_fs: int,
            output_channels: int,
            channel_base: int = int(32768/2**4),
            channel_max: int = 256,
            num_layers: int = 14,
            num_critical: int = 2,
            first_cutoff: float = 0.125,
            first_stopband: float = 0.3,
            last_stopband_rel: float = 2**0.3,
            margin_size: int = 10,
            output_scale: float = 1.0,
            conv_ksize: int = 3,
            last_conv_ksize: int = 1,
            filter_size: int = 6,
            lrelu_upsampling: int = 2,
            conv_clamp: float | None = None,
            magnitude_ema_beta: float = 0.999,
            rngs: nnx.Rngs):
        self.num_layers = num_layers
        self.output_scale = output_scale
        widths, channels, sampling_rates, cutoffs, half_widths = self.compute_param_progression(
            width_sec,
            output_fs,
            output_channels,
            channel_base,
            channel_max,
            margin_size,
            last_stopband_rel,
            num_layers,
            num_critical,
            first_cutoff,
            first_stopband
        )
        self.synthesis_input = SynthesisInput(
            w_dim=w_dim,
            channels=int(channels[0]),
            width=int(widths[0]),
            fs=sampling_rates[0],
            bandwidth=cutoffs[0],
            rngs=rngs
        )

        self.synthesis_layers = nnx.data([])
        for i in range(self.num_layers + 1):
            prev = max(i-1, 0)
            is_last_layer = (i == num_layers)
            is_critically_sampled = (i >= num_layers - num_critical)

            _layer = SynthesisLayer(
                w_dim=w_dim,
                in_features=int(channels[prev]),
                out_features=int(channels[i]),
                in_width=int(widths[prev]),
                out_width=int(widths[i]),
                in_fs=sampling_rates[prev],
                out_fs=sampling_rates[i],
                in_cutoff=cutoffs[prev],
                out_cutoff=cutoffs[i],
                in_fh=half_widths[prev],
                out_fh=half_widths[i],
                is_last_layer=is_last_layer,
                is_critically_sampled=is_critically_sampled,
                conv_ksize=conv_ksize if not is_last_layer else last_conv_ksize,
                filter_size=filter_size,
                lrelu_upsampling=lrelu_upsampling,
                conv_clamp=conv_clamp,
                magnitude_ema_beta=magnitude_ema_beta,
                rngs=rngs
            )
            self.synthesis_layers.append(_layer)


    def __call__(self, ws: jax.Array, update_emas: bool = False) -> jax.Array:
        x = self.synthesis_input(ws[:, 0])
        for i in range(self.num_layers + 1):
            x = self.synthesis_layers[i](x, ws[:, i+1], update_emas=update_emas)

        if self.output_scale != 1:
            x *= self.output_scale

        return x

    @staticmethod
    def compute_param_progression(
            width_sec,
            output_fs,
            output_channels,
            channel_base,
            channel_max,
            margin_size,
            last_stopband_rel,
            num_layers,
            num_critical,
            first_cutoff,
            first_stopband,):
        """Progression of network parameters modified from the original StyleGAN3."""
        last_cutoff = output_fs / 2
        last_stopband = last_cutoff * last_stopband_rel
        exponents = np.minimum(np.arange(num_layers + 1) / (num_layers - num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, output_fs))))
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs
        widths = width_sec * sampling_rates + margin_size * 2
        widths[-2:] = width_sec * output_fs
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = output_channels

        return widths, channels, sampling_rates, cutoffs, half_widths


class Generator(nnx.Module):

    def __init__(
            self,
            *,
            width_sec: int,
            output_fs: int,
            output_channels: int,

            # Mapping network default parameters
            z_dim: int = 512,
            w_dim: int = 512,
            num_mapping_layers: int = 6,
            mapping_lr_multiplier: float = 1.0,

            # Synthesis network default parameters
            channel_base: int = int(32768/2**4),
            channel_max: int = 256,
            num_synthesis_layers: int = 14,
            num_critical: int = 2,
            first_cutoff: float = 0.125,
            first_stopband: float = 0.3,
            last_stopband_rel: float = 2**0.3,
            margin_size: int = 10,
            output_scale: float = 1.0,
            conv_ksize: int = 3,
            last_conv_ksize: int = 1,
            filter_size: int = 6,
            lrelu_upsampling: int = 2,
            conv_clamp: float | None = None,
            magnitude_ema_beta: float = 0.9995,
            rngs: nnx.Rngs):
        self.mapping = MappingNetwork(
            z_dim=z_dim,
            w_dim=w_dim,
            n_ws=num_synthesis_layers + 2,  # Input, output, and synthesis layers
            n_layers=num_mapping_layers,
            lr_multiplier=mapping_lr_multiplier,
            rngs=rngs
        )
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            width_sec=width_sec,
            output_fs=output_fs,
            output_channels=output_channels,
            channel_base=channel_base,
            channel_max=channel_max,
            num_layers=num_synthesis_layers,
            num_critical=num_critical,
            first_cutoff=first_cutoff,
            first_stopband=first_stopband,
            last_stopband_rel=last_stopband_rel,
            margin_size=margin_size,
            output_scale=output_scale,
            conv_ksize=conv_ksize,
            last_conv_ksize=last_conv_ksize,
            filter_size=filter_size,
            lrelu_upsampling=lrelu_upsampling,
            conv_clamp=conv_clamp,
            magnitude_ema_beta=magnitude_ema_beta,
            rngs=rngs
        )

    def __call__(self, z: jax.Array, update_emas: bool = False) -> jax.Array:
        ws = self.mapping(z)
        x = self.synthesis(ws, update_emas=update_emas)

        # Return [NWC]
        return jnp.transpose(x, axes=[0, 2, 1])
