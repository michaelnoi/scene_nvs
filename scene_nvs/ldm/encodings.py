# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://github.com/nerfstudio-project/nerfstudio/blob/4c966b9bd4bab5a4089e5ca767b4a6af9b3e1bf6/nerfstudio/field_components/encodings.py#L39

# mypy: ignore-errors
# flake8: noqa: F722

from abc import abstractmethod
from typing import Literal, Optional

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn


class FieldComponent(nn.Module):
    # from nerfstudio.field_components.base_field_component import FieldComponent

    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Output dimension to module.
    """

    def __init__(
        self, in_dim: Optional[int] = None, out_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(
        self, in_tensor: Shaped[Tensor, "*bs input_dim"]
    ) -> Shaped[Tensor, "*bs output_dim"]:
        """
        Returns processed tensor

        Args:
            in_tensor: Input tensor to process
        """
        raise NotImplementedError


def expected_sin(x_means: torch.Tensor, x_vars: torch.Tensor) -> torch.Tensor:
    # from nerfstudio.utils.math expected_sin

    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """
    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @classmethod
    def get_tcnn_encoding_config(cls) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        raise NotImplementedError("Encoding does not have a TCNN implementation")

    @abstractmethod
    def forward(
        self, in_tensor: Shaped[Tensor, "*bs input_dim"]
    ) -> Shaped[Tensor, "*bs output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(
        self, in_tensor: Shaped[Tensor, "*bs input_dim"]
    ) -> Shaped[Tensor, "*bs output_dim"]:
        return in_tensor


class ScalingAndOffset(Encoding):
    """Simple scaling and offset to input

    Args:
        in_dim: Input dimension of tensor
        scaling: Scaling applied to tensor.
        offset: Offset applied to tensor.
    """

    def __init__(self, in_dim: int, scaling: float = 1.0, offset: float = 0.0) -> None:
        super().__init__(in_dim)

        self.scaling = scaling
        self.offset = offset

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(
        self, in_tensor: Float[Tensor, "*bs input_dim"]
    ) -> Float[Tensor, "*bs output_dim"]:
        return self.scaling * in_tensor + self.offset


class NeRFEncoding(Encoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        implementation: Literal["torch"] = "torch",
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def pytorch_fwd(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(
            self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device
        )
        scaled_inputs = (
            scaled_in_tensor[..., None] * freqs
        )  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(
            *scaled_inputs.shape[:-2], -1
        )  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
            )
        else:
            input_var = (
                torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None]
                * freqs[None, :] ** 2
            )
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1),
                torch.cat(2 * [input_var], dim=-1),
            )
        return encoded_inputs

    def forward(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        encoded_inputs = self.pytorch_fwd(in_tensor, covs)
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs
