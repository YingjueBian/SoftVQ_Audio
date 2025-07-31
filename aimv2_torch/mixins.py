# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from typing import Any, List, Optional, Tuple, Union

import torch

__all__ = ["AIMv2VisionMixin", "AIMv2TextMixin"]


class AIMv2VisionMixin:
    preprocessor: Any
    trunk: Any
    head: Any

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        x = self.preprocessor(x)
        x, features = self.trunk(x, mask=mask)
        x = self.head(x)

        if output_features:
            return x, tuple(features)

        return x


class AIMv2TextMixin:
    preprocessor: Any
    trunk: Any
    head: Any

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        tokens, eos_token_mask = self.preprocessor(input_ids)
        x, features = self.trunk(tokens, mask=mask)
        x = self.head(x, eos_token_mask)

        if output_features:
            return x, tuple(features)

        return x