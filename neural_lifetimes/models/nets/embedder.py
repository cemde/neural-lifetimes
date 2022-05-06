from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neural_lifetimes.utils.data import FeatureDictionaryEncoder


class CombinedEmbedder(nn.Module):
    """
    An embedder for continous and discrete features. Is a nn.Module.

    Args:
        continuous_features (List[str]): list of continuous features
        category_dict (Dict[str, List]): dictionary of discrete features
        embed_dim (int): embedding dimension
        drop_rate (float): dropout rate. Defaults to ``0.0``.
        pre_encoded (bool): whether to use the input data as is. Defaults to ``False``.
    """

    def __init__(
        self,
        feature_encoder: FeatureDictionaryEncoder,
        embed_dim: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.encoder = feature_encoder

        # create the continuous feature encoding, with one hidden layer for good measure
        num_cf = len(self.continuous_features)
        self.c1 = nn.Linear(num_cf, 2 * num_cf)
        self.c2 = nn.Linear(2 * num_cf, embed_dim)

        # create the discrete feature encoding
        self.enc = {}
        self.emb = nn.ModuleDict()
        for name in self.discrete_features:
            # self.encoder.categories_[0].__len__() + 1
            self.emb[name] = nn.Embedding(self.encoder.feature_size(name), embed_dim)

        self.output_shape = [None, embed_dim]

    @property
    def continuous_features(self):
        return self.encoder.continuous_features

    @property
    def discrete_features(self):
        return self.encoder.discrete_features

    def build_parameter_dict(self) -> Dict[str, Any]:
        """Return a dict of parameters.

        Returns:
            Dict[str, Any]: Parameters of the embedder
        """
        return {
            "embed_dim": self.embed_dim,
            "embedder_drop_rate": self.drop_rate,
        }

    def forward(self, x: Dict[str, torch.Tensor]):
        # batch x num_cont_features
        cf = torch.stack([x[f] for f in self.continuous_features], dim=1)
        cf[cf.isnan()] = 0

        out = F.dropout(F.relu(self.c1(cf)), self.drop_rate, self.training)
        # batch x embed_dim
        out = F.dropout(F.relu(self.c2(out)), self.drop_rate, self.training)
        assert not torch.isnan(out.sum())

        return out
