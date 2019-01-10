import torch.nn as nn

norm_cfg = {'BN': nn.BatchNorm2d, 'SyncBN': None, 'GN': None}


def build_norm_layer(cfg, num_features):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    cfg_.setdefault('eps', 1e-5)
    layer_type = cfg_.pop('type')

    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    elif norm_cfg[layer_type] is None:
        raise NotImplementedError

    return norm_cfg[layer_type](num_features, **cfg_)


def l2_normalize(x, axis=None, epsilon=1e-12, dim=None):
    # type: (Tensor, float, int, float, Optional[Tensor]) -> Tensor
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
        out (Tensor, optional): the output tensor. If :attr:`out` is used, this
                                operation won't be differentiable.
    """
    if out is None:
        denom = input.norm(2, dim, True).clamp(min=eps).expand_as(input)
        ret = input / denom
    else:
        denom = input.norm(2, dim, True).clamp_(min=eps).expand_as(input)
        ret = torch.div(input, denom, out=torch.jit._unwrap_optional(out))
    return ret
