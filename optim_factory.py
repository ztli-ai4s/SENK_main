"""
Optimizer Factory w/ Custom Weight Decay
Modified from timm

Use parameter name to remove weight decay since
tensor product weight is always one-dimensional.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Optional import of timm optimizers to avoid ImportError caused by timm version differences
try:
    from timm.optim.adafactor import Adafactor
except Exception:
    Adafactor = None
try:
    from timm.optim.adahessian import Adahessian
except Exception:
    Adahessian = None
try:
    from timm.optim.adamp import AdamP
except Exception:
    AdamP = None
try:
    from timm.optim.lookahead import Lookahead
except Exception:
    Lookahead = None
# Nadam: compat import handled below
try:
    from timm.optim.nadam import Nadam as TimmNadam
except Exception:
    TimmNadam = None
try:
    from timm.optim.novograd import NovoGrad
except Exception:
    NovoGrad = None
try:
    from timm.optim.nvnovograd import NvNovoGrad
except Exception:
    NvNovoGrad = None
try:
    from timm.optim.radam import RAdam as TimmRAdam
except Exception:
    TimmRAdam = None
try:
    from timm.optim.rmsprop_tf import RMSpropTF
except Exception:
    RMSpropTF = None
try:
    from timm.optim.sgdp import SGDP
except Exception:
    SGDP = None
try:
    from timm.optim.adabelief import AdaBelief
except Exception:
    AdaBelief = None


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (name.endswith(".bias") or name.endswith(".affine_weight")  
            or name.endswith(".affine_bias") or name.endswith('.mean_shift')
            or 'bias.' in name 
            or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        optimizer_name=cfg.opt,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """Convenience wrapper around create_optimizer_v2 for argparse-style calls."""
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_v2(
        model: nn.Module,
        optimizer_name: str = 'sgd',
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        **kwargs):
    """ Create an optimizer.

    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model (nn.Module): model containing parameters to optimize
        optimizer_name: name of optimizer to create
        learning_rate: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    opt_lower = optimizer_name.lower()
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    #if 'fused' in opt_lower:
    #    assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=learning_rate, weight_decay=weight_decay, **kwargs)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args) 
    elif opt_lower == 'adabelief':
        if AdaBelief is not None:
            optimizer = AdaBelief(parameters, rectify=False, **opt_args)
        else:
            optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        if TimmNadam is not None:
            optimizer = TimmNadam(parameters, **opt_args)
        elif hasattr(optim, 'NAdam'):
            optimizer = optim.NAdam(parameters, **opt_args)
        else:
            optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'radam':
        if TimmRAdam is not None:
            optimizer = TimmRAdam(parameters, **opt_args)
        elif hasattr(optim, 'RAdam'):
            optimizer = optim.RAdam(parameters, **opt_args)
        else:
            optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        if AdamP is not None:
            optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
        else:
            optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'sgdp':
        if SGDP is not None:
            optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)
        else:
            optimizer = optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if Adafactor is not None:
            if not learning_rate:
                opt_args['lr'] = None
            optimizer = Adafactor(parameters, **opt_args)
        else:
            optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        if Adahessian is not None:
            optimizer = Adahessian(parameters, **opt_args)
        else:
            optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        if RMSpropTF is not None:
            optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)
        else:
            optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'novograd':
        if NovoGrad is not None:
            optimizer = NovoGrad(parameters, **opt_args)
        else:
            optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        if NvNovoGrad is not None:
            optimizer = NvNovoGrad(parameters, **opt_args)
        else:
            optimizer = optim.Adam(parameters, **opt_args)
    #elif opt_lower == 'fusedsgd':
    #    opt_args.pop('eps', None)
    #    optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    #elif opt_lower == 'fusedmomentum':
    #    opt_args.pop('eps', None)
    #    optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    #elif opt_lower == 'fusedadam':
    #    optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    #elif opt_lower == 'fusedadamw':
    #    optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    #elif opt_lower == 'fusedlamb':
    #    optimizer = FusedLAMB(parameters, **opt_args)
    #elif opt_lower == 'fusednovograd':
    #    opt_args.setdefault('betas', (0.95, 0.98))
    #    optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead' and Lookahead is not None:
            optimizer = Lookahead(optimizer)

    return optimizer 