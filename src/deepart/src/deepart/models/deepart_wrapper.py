"""
A module containing a class that wraps around an existing PyTorch MLP/CNN and implements DeepART forward and update rules.
"""

# region DEPENDENCIES

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    # Iterable,
    List,
    Optional,
    Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# endregion


class ComplementCode(nn.Module):
    """
    Complement coding layer.

    Dense input: concat on feature axis (B, F) -> (B, 2F)
    Conv input: concat on channel axis (B, C, H, W) -> (B, 2C, H, W)
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, 1.0 - x], dim=self.dim)


@dataclass
class _LayerCache:
    x: torch.Tensor
    y: torch.Tensor
    z: Optional[torch.Tensor] = None


class DeepARTWrapper(nn.Module):
    """
    Wrap an arbitrary PyTorch model with:
    1) complement coding before Linear/Conv2d layers
    2) local layer-wise updates (fuzzyart, instar, oja)
    3) local updates for Transformer attention blocks (nn.MultiheadAttention)

    Typical use:
        wrapped = DeepARTWrapper(model, learning_rule="oja", eta=1e-2)
        logits = wrapped.learn_step(x, target=y_one_hot)  # optional target
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rule: str = "oja",
        eta: float = 0.01,
        beta_d: float = 1.0,
        beta_rule: str = "wta",
        beta_normalize: bool = True,
        apply_complement_coding: bool = True,
        update_final_with_target: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rule = learning_rule.lower()
        self.eta = float(eta)
        self.beta_d = float(beta_d)
        self.beta_rule = beta_rule.lower()
        self.beta_normalize = bool(beta_normalize)
        self.apply_complement_coding = bool(apply_complement_coding)
        self.update_final_with_target = bool(update_final_with_target)

        self._supported = (nn.Linear, nn.Conv2d, nn.MultiheadAttention)
        self._wrapped_layers: List[nn.Module] = []
        self._layer_cache: Dict[int, _LayerCache] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        if self.apply_complement_coding:
            self._inject_complement_coding(self.model)
        self._register_hooks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def learn_step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._layer_cache.clear()
        y_hat = self.forward(x)
        self.local_update(target=target)
        return y_hat

    @torch.no_grad()
    def local_update(self, target: Optional[torch.Tensor] = None) -> None:
        if not self._wrapped_layers:
            return

        for idx, layer in enumerate(self._wrapped_layers):
            cache = self._layer_cache.get(id(layer))
            if cache is None:
                continue

            is_last = idx == (len(self._wrapped_layers) - 1)
            if (
                is_last
                and target is not None
                and self.update_final_with_target
                and isinstance(layer, nn.Linear)
            ):
                self._widrow_hoff_update(layer, cache.x, cache.y, target)
                continue

            if isinstance(layer, nn.Linear):
                self._update_linear(layer, cache.x, cache.y)
            elif isinstance(layer, nn.Conv2d):
                self._update_conv2d(layer, cache.x, cache.y)
            elif isinstance(layer, nn.MultiheadAttention):
                self._update_mha(layer, cache.x, cache.y, cache.z)

    def decay(self, decay_rate: float = 0.975) -> None:
        self.eta *= float(decay_rate)

    def _register_hooks(self) -> None:
        self._clear_hooks()

        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
            if not inputs:
                return
            if isinstance(module, nn.MultiheadAttention):
                # MHA forward signature is (query, key, value, ...)
                if len(inputs) < 3:
                    return
                query = inputs[0].detach()
                key = inputs[1].detach()
                value = inputs[2].detach()
                attn_out = output[0].detach() if isinstance(output, tuple) else output.detach()
                self._layer_cache[id(module)] = _LayerCache(
                    x=query,
                    y=attn_out,
                    z=torch.cat([key.flatten(0, -2), value.flatten(0, -2)], dim=0),
                )
                return

            self._layer_cache[id(module)] = _LayerCache(x=inputs[0].detach(), y=output.detach())

        for module in self.model.modules():
            if isinstance(module, self._supported):
                self._wrapped_layers.append(module)
                self._hooks.append(module.register_forward_hook(hook_fn))

    def _clear_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._wrapped_layers.clear()

    def _inject_complement_coding(self, root: nn.Module) -> None:
        for name, child in list(root.named_children()):
            if isinstance(child, nn.Linear):
                cc = ComplementCode(dim=-1)
                replaced = self._expand_linear_in_features(child)
                setattr(root, name, nn.Sequential(cc, replaced))
            elif isinstance(child, nn.Conv2d):
                cc = ComplementCode(dim=1)
                replaced = self._expand_conv2d_in_channels(child)
                setattr(root, name, nn.Sequential(cc, replaced))
            elif isinstance(child, nn.MultiheadAttention):
                # MHA cannot be dimension-doubled via external CC because embed_dim is fixed.
                # We keep structure intact and update MHA weights directly in local updates.
                continue
            else:
                self._inject_complement_coding(child)

    def _expand_linear_in_features(self, layer: nn.Linear) -> nn.Linear:
        new_layer = nn.Linear(
            in_features=2 * layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        with torch.no_grad():
            new_layer.weight.zero_()
            new_layer.weight[:, :layer.in_features].copy_(layer.weight)
            if layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.copy_(layer.bias)
        return new_layer

    def _expand_conv2d_in_channels(self, layer: nn.Conv2d) -> nn.Conv2d:
        new_layer = nn.Conv2d(
            in_channels=2 * layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        with torch.no_grad():
            new_layer.weight.zero_()
            new_layer.weight[:, :layer.in_channels].copy_(layer.weight)
            if layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.copy_(layer.bias)
        return new_layer

    def _beta(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(0)

        if self.beta_rule in ("wta", "wta-norm"):
            scale = self.beta_d
            if self.beta_rule == "wta-norm":
                scale = self.beta_d * (y.size(-1) ** 0.5)
            return torch.full((y.size(0), y.size(1)), scale, dtype=y.dtype, device=y.device)

        if self.beta_rule == "softmax":
            local_soft = torch.softmax(y, dim=1)
            max_soft = local_soft.max(dim=1, keepdim=True).values if self.beta_normalize else 1.0
            return self.beta_d * local_soft / max_soft

        if self.beta_rule == "contrast":
            local_soft = -torch.softmax(y, dim=1)
            max_ix = y.argmax(dim=1, keepdim=True)
            local_soft.scatter_(1, max_ix, -local_soft.gather(1, max_ix))
            max_soft = local_soft.abs().max(dim=1, keepdim=True).values if self.beta_normalize else 1.0
            return self.beta_d * local_soft / max_soft

        raise ValueError(f"Unsupported beta_rule: {self.beta_rule}")

    @torch.no_grad()
    def _update_linear(self, layer: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
        if y.ndim > 2:
            y = y.reshape(-1, y.size(-1))

        beta = self._beta(y)
        n = x.size(0)
        w = layer.weight.data

        if self.learning_rule == "instar":
            s = beta * y
            term1 = (s.t() @ x) / n
            term2 = s.mean(dim=0, keepdim=True).t() * w
            dw = self.eta * (term1 - term2)
            w.add_(dw)
            return

        if self.learning_rule == "oja":
            s = beta * y
            term1 = (s.t() @ x) / n
            term2 = (beta * y * y).mean(dim=0, keepdim=True).t() * w
            dw = self.eta * (term1 - term2)
            w.add_(dw)
            return

        if self.learning_rule == "fuzzyart":
            if self.beta_rule in ("wta", "wta-norm"):
                j = y.mean(dim=0).argmax()
                b = beta[0, 0]
                wj = w[j]
                xbar = x.mean(dim=0)
                w[j].copy_(b * torch.minimum(xbar, wj) + (1.0 - b) * wj)
            else:
                xbar = x.mean(dim=0, keepdim=True).expand_as(w)
                b = beta.mean(dim=0, keepdim=True).t().expand_as(w)
                w.copy_(b * torch.minimum(xbar, w) + (1.0 - b) * w)
            return

        raise ValueError(f"Unsupported learning_rule: {self.learning_rule}")

    @torch.no_grad()
    def _update_conv2d(self, layer: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim != 4 or y.ndim != 4:
            return

        w = layer.weight.data
        bsz, _, out_h, out_w = y.shape
        unfolded = F.unfold(
            x,
            kernel_size=layer.kernel_size,
            dilation=layer.dilation,
            padding=layer.padding,
            stride=layer.stride,
        )  # (B, F, L)
        n_windows = unfolded.shape[-1]
        x_flat = unfolded.permute(0, 2, 1).reshape(-1, unfolded.size(1))  # (B*L, F)
        y_flat = y.permute(0, 2, 3, 1).reshape(-1, y.size(1))  # (B*L, O)

        beta = self._beta(y_flat)
        n = float(bsz * n_windows)
        w_flat = w.view(w.size(0), -1)

        if self.learning_rule == "instar":
            s = beta * y_flat
            term1 = (s.t() @ x_flat) / n
            term2 = s.mean(dim=0, keepdim=True).t() * w_flat
            dw = self.eta * (term1 - term2)
            w_flat.add_(dw)
            return

        if self.learning_rule == "oja":
            s = beta * y_flat
            term1 = (s.t() @ x_flat) / n
            term2 = (beta * y_flat * y_flat).mean(dim=0, keepdim=True).t() * w_flat
            dw = self.eta * (term1 - term2)
            w_flat.add_(dw)
            return

        if self.learning_rule == "fuzzyart":
            if self.beta_rule in ("wta", "wta-norm"):
                j = y_flat.mean(dim=0).argmax()
                b = beta[0, 0]
                wj = w_flat[j]
                xbar = x_flat.mean(dim=0)
                w_flat[j].copy_(b * torch.minimum(xbar, wj) + (1.0 - b) * wj)
            else:
                xbar = x_flat.mean(dim=0, keepdim=True).expand_as(w_flat)
                b = beta.mean(dim=0, keepdim=True).t().expand_as(w_flat)
                w_flat.copy_(b * torch.minimum(xbar, w_flat) + (1.0 - b) * w_flat)
            return

        raise ValueError(f"Unsupported learning_rule: {self.learning_rule}")

    @torch.no_grad()
    def _widrow_hoff_update(
        self,
        layer: nn.Linear,
        x: torch.Tensor,
        y: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
        if y.ndim > 2:
            y = y.reshape(-1, y.size(-1))

        if target.ndim == 1:
            target = F.one_hot(target.long(), num_classes=y.size(1)).to(y.dtype)
        target = target.to(device=y.device, dtype=y.dtype)

        err = target - y
        dw = self.eta * (err.t() @ x) / x.size(0)
        layer.weight.data.add_(dw)

    @torch.no_grad()
    def _update_mha(
        self,
        layer: nn.MultiheadAttention,
        query: torch.Tensor,
        attn_out: torch.Tensor,
        kv_flat_cache: Optional[torch.Tensor] = None,
    ) -> None:
        # We update in_proj_weight via local rules using token-level query activations.
        # out_proj is already an nn.Linear child and is updated by _update_linear.
        del attn_out
        del kv_flat_cache

        q_flat = query.flatten(0, -2)
        if q_flat.ndim != 2:
            return

        w_in = layer.in_proj_weight
        if w_in is None:
            return

        proj = F.linear(q_flat, w_in, layer.in_proj_bias)  # (N_tokens, 3 * embed_dim)
        beta = self._beta(proj)
        n = q_flat.size(0)

        if self.learning_rule == "instar":
            s = beta * proj
            term1 = (s.t() @ q_flat) / n
            term2 = s.mean(dim=0, keepdim=True).t() * w_in.data
            w_in.data.add_(self.eta * (term1 - term2))
            return

        if self.learning_rule == "oja":
            s = beta * proj
            term1 = (s.t() @ q_flat) / n
            term2 = (beta * proj * proj).mean(dim=0, keepdim=True).t() * w_in.data
            w_in.data.add_(self.eta * (term1 - term2))
            return

        if self.learning_rule == "fuzzyart":
            if self.beta_rule in ("wta", "wta-norm"):
                j = proj.mean(dim=0).argmax()
                b = beta[0, 0]
                wj = w_in.data[j]
                qbar = q_flat.mean(dim=0)
                w_in.data[j].copy_(b * torch.minimum(qbar, wj) + (1.0 - b) * wj)
            else:
                qbar = q_flat.mean(dim=0, keepdim=True).expand_as(w_in.data)
                b = beta.mean(dim=0, keepdim=True).t().expand_as(w_in.data)
                w_in.data.copy_(b * torch.minimum(qbar, w_in.data) + (1.0 - b) * w_in.data)
            return

        raise ValueError(f"Unsupported learning_rule: {self.learning_rule}")

    def __del__(self):
        self._clear_hooks()
