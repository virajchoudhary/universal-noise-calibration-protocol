"""Noise generators — the probe arsenal for the NSA diagnostic.

Each generator implements ``apply(x, magnitude) -> x_noisy`` where
``magnitude ∈ [0, 1]``. All vision generators operate on batched tensors of
shape (B, C, H, W) in the [0, 1] range and return fresh tensors
(no in-place mutation).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class NoiseGenerator:
    """Abstract base class."""

    def __init__(self, noise_type: str, domain: str) -> None:
        self.noise_type = noise_type
        self.domain = domain

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        raise NotImplementedError

    def get_name(self) -> str:
        return self.noise_type

    @staticmethod
    def _check_magnitude(m: float) -> None:
        if not 0.0 <= m <= 1.0:
            raise ValueError(f"magnitude must be in [0, 1], got {m}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(domain={self.domain})"


# -----------------------------------------------------------------------------
# Vision
# -----------------------------------------------------------------------------

class GaussianNoise(NoiseGenerator):
    """Isotropic Gaussian noise — targets texture / high-frequency shortcuts."""

    def __init__(self) -> None:
        super().__init__("gaussian", "vision")

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        noise = torch.randn_like(x) * float(magnitude)
        return torch.clamp(x + noise, 0.0, 1.0)


class SpatialMasking(NoiseGenerator):
    """Random square patch occlusion — targets local spatial shortcuts."""

    def __init__(self) -> None:
        super().__init__("spatial_masking", "vision")

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        x = x.clone()
        b, c, h, w = x.shape
        patch = max(1, int(magnitude * min(h, w)))
        if patch == 0:
            return x
        for i in range(b):
            top = torch.randint(0, max(1, h - patch + 1), (1,)).item()
            left = torch.randint(0, max(1, w - patch + 1), (1,)).item()
            x[i, :, top:top + patch, left:left + patch] = 0.0
        return x


class FrequencyFilter(NoiseGenerator):
    """FFT low-pass filter — zeros out high frequencies. Targets texture/color
    modulation while preserving coarse shape."""

    def __init__(self) -> None:
        super().__init__("frequency_filter", "vision")

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        b, c, h, w = x.shape
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device) - h / 2.0,
            torch.arange(w, device=x.device) - w / 2.0,
            indexing="ij",
        )
        radius = torch.sqrt(yy ** 2 + xx ** 2)
        max_r = math.sqrt((h / 2) ** 2 + (w / 2) ** 2)
        # magnitude = 0 → keep all (cutoff = max_r); magnitude = 1 → block all (cutoff = 0)
        cutoff = max_r * (1.0 - float(magnitude))
        mask = (radius <= cutoff).float()[None, None]
        fft_filtered = fft * mask
        fft_filtered = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        out = torch.fft.ifft2(fft_filtered).real
        return torch.clamp(out, 0.0, 1.0)


class GaussianBlur(NoiseGenerator):
    """Gaussian blur — targets fine-detail shape cues; preserves coarse color."""

    def __init__(self) -> None:
        super().__init__("gaussian_blur", "vision")

    @staticmethod
    def _kernel(sigma: float, device: torch.device) -> torch.Tensor:
        ks = max(3, int(2 * round(2 * sigma) + 1))
        ax = torch.arange(ks, device=device) - (ks - 1) / 2.0
        g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k2d = g[:, None] * g[None, :]
        return k2d

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        if magnitude < 1e-6:
            return x.clone()
        sigma = max(0.1, 5.0 * float(magnitude))
        k = self._kernel(sigma, x.device)
        ks = k.shape[0]
        k = k.expand(x.shape[1], 1, ks, ks)
        padded = F.pad(x, [ks // 2] * 4, mode="reflect")
        out = F.conv2d(padded, k, groups=x.shape[1])
        return torch.clamp(out, 0.0, 1.0)


class ColorJitter(NoiseGenerator):
    """Random brightness/contrast/saturation/hue — directly perturbs color."""

    def __init__(self) -> None:
        super().__init__("color_jitter", "vision")

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        out = x.clone()
        b = out.size(0)
        m = float(magnitude)
        # brightness
        bright = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * m
        out = out * bright
        # contrast
        contr = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * m
        mean = out.mean(dim=(2, 3), keepdim=True)
        out = (out - mean) * contr + mean
        # saturation (shift channels around the luminance)
        lum = out.mean(dim=1, keepdim=True)
        sat = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * m
        out = (out - lum) * sat + lum
        # hue (channel permutation with bernoulli-weighted blending)
        perm = out[:, torch.tensor([1, 2, 0], device=x.device), :, :]
        hue_mix = (torch.rand(b, 1, 1, 1, device=x.device)) * m
        out = out * (1.0 - hue_mix) + perm * hue_mix
        return torch.clamp(out, 0.0, 1.0)


class PatchShuffle(NoiseGenerator):
    """Grid-patch shuffle — destroys global spatial layout, keeps texture."""

    def __init__(self) -> None:
        super().__init__("patch_shuffle", "vision")

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        b, c, h, w = x.shape
        # magnitude controls grid fineness: 0.1 → 2, 1.0 → h (pixel shuffle)
        grid = max(1, int(1 + magnitude * (min(h, w) // 2 - 1)))
        if grid <= 1:
            return x.clone()
        patches = x.unfold(2, h // grid, h // grid).unfold(3, w // grid, w // grid)
        # patches: (B, C, grid, grid, ph, pw)
        bb, cc, gh, gw, ph, pw = patches.shape
        flat = patches.contiguous().view(bb, cc, gh * gw, ph, pw)
        out = torch.empty_like(flat)
        for i in range(bb):
            perm = torch.randperm(gh * gw, device=x.device)
            out[i] = flat[i, :, perm]
        out = out.view(bb, cc, gh, gw, ph, pw)
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous().view(bb, cc, gh * ph, gw * pw)
        if out.shape[-2:] != (h, w):
            out = F.pad(out, [0, w - out.size(-1), 0, h - out.size(-2)])
        return torch.clamp(out, 0.0, 1.0)


# -----------------------------------------------------------------------------
# NLP
# -----------------------------------------------------------------------------

class EmbeddingGaussian(NoiseGenerator):
    """Isotropic Gaussian noise in embedding space (BERT hooks).

    Applied via forward-hook in the NLP pipeline; here we provide a direct
    ``apply`` that operates on token embeddings if passed.
    """

    def __init__(self) -> None:
        super().__init__("embedding_gaussian", "nlp")

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        return x + torch.randn_like(x) * float(magnitude)


class TokenMasking(NoiseGenerator):
    """Random replacement of tokens with the BERT [MASK] token.

    ``apply`` expects a dict with ``input_ids``, ``attention_mask`` and
    optionally ``special_tokens_mask``; returns a new dict with
    ``input_ids`` masked at ``magnitude`` probability per non-special token.
    """

    def __init__(self, mask_token_id: int = 103) -> None:
        super().__init__("token_masking", "nlp")
        self.mask_token_id = mask_token_id

    def apply(self, inputs, magnitude: float):  # type: ignore[override]
        self._check_magnitude(magnitude)
        if isinstance(inputs, torch.Tensor):
            ids = inputs.clone()
            mask = torch.rand_like(ids.float()) < magnitude
            ids[mask] = self.mask_token_id
            return ids
        out = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
        special = out.get("special_tokens_mask", torch.zeros_like(out["input_ids"]))
        attn = out.get("attention_mask", torch.ones_like(out["input_ids"]))
        rand = torch.rand_like(out["input_ids"].float())
        mask = (rand < magnitude) & (attn == 1) & (special == 0)
        out["input_ids"][mask] = self.mask_token_id
        return out


class SynonymSubstitution(NoiseGenerator):
    """Synonym substitution using WordNet — magnitude controls swap probability.

    This is a placeholder lexical-perturbation generator; in practice it is
    applied at text-preprocess time. For NSA compatibility we provide a
    simple ``apply`` that randomly permutes a fraction of non-stop-word
    tokens within the sequence.
    """

    def __init__(self) -> None:
        super().__init__("synonym_substitution", "nlp")

    def apply(self, input_ids: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        out = input_ids.clone()
        b, n = out.shape
        for i in range(b):
            k = int(n * float(magnitude))
            if k < 2:
                continue
            idx = torch.randperm(n)[:k]
            out[i, idx] = out[i, idx[torch.randperm(k)]]
        return out


# -----------------------------------------------------------------------------
# Tabular
# -----------------------------------------------------------------------------

class FeaturePermutation(NoiseGenerator):
    """Permute values within a random subset of feature columns.

    ``magnitude`` controls the *fraction* of columns to permute.
    """

    def __init__(self) -> None:
        super().__init__("feature_permutation", "tabular")

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        out = x.clone()
        b, d = out.shape
        k = int(d * float(magnitude))
        if k == 0:
            return out
        cols = torch.randperm(d)[:k]
        for c in cols:
            out[:, c] = out[torch.randperm(b), c]
        return out


class ContinuousGaussian(NoiseGenerator):
    """Gaussian noise scaled by per-feature std — continuous features only."""

    def __init__(self, continuous_mask: Optional[torch.Tensor] = None) -> None:
        super().__init__("continuous_gaussian", "tabular")
        self.continuous_mask = continuous_mask

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        std = x.std(dim=0, keepdim=True) + 1e-6
        noise = torch.randn_like(x) * std * float(magnitude)
        if self.continuous_mask is not None:
            noise = noise * self.continuous_mask.to(x.device)
        return x + noise


class CategorySwap(NoiseGenerator):
    """Swap values of categorical features (columns with few unique values)."""

    def __init__(self, categorical_cols: Optional[List[int]] = None) -> None:
        super().__init__("category_swap", "tabular")
        self.categorical_cols = categorical_cols

    def apply(self, x: torch.Tensor, magnitude: float) -> torch.Tensor:
        self._check_magnitude(magnitude)
        out = x.clone()
        cols = self.categorical_cols
        if cols is None:
            cols = [c for c in range(out.shape[1])
                    if out[:, c].unique().numel() <= 10]
        for c in cols:
            swap = torch.rand(out.shape[0], device=x.device) < float(magnitude)
            if swap.any():
                idx = swap.nonzero(as_tuple=True)[0]
                perm = idx[torch.randperm(idx.size(0))]
                out[idx, c] = out[perm, c]
        return out


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

VISION_GENERATORS = {
    "gaussian": GaussianNoise,
    "spatial_masking": SpatialMasking,
    "frequency_filter": FrequencyFilter,
    "gaussian_blur": GaussianBlur,
    "color_jitter": ColorJitter,
    "patch_shuffle": PatchShuffle,
}

NLP_GENERATORS = {
    "embedding_gaussian": EmbeddingGaussian,
    "token_masking": TokenMasking,
    "synonym_substitution": SynonymSubstitution,
}

TABULAR_GENERATORS = {
    "feature_permutation": FeaturePermutation,
    "continuous_gaussian": ContinuousGaussian,
    "category_swap": CategorySwap,
}


def get_noise_generators(
    domain: str, noise_types: Optional[List[str]] = None
) -> Dict[str, NoiseGenerator]:
    """Factory: return ``{name: NoiseGenerator}`` for a domain."""
    if domain == "vision":
        pool = VISION_GENERATORS
    elif domain == "nlp":
        pool = NLP_GENERATORS
    elif domain == "tabular":
        pool = TABULAR_GENERATORS
    else:
        raise ValueError(f"Unknown domain: {domain}")
    if noise_types is None:
        noise_types = list(pool.keys())
    return {nt: pool[nt]() for nt in noise_types if nt in pool}
