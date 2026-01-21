"""CADET model implementation.

This module implements the CADET VAE model with disentangled latent representations
for cross-style generalization in hate speech detection.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BartForConditionalGeneration, RobertaModel


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training.

    This implements the gradient reversal operation used in domain adaptation.
    Forward pass is identity, but backward pass reverses and scales the gradient.
    """

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        """Forward pass (identity).

        Args:
            ctx: Context object to save alpha
            x: Input tensor
            alpha: Reversal strength

        Returns:
            Input tensor (unchanged)
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        """Backward pass (reverse and scale gradient).

        Args:
            ctx: Context object with saved alpha
            grad_output: Gradient from next layer

        Returns:
            Tuple of (reversed_gradient, None) - None for alpha gradient
        """
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for Gradient Reversal Function.

    This module applies gradient reversal during backpropagation,
    which is essential for adversarial training to remove confounders.
    """

    def __init__(self, alpha: float = 1.0):
        """Initialize GRL with reversal strength.

        Args:
            alpha: Gradient reversal strength (typically 0.1 to 1.0)
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        """Apply gradient reversal.

        Args:
            x: Input tensor

        Returns:
            Output tensor (same as input in forward pass)
        """
        return GradientReversal.apply(x, self.alpha)  # type: ignore


class CADET(nn.Module):
    """CADET model.

    Renamed from CausalVAE per change notes.

    Architecture:
    - Encoder: RoBERTa → hidden representation (from checkpoint)
    - Latents: M (motivation), T (target), style (explicit/implicit), U (confounder)
    - Decoder: BART reconstruction (from checkpoint)
    - Classifiers: Hate (from M), Target (from T), Style (explicit/implicit)
    """

    def __init__(
        self,
        n_targets: int,
        encoder_checkpoint: str,
        decoder_checkpoint: str,
        style_dim: int = 2,
        conf_dim: int = 256,
        orth_dim: int = 128,
        use_confounder_for_prediction: bool = False,
    ):
        """Initialize CADET model.

        Args:
            n_targets: Number of demographic target groups
            encoder_checkpoint: Checkpoint path for RoBERTa encoder
            decoder_checkpoint: Checkpoint path for BART decoder
            style_dim: Style latent dimension (2 for explicit/implicit)
            conf_dim: Confounder latent dimension
            orth_dim: Orthogonality projection dimension
            use_confounder_for_prediction: If True, predict hate from M + U (in-distribution),
                                           if False, predict from M only (cross-style, default)
        """
        super().__init__()

        self.n_targets = n_targets
        self.style_dim = style_dim
        self.conf_dim = conf_dim
        self.orth_dim = orth_dim
        self.use_confounder_for_prediction = use_confounder_for_prediction

        # Encoder (load from checkpoint)
        self.encoder = RobertaModel.from_pretrained(encoder_checkpoint)
        enc_hidden = self.encoder.config.hidden_size  # 768

        # Confounder (U) - continuous
        self.fc_mu_u = nn.Linear(enc_hidden, conf_dim)
        self.fc_logvar_u = nn.Linear(enc_hidden, conf_dim)

        # Motivation (M) - continuous (raw, before purification)
        self.fc_mu_m_raw = nn.Linear(enc_hidden, enc_hidden)
        self.fc_logvar_m_raw = nn.Linear(enc_hidden, enc_hidden)

        # Target (T) - discrete (raw, before purification)
        self.fc_target_logits_raw = nn.Linear(enc_hidden, n_targets)

        # Style (explicit/implicit) - discrete (raw, before purification)
        self.fc_style_logits_raw = nn.Linear(enc_hidden, style_dim)

        # Purification networks (remove confounder influence)
        self.purify_m = nn.Sequential(
            nn.Linear(enc_hidden + conf_dim, enc_hidden),
            nn.LayerNorm(enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden),
        )

        self.purify_t = nn.Sequential(
            nn.Linear(n_targets + conf_dim, n_targets),
            nn.LayerNorm(n_targets),
            nn.ReLU(),
            nn.Linear(n_targets, n_targets),
        )

        self.purify_style = nn.Sequential(
            nn.Linear(style_dim + conf_dim, style_dim),
            nn.LayerNorm(style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
        )

        # Gradient reversal for adversarial training
        self.gradient_reversal = GradientReversalLayer(alpha=1.0)

        # Adversarial networks (predict U from purified latents)
        self.adv_m_to_u = nn.Sequential(
            nn.Linear(enc_hidden, conf_dim), nn.ReLU(), nn.Linear(conf_dim, conf_dim)
        )

        self.adv_t_to_u = nn.Sequential(
            nn.Linear(n_targets, conf_dim), nn.ReLU(), nn.Linear(conf_dim, conf_dim)
        )

        self.adv_style_to_u = nn.Sequential(
            nn.Linear(style_dim, conf_dim), nn.ReLU(), nn.Linear(conf_dim, conf_dim)
        )

        # Decoder (load from checkpoint)
        self.decoder = BartForConditionalGeneration.from_pretrained(
            decoder_checkpoint, num_labels=2
        )
        bart_hidden = self.decoder.config.d_model  # 768

        # Projections to BART hidden space
        self.proj_m = nn.Linear(enc_hidden, bart_hidden)
        self.proj_u = nn.Linear(conf_dim, bart_hidden)

        # Reduce target and style dimensions first
        self.proj_t = nn.Sequential(
            nn.Linear(n_targets, bart_hidden // 2),
            nn.ReLU(),
            nn.Linear(bart_hidden // 2, bart_hidden),
        )

        self.proj_style = nn.Linear(style_dim, bart_hidden)

        # Classification heads (predict from purified latents)
        # Hate classifier dimension depends on variant
        hate_input_dim = enc_hidden
        if use_confounder_for_prediction:
            hate_input_dim += conf_dim  # M* + U -> Y (version 2)
        self.cls_hate = nn.Linear(hate_input_dim, 2)  # Binary hate classification
        self.cls_tgt = nn.Linear(n_targets, n_targets)  # Target classification from T (identity)
        self.cls_style = nn.Linear(
            style_dim, style_dim
        )  # Style classification from style (identity)

        # Orthogonality projections (for disentanglement)
        self.orth_proj_m = nn.Sequential(
            nn.Linear(enc_hidden, orth_dim), nn.ReLU(), nn.Linear(orth_dim, orth_dim)
        )

        self.orth_proj_t = nn.Sequential(
            nn.Linear(n_targets, orth_dim), nn.ReLU(), nn.Linear(orth_dim, orth_dim)
        )

        self.orth_proj_style = nn.Sequential(
            nn.Linear(style_dim, orth_dim), nn.ReLU(), nn.Linear(orth_dim, orth_dim)
        )

        self.orth_proj_u = nn.Sequential(
            nn.Linear(conf_dim, orth_dim), nn.ReLU(), nn.Linear(orth_dim, orth_dim)
        )

    def forward(
        self,
        enc_ids: Tensor,
        enc_mask: Tensor,
        dec_ids: Tensor,
        dec_mask: Tensor,
        hate_labels: Optional[Tensor] = None,
        style_labels: Optional[Tensor] = None,  # Renamed from style_labels
        tgt_labels: Optional[Tensor] = None,
        tgt_conf: Optional[Tensor] = None,
        style_tau: float = 0.5,  # Renamed from style_tau
        target_tau: float = 0.5,
        flip_style: bool = False,  # Renamed from flip_style
        target_override: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Forward pass with confounder mitigation.

        Args:
            enc_ids: RoBERTa input IDs (B, seq_len)
            enc_mask: RoBERTa attention mask
            dec_ids: BART decoder input IDs
            dec_mask: BART decoder attention mask
            hate_labels: Binary hate labels
            style_labels: Style labels (0=implicit, 1=explicit)
            tgt_labels: Target group labels
            tgt_conf: Target confidence scores
            style_tau: Gumbel-Softmax temperature for style
            target_tau: Gumbel-Softmax temperature for target
            flip_style: Whether to flip style (for counterfactuals)
            target_override: Override target (for counterfactuals)

        Returns:
            Dictionary with:
            - rec_loss: Reconstruction loss
            - hate_logits: Hate classification logits
            - tgt_logits: Target classification logits
            - style_logits_pred: Style classification logits
            - KL_m, KL_u, KL_t, KL_style: KL divergence terms
            - adv_loss: Adversarial loss
            - o_mt, o_m_style, o_t_style, o_mu, o_tu, o_style_u: Orthogonality terms
            - zm, zu, zt_onehot, z_style_onehot: Latent representations
            - h0: Encoder hidden state
        """
        # 1. Encode input
        encoder_outputs = self.encoder(input_ids=enc_ids, attention_mask=enc_mask)
        h0 = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] token (B, 768)

        # 2. Infer confounder U (continuous)
        mu_u = self.fc_mu_u(h0)
        logvar_u = self.fc_logvar_u(h0)
        zu = self._sample_gaussian(mu_u, logvar_u)

        # 3. Infer raw latents (before purification)
        mu_m_raw = self.fc_mu_m_raw(h0)
        logvar_m_raw = self.fc_logvar_m_raw(h0)
        zm_raw = self._sample_gaussian(mu_m_raw, logvar_m_raw)

        target_logits_raw = self.fc_target_logits_raw(h0)
        style_logits_raw = self.fc_style_logits_raw(h0)

        # 4. Purify latents (remove confounder influence)
        zm = self.purify_m(torch.cat([zm_raw, zu], dim=-1))

        # For discrete variables, purify the logits
        target_logits_purified = self.purify_t(torch.cat([target_logits_raw, zu], dim=-1))
        style_logits_purified = self.purify_style(torch.cat([style_logits_raw, zu], dim=-1))

        # 5. Sample discrete latents with Gumbel-Softmax
        if target_override is not None:
            # Use provided target (for counterfactuals)
            zt_onehot = target_override
        else:
            zt_onehot = self._sample_gumbel_softmax(target_logits_purified, target_tau)

        if flip_style and style_labels is not None:
            # Flip style (for counterfactuals)
            z_style_onehot = 1 - F.one_hot(style_labels, num_classes=self.style_dim).float()
        else:
            z_style_onehot = self._sample_gumbel_softmax(style_logits_purified, style_tau)

        # 6. Compute KL divergences
        # KL for continuous latents (properly normalized: mean over batch, sum over dimensions)
        KL_m = -0.5 * torch.mean(
            torch.sum(1 + logvar_m_raw - mu_m_raw.pow(2) - logvar_m_raw.exp(), dim=-1)
        )
        KL_u = -0.5 * torch.mean(torch.sum(1 + logvar_u - mu_u.pow(2) - logvar_u.exp(), dim=-1))

        # KL for discrete variables (KL(q||uniform) regularization) - already properly normalized
        target_probs = F.softmax(target_logits_purified, dim=-1)
        n_targets = target_probs.size(-1)
        KL_t = torch.sum(
            target_probs * (torch.log(target_probs + 1e-10) - math.log(1.0 / n_targets)), dim=-1
        ).mean()

        style_probs = F.softmax(style_logits_purified, dim=-1)
        n_styles = style_probs.size(-1)
        KL_style = torch.sum(
            style_probs * (torch.log(style_probs + 1e-10) - math.log(1.0 / n_styles)), dim=-1
        ).mean()

        # 7. Classification from purified latents
        # Version 1 (default): M* -> Y (cross-style transfer)
        # Version 2 (optional): M* + U -> Y (in-distribution accuracy)
        if self.use_confounder_for_prediction:
            hate_input = torch.cat([zm, zu], dim=-1)
        else:
            hate_input = zm
        hate_logits = self.cls_hate(hate_input)
        tgt_logits = self.cls_tgt(zt_onehot)
        style_logits_pred = self.cls_style(z_style_onehot)

        # 8. Adversarial training (predict U from purified latents)
        zm_reversed = self.gradient_reversal(zm)
        zt_reversed = self.gradient_reversal(zt_onehot)
        z_style_reversed = self.gradient_reversal(z_style_onehot)

        u_from_m = self.adv_m_to_u(zm_reversed)
        u_from_t = self.adv_t_to_u(zt_reversed)
        u_from_style = self.adv_style_to_u(z_style_reversed)

        # Adversarial loss (MSE to true U)
        adv_loss = (
            F.mse_loss(u_from_m, zu.detach())
            + F.mse_loss(u_from_t, zu.detach())
            + F.mse_loss(u_from_style, zu.detach())
        ) / 3.0

        # 9. Compute orthogonality constraints
        (
            o_mt,
            o_m_style,
            o_t_style,
            o_mu,
            o_tu,
            o_style_u,
        ) = self.compute_orthogonality(zm, zt_onehot, z_style_onehot, zu)

        # 10. Decode (reconstruct text)
        # Project latents to BART hidden space and sum
        z_combined = (
            self.proj_m(zm)
            + self.proj_u(zu)
            + self.proj_t(zt_onehot)
            + self.proj_style(z_style_onehot)
        )

        # Expand to sequence length
        z_combined = z_combined.unsqueeze(1).expand(
            -1, enc_ids.size(1), -1
        )  # (B, seq_len, bart_hidden)

        # BART decoder
        decoder_outputs = self.decoder(
            encoder_outputs=(z_combined, None, None),  # Custom encoder outputs
            decoder_input_ids=dec_ids,
            decoder_attention_mask=dec_mask,
            labels=dec_ids,  # Teacher forcing
            return_dict=True,
        )

        rec_loss = decoder_outputs.loss

        # Return all outputs
        return {
            "rec_loss": rec_loss,
            "hate_logits": hate_logits,
            "tgt_logits": tgt_logits,
            "style_logits_pred": style_logits_pred,
            "KL_m": KL_m,
            "KL_u": KL_u,
            "KL_t": KL_t,
            "KL_style": KL_style,
            "adv_loss": adv_loss,
            "o_mt": o_mt,
            "o_m_style": o_m_style,
            "o_t_style": o_t_style,
            "o_mu": o_mu,
            "o_tu": o_tu,
            "o_style_u": o_style_u,
            "zm": zm,
            "zu": zu,
            "zt_onehot": zt_onehot,
            "z_style_onehot": z_style_onehot,
            "h0": h0,
        }

    def _sample_gaussian(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample from Gaussian using reparameterization trick.

        Args:
            mu: Mean tensor
            logvar: Log variance tensor

        Returns:
            Sampled tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _sample_gumbel_softmax(
        self, logits: Tensor, tau: float = 1.0, hard: bool = False
    ) -> Tensor:
        """Sample from Gumbel-Softmax distribution.

        Args:
            logits: Logits tensor
            tau: Temperature parameter
            hard: Whether to use hard (discrete) samples

        Returns:
            Sampled one-hot tensor (soft or hard)
        """
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

    def compute_orthogonality(
        self, m: Tensor, t: Tensor, s: Tensor, u: Tensor
    ) -> tuple[Tensor, ...]:
        """Compute pairwise orthogonality constraints.

        Projects latents to common dimension and computes cosine similarity.
        Goal: Minimize similarity to enforce disentanglement.

        Args:
            m: Motivation latent (B, enc_hidden)
            t: Target latent (B, n_targets)
            s: Style latent (B, style_dim)
            u: Confounder latent (B, conf_dim)

        Returns:
            Tuple of (o_mt, o_ms, o_ts, o_mu, o_tu, o_su)
            Each is the mean absolute cosine similarity
        """
        # Project to common orthogonality space
        m_proj = self.orth_proj_m(m)
        t_proj = self.orth_proj_t(t)
        s_proj = self.orth_proj_style(s)
        u_proj = self.orth_proj_u(u)

        # Normalize
        m_n = F.normalize(m_proj, dim=1)
        t_n = F.normalize(t_proj, dim=1)
        s_n = F.normalize(s_proj, dim=1)
        u_n = F.normalize(u_proj, dim=1)

        # Compute pairwise orthogonality - use squared inner product
        o_mt = torch.mean((m_n * t_n).sum(dim=1).pow(2))
        o_ms = torch.mean((m_n * s_n).sum(dim=1).pow(2))
        o_ts = torch.mean((t_n * s_n).sum(dim=1).pow(2))

        # Orthogonality with confounder - we want to maximize these
        # by minimizing the inner product (ensuring independence)
        o_mu = torch.mean((m_n * u_n).sum(dim=1).pow(2))
        o_tu = torch.mean((t_n * u_n).sum(dim=1).pow(2))
        o_su = torch.mean((s_n * u_n).sum(dim=1).pow(2))

        return o_mt, o_ms, o_ts, o_mu, o_tu, o_su

    def get_latent_counterfactual(
        self,
        enc_ids: Tensor,
        enc_mask: Tensor,
        flip_style: bool = False,  # Renamed from flip_style
        style_labels: Optional[Tensor] = None,  # Renamed from style_labels
        target_idx: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """Generate latent counterfactuals for analysis.

        Args:
            enc_ids: Input token IDs
            enc_mask: Attention mask
            flip_style: Whether to flip style
            style_labels: Original style labels
            target_idx: Target index to override

        Returns:
            Dictionary with:
            - zm: Purified motivation
            - zu: Confounder
            - zt_onehot: Target one-hot
            - z_style_onehot: Style one-hot
            - hate_logits: Predicted hate logits
        """
        # Create dummy decoder inputs (not used for counterfactual generation)
        batch_size = enc_ids.size(0)
        dummy_dec_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=enc_ids.device)
        dummy_dec_mask = torch.ones((batch_size, 1), dtype=torch.long, device=enc_ids.device)

        # Create target override if specified
        target_override = None
        if target_idx is not None:
            target_override = F.one_hot(
                torch.tensor([target_idx] * batch_size, device=enc_ids.device),
                num_classes=self.n_targets,
            ).float()

        # Forward pass with counterfactual settings
        outputs = self.forward(
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            dec_ids=dummy_dec_ids,
            dec_mask=dummy_dec_mask,
            flip_style=flip_style,
            style_labels=style_labels,
            target_override=target_override,
        )

        return {
            "zm": outputs["zm"],
            "zu": outputs["zu"],
            "zt_onehot": outputs["zt_onehot"],
            "z_style_onehot": outputs["z_style_onehot"],
            "hate_logits": outputs["hate_logits"],
        }
