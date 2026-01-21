"""CADET Trainer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from cadet.datasets.dataloader import CADETLoader
from cadet.models.cadet import CADET
from cadet.training.base_trainer import BaseTrainer
from cadet.utils import to_serializable

logger = logging.getLogger(__name__)


class CADETTrainer(BaseTrainer):
    """Trainer for CADET model.

    Handles:
    - Multi-objective loss computation with modular design
    - Loss ablation support via drop_losses parameter
    - Progressive training schedule
    - Early stopping with validation (not before epoch 5)
    - HuggingFace evaluate library for epoch-wise evaluation
    - Checkpoint management
    - Latent extraction

    Attributes:
        model: CADET model instance (initialized in __init__)
        f1_metric: HuggingFace F1 metric evaluator
    """

    model: CADET  # Type annotation to fix lint errors
    f1_metric: Any  # Type annotation for evaluate metric

    def __init__(
        self,
        data_loader: CADETLoader,
        model_config: dict,
        training_config: dict,
        output_path: str | Path,
        device: str = "cuda",
        random_seed: int = 42,
    ):
        """Initialize CADET trainer.

        Args:
            data_loader: CADETLoader instance
            model_config: Model configuration dict
            training_config: Training configuration dict (includes drop_losses list)
            output_path: Output directory path
            device: Device for training
            random_seed: Random seed
        """
        super().__init__(
            source_style=data_loader.source_style,
            target_style=data_loader.target_style,
            model_config=model_config,
            output_path=Path(output_path),
            device=device,
            random_seed=random_seed,
        )

        self.data_loader = data_loader
        self.training_config = training_config

        # Loss ablation configuration - parse comma-separated string
        drop_losses_str = training_config.get("drop_losses", None)
        if drop_losses_str is None or drop_losses_str == "":
            self.drop_losses = []
        else:
            # Parse comma-separated string
            self.drop_losses = [loss.strip() for loss in drop_losses_str.split("-") if loss.strip()]

        self.active_losses = self._get_active_losses()

        logger.info(f"Active losses: {self.active_losses}")
        if self.drop_losses:
            logger.info(f"Dropped losses: {self.drop_losses}")

        # Initialize model
        # Load data to initialize target encoding
        data_loader.load_data()

        # Get number of target classes from dataloader
        if data_loader.n_targets is None:
            raise RuntimeError("n_targets not initialized in dataloader")
        n_targets = data_loader.n_targets

        self.model = CADET(
            n_targets=n_targets,
            encoder_checkpoint=model_config["encoder_checkpoint"],
            decoder_checkpoint=model_config["decoder_checkpoint"],
            style_dim=model_config["style_dim"],
            conf_dim=model_config["conf_dim"],
            orth_dim=model_config["orth_dim"],
            use_confounder_for_prediction=model_config["use_confounder_for_prediction"],
        ).to(device)

        # Setup optimizer
        self._setup_optimizer()

        # Training state
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.best_threshold = 0.5
        self.best_state_dict = None
        self.no_early_stop_before = training_config.get("progressive", {}).get(
            "no_early_stop_before_epoch", 5
        )

        # Batch loss tracking
        self.batch_losses = []  # Store all batch losses for analysis

        # Evaluation metric
        self.f1_metric = evaluate.load("f1")

        # Create output directories
        (self.output_path / "checkpoints" / "best").mkdir(parents=True, exist_ok=True)
        (self.output_path / "predictions").mkdir(parents=True, exist_ok=True)
        (self.output_path / "models").mkdir(parents=True, exist_ok=True)

    def load_model(self) -> CADET:
        """Load and initialize the model.

        Returns:
            CADET model instance

        Note:
            Model is already initialized in __init__, this method is for
            BaseTrainer interface compatibility.
        """
        return self.model

    def load_data(self) -> tuple[Any, Any, Any]:
        """Load training, validation, and test datasets.

        Returns:
            Tuple of (train_data, val_data, test_data)

        Note:
            Data loading is handled by CADETLoader, this method is for
            BaseTrainer interface compatibility.
        """
        return self.data_loader.load_data()

    def _get_active_losses(self) -> set[str]:
        """Get set of active loss names (not in drop_losses).

        Returns:
            Set of active loss names
        """
        all_losses = {
            "reconstruction",
            "hate",
            "target",
            "style",
            "kl",
            "orthogonality",
            "counterfactual",
            "cycle",
            "adversarial",
        }
        return all_losses - set(self.drop_losses)

    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for backbone and heads."""
        optimizer_config = self.training_config.get("optimizer", {})
        backbone_lr = optimizer_config.get("backbone_lr", 3e-5)
        head_lr = optimizer_config.get("head_lr", 2e-4)
        weight_decay = optimizer_config.get("weight_decay", 1e-2)

        # Separate backbone and head parameters
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "encoder" in name or "decoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )

    def train(self) -> dict[str, Any]:
        """Train CADET model end-to-end.

        Returns:
            Dictionary with training results:
            {
                'best_epoch': int,
                'best_val_f1': float,
                'best_threshold': float,
                'num_train_samples': int,
                'num_val_samples': int,
                'training_history': list
            }
        """
        logger.info("Starting CADET training...")

        # Load data
        train_loader, val_loader, _ = self.data_loader.get_dataloaders(
            self.training_config["batch_size"]
        )

        max_epochs = self.training_config["max_epochs"]
        patience = self.training_config["patience"]
        training_history = []

        for epoch in range(1, max_epochs + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch}/{max_epochs}")
            logger.info(f"{'=' * 60}")

            # Train for one epoch
            train_metrics = self._train_epoch(epoch, train_loader)

            # Log training metrics
            logger.info(f"Training - Total Loss: {train_metrics['total_loss']:.4f}")
            for loss_name, loss_value in train_metrics.items():
                if loss_name != "total_loss":
                    logger.info(f"  {loss_name}: {loss_value:.4f}")

            # Validate (skip before epoch 5 per change notes)
            if epoch >= self.no_early_stop_before:
                val_f1, val_threshold = self._validate(val_loader)
                logger.info(f"Validation - F1: {val_f1:.4f}, Threshold: {val_threshold:.4f}")

                # Save checkpoint if improved
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_threshold = val_threshold
                    self.best_state_dict = self.model.state_dict().copy()
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_f1, val_threshold)
                    logger.info(f"✓ New best model saved (F1: {val_f1:.4f})")
                else:
                    self.patience_counter += 1
                    logger.info(f"No improvement ({self.patience_counter}/{patience})")

                # Early stopping check
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                logger.info(f"Skipping validation (epoch < {self.no_early_stop_before})")

            # Record history
            history_entry = {"epoch": epoch, **train_metrics}
            if epoch >= self.no_early_stop_before:
                history_entry["val_f1"] = val_f1
                history_entry["val_threshold"] = val_threshold
            training_history.append(history_entry)

        # Load best model
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            logger.info(f"\nLoaded best model (F1: {self.best_val_f1:.4f})")

        # Save batch losses to CSV
        self._save_batch_losses()

        # Prepare training results
        train_dataset, val_dataset, _ = self.data_loader.load_data()
        results = {
            "best_epoch": len(training_history),
            "best_val_f1": self.best_val_f1,
            "best_threshold": self.best_threshold,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
            "training_history": training_history,
        }

        logger.info("\nTraining complete!")
        return results

    def _train_epoch(self, epoch: int, train_loader: DataLoader) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number
            train_loader: Training data loader

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_losses = {loss_name: 0.0 for loss_name in self.active_losses}
        epoch_losses["total_loss"] = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), 1):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                enc_ids=batch["enc_input_ids"],
                enc_mask=batch["enc_attention_mask"],
                dec_ids=batch["dec_input_ids"],
                dec_mask=batch["dec_attention_mask"],
                hate_labels=batch.get("hate_label"),
                style_labels=batch.get("style"),
                tgt_labels=batch.get("target_id"),
                tgt_conf=batch.get("target_conf"),
                style_tau=self.model_config.get("style_tau", 0.5),
                target_tau=self.model_config.get("target_tau", 0.5),
            )

            # Compute loss
            total_loss, loss_components = self._compute_loss(outputs, batch, epoch)

            # Record batch loss (for analysis, do not log to avoid clutter)
            batch_loss_record = {
                "epoch": epoch,
                "batch": batch_idx,
                "total_loss": total_loss.item(),
                **loss_components,
            }
            self.batch_losses.append(batch_loss_record)

            # Backward and optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses for epoch average
            epoch_losses["total_loss"] += total_loss.item()
            for loss_name, loss_value in loss_components.items():
                if loss_name in self.active_losses:
                    epoch_losses[loss_name] += loss_value

        # Average losses
        n_batches = len(train_loader)

        # Handle empty data loader case
        if n_batches == 0:
            logger.warning("Training data loader is empty! Check dataset size and filtering.")
            # Return zero losses for empty loader
            return {loss_name: 0.0 for loss_name in self.active_losses | {"total_loss"}}

        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        return epoch_losses

    def _save_batch_losses(self) -> None:
        """Save batch-level losses to CSV for analysis."""
        if not self.batch_losses:
            logger.warning("No batch losses to save")
            return

        import pandas as pd

        # Convert to DataFrame
        losses_df = pd.DataFrame(self.batch_losses)

        # Save to models directory
        losses_path = self.output_path / "models" / "training_losses.csv"
        losses_df.to_csv(losses_path, index=False)

        logger.info(f"Saved {len(self.batch_losses)} batch loss records to {losses_path}")

        # Log summary statistics (only after training complete)
        if len(losses_df) > 0:
            final_epoch_losses = losses_df[losses_df["epoch"] == losses_df["epoch"].max()]
            avg_final_loss = final_epoch_losses["total_loss"].mean()
            logger.info(f"Final epoch average total loss: {avg_final_loss:.4f}")

    def _compute_loss(
        self, outputs: dict, batch: dict, epoch: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-objective loss with modular design.

        Supports loss ablation via self.drop_losses configuration.

        Args:
            outputs: Model forward outputs
            batch: Input batch
            epoch: Current epoch (for progressive weighting)

        Returns:
            (total_loss, loss_components_dict)

        Implementation:
            For each loss in active_losses:
                - Compute loss component
                - Apply progressive weight
                - Add to total_loss
            Skipped losses are not computed (efficient ablation)
        """
        loss_components = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Get progressive weights
        weights = self._get_progressive_weights(epoch)

        # 1. Reconstruction loss
        if "reconstruction" in self.active_losses:
            rec_loss = outputs["rec_loss"]
            weighted_rec = weights["reconstruction"] * rec_loss
            total_loss = total_loss + weighted_rec
            loss_components["reconstruction"] = rec_loss.item()

        # 2. Hate classification loss (from M) with dynamic class weighting
        if "hate" in self.active_losses:
            hate_logits = outputs["hate_logits"]
            hate_labels = batch["hate_label"]
            # Compute class weights dynamically from batch
            num_classes = hate_logits.shape[-1] if hate_logits.dim() > 1 else 2
            if num_classes == 2 and (hate_logits.dim() == 1 or hate_logits.shape[-1] == 1):
                # Binary case: use BCEWithLogitsLoss with pos_weight
                hate_labels = hate_labels.float()
                pos = (hate_labels == 1).sum().item()
                neg = (hate_labels == 0).sum().item()
                total = pos + neg
                # Avoid division by zero
                weight_pos = total / (2 * pos) if pos > 0 else 1.0
                pos_weight = torch.tensor(weight_pos, device=hate_labels.device, dtype=torch.float)
                hate_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                hate_loss = hate_loss_fn(hate_logits.squeeze(-1), hate_labels)
            else:
                # Multiclass case: use CrossEntropyLoss with class weights
                hate_labels = hate_labels.long()
                class_counts = torch.bincount(hate_labels, minlength=num_classes).float()
                total = class_counts.sum()
                class_weights = total / (num_classes * (class_counts + 1e-6))
                hate_loss_fn = torch.nn.CrossEntropyLoss(
                    weight=class_weights.to(hate_logits.device)
                )
                hate_loss = hate_loss_fn(hate_logits, hate_labels)
            weighted_hate = weights["hate"] * hate_loss
            total_loss = total_loss + weighted_hate
            loss_components["hate"] = hate_loss.item()

        # 3. Target classification loss (weighted by confidence)
        if "target" in self.active_losses:
            target_loss = F.cross_entropy(
                outputs["tgt_logits"], batch["target_id"], reduction="none"
            )
            target_loss = (target_loss * batch["target_conf"]).mean()
            weighted_target = weights["target"] * target_loss
            total_loss = total_loss + weighted_target
            loss_components["target"] = target_loss.item()

        # 4. Style classification loss
        if "style" in self.active_losses:
            style_loss = F.cross_entropy(outputs["style_logits_pred"], batch["style"])
            weighted_style = weights["style"] * style_loss
            total_loss = total_loss + weighted_style
            loss_components["style"] = style_loss.item()

        # 5. KL divergence losses
        if "kl" in self.active_losses:
            # KL terms are now properly normalized in the model (mean over batch)
            kl_loss = outputs["KL_m"] + outputs["KL_u"] + outputs["KL_t"] + outputs["KL_style"]
            weighted_kl = weights["kl"] * kl_loss
            total_loss = total_loss + weighted_kl
            loss_components["kl"] = kl_loss.item()

        # 6. Orthogonality loss
        if "orthogonality" in self.active_losses:
            orth_loss = (
                outputs["o_mt"]
                + outputs["o_m_style"]
                + outputs["o_t_style"]
                + outputs["o_mu"]
                + outputs["o_tu"]
                + outputs["o_style_u"]
            ) / 6.0
            weighted_orth = weights["orthogonality"] * orth_loss
            total_loss = total_loss + weighted_orth
            loss_components["orthogonality"] = orth_loss.item()

        # 7. Counterfactual loss (hate prediction should be stable when flipping style)
        if "counterfactual" in self.active_losses:
            # Generate counterfactual with flipped style
            outputs_cf = self.model(
                enc_ids=batch["enc_input_ids"],
                enc_mask=batch["enc_attention_mask"],
                dec_ids=batch["dec_input_ids"],
                dec_mask=batch["dec_attention_mask"],
                flip_style=True,
                style_labels=batch["style"],
            )
            # Hate prediction should be consistent
            cf_loss = F.mse_loss(
                F.softmax(outputs["hate_logits"], dim=-1),
                F.softmax(outputs_cf["hate_logits"], dim=-1),
            )
            weighted_cf = weights["counterfactual"] * cf_loss
            total_loss = total_loss + weighted_cf
            loss_components["counterfactual"] = cf_loss.item()

        # 8. Cycle consistency loss (style flip and back)
        if "cycle" in self.active_losses:
            # This is computationally expensive, so it's optional
            # Flip style twice should give same latents
            outputs_cycle = self.model(
                enc_ids=batch["enc_input_ids"],
                enc_mask=batch["enc_attention_mask"],
                dec_ids=batch["dec_input_ids"],
                dec_mask=batch["dec_attention_mask"],
                flip_style=True,
                style_labels=batch["style"],
            )
            # Note: In practice, cycle consistency is hard to enforce in VAE
            # We approximate by ensuring motivation (M) is similar
            cycle_loss = F.mse_loss(outputs["zm"], outputs_cycle["zm"])
            weighted_cycle = weights["cycle"] * cycle_loss
            total_loss = total_loss + weighted_cycle
            loss_components["cycle"] = cycle_loss.item()

        # 9. Adversarial loss
        if "adversarial" in self.active_losses:
            adv_loss = outputs["adv_loss"]
            weighted_adv = weights["adversarial"] * adv_loss
            total_loss = total_loss + weighted_adv
            loss_components["adversarial"] = adv_loss.item()

        return total_loss, loss_components

    def _get_progressive_weights(self, epoch: int) -> dict[str, float]:
        """Get progressive loss weights for current epoch.

        Adjusts weights dynamically for dropped losses (sets to 0).

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of loss weights
        """
        # Get base weights from config
        base_weights = self.training_config.get("loss_weights", {})
        progressive_config = self.training_config.get("progressive", {})

        # Initialize weights
        weights = {}

        # Get progressive schedule parameters
        rec_ramp_epochs = progressive_config.get("reconstruction_ramp_epochs", 5)
        kl_start_epoch = progressive_config.get("kl_start_epoch", 2)
        kl_full_epoch = progressive_config.get("kl_full_epoch", 6)
        orth_start_epoch = progressive_config.get("orth_start_epoch", 3)
        adv_start_epoch = progressive_config.get("adv_start_epoch", 2)

        # 1. Reconstruction: Ramp from 0 to full weight
        if epoch <= rec_ramp_epochs:
            if rec_ramp_epochs == 1:
                rec_scale = 1.0  # Full weight from epoch 1
            else:
                rec_scale = epoch / rec_ramp_epochs
        else:
            rec_scale = 1.0
        weights["reconstruction"] = base_weights.get("reconstruction", 0.5) * rec_scale

        # 2. Hate: Active from start
        weights["hate"] = base_weights.get("hate", 2.0)

        # 3. Target: Active from start
        weights["target"] = base_weights.get("target", 0.5)

        # 4. Style: Active from start
        weights["style"] = base_weights.get("style", 1.0)

        # 5. KL: Ramp from kl_start_epoch to kl_full_epoch
        if epoch < kl_start_epoch:
            kl_scale = 0.0
        elif epoch <= kl_full_epoch:
            if kl_full_epoch == kl_start_epoch:
                kl_scale = 1.0  # No ramping, full weight immediately
            else:
                kl_scale = (epoch - kl_start_epoch) / (kl_full_epoch - kl_start_epoch)
        else:
            kl_scale = 1.0
        weights["kl"] = base_weights.get("kl", 0.1) * kl_scale

        # 6. Orthogonality: Active from orth_start_epoch
        if epoch < orth_start_epoch:
            orth_scale = 0.0
        else:
            orth_scale = 1.0
        weights["orthogonality"] = base_weights.get("orthogonality", 3.0) * orth_scale

        # 7. Counterfactual: Active from start
        weights["counterfactual"] = base_weights.get("counterfactual", 0.5)

        # 8. Cycle: Active from start
        weights["cycle"] = base_weights.get("cycle", 0.5)

        # 9. Adversarial: Ramp from adv_start_epoch
        if epoch < adv_start_epoch:
            adv_scale = 0.1
        elif epoch <= rec_ramp_epochs:
            if rec_ramp_epochs == adv_start_epoch:
                adv_scale = 1.0  # No ramping, full weight immediately
            else:
                adv_scale = 0.1 + 0.9 * (epoch - adv_start_epoch) / (
                    rec_ramp_epochs - adv_start_epoch
                )
        else:
            adv_scale = 1.0
        weights["adversarial"] = base_weights.get("adversarial", 1.0) * adv_scale

        # Set dropped losses to 0
        for loss_name in self.drop_losses:
            if loss_name in weights:
                weights[loss_name] = 0.0

        return weights

    def _validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """Run validation and optimize threshold.

        Args:
            val_loader: Validation data loader

        Returns:
            (best_f1, best_threshold)
        """
        self.model.eval()

        # Handle empty validation loader
        if len(val_loader) == 0:
            logger.warning("Validation data loader is empty! Using default threshold.")
            return 0.5, 0.5  # Default F1 and threshold

        all_hate_logits = []
        all_hate_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    enc_ids=batch["enc_input_ids"],
                    enc_mask=batch["enc_attention_mask"],
                    dec_ids=batch["dec_input_ids"],
                    dec_mask=batch["dec_attention_mask"],
                    style_tau=self.model_config.get("style_tau", 0.5),
                    target_tau=self.model_config.get("target_tau", 0.5),
                )

                all_hate_logits.append(outputs["hate_logits"].cpu())
                all_hate_labels.append(batch["hate_label"].cpu())

        # Concatenate all batches
        hate_logits = torch.cat(all_hate_logits, dim=0)
        hate_labels = torch.cat(all_hate_labels, dim=0)

        # Convert logits to probabilities
        hate_probs = F.softmax(hate_logits, dim=-1)[:, 1].numpy()
        hate_labels_np = hate_labels.numpy()

        # Optimize threshold
        best_f1, best_threshold = self._optimize_threshold(hate_labels_np, hate_probs)

        return best_f1, best_threshold

    def _optimize_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
        """Optimize classification threshold.

        Args:
            y_true: True labels
            y_scores: Predicted probabilities

        Returns:
            (best_f1, best_threshold)
        """
        threshold_config = self.training_config
        threshold_range = threshold_config.get("threshold_range", [0.05, 0.95])
        n_thresholds = threshold_config.get("n_thresholds", 19)

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = self.f1_metric.compute(predictions=y_pred, references=y_true)["f1"]

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_f1, best_threshold

    def _save_checkpoint(self, epoch: int, val_f1: float, threshold: float):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            val_f1: Validation F1 score
            threshold: Optimal threshold
        """
        checkpoint_dir = self.output_path / "checkpoints" / "best"
        checkpoint_path = checkpoint_dir / "model.pt"

        # Save model state
        torch.save(self.model.state_dict(), checkpoint_path)

        # Save metadata (ensure configs are serializable)
        metadata = {
            "epoch": epoch,
            "val_f1": val_f1,
            "threshold": threshold,
            "model_config": to_serializable(self.model_config),
            "training_config": to_serializable(self.training_config),
        }

        metadata_path = checkpoint_dir / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def inference(self) -> dict[str, Any]:
        """Run inference on test set with memory-efficient embeddings streaming.

        Saves two separate files:
        1. embeddings.npz: Last activation layer (zm) used for hate prediction and t-SNE
        2. latents.npz: All latents (zm, zu, zt, z_style) for causal analysis

        Returns:
            Dictionary with:
            {
                'predictions_file': str,
                'embeddings_file': str,
                'latents_file': str,
                'num_samples': int
            }
        """
        logger.info("Running inference with memory-optimized embeddings streaming...")
        self.model.eval()

        # Load test data
        _, _, test_loader = self.data_loader.get_dataloaders(self.training_config["batch_size"])

        # Get total samples from test dataset
        _, _, test_dataset = self.data_loader.load_data()
        total_samples = len(test_dataset)
        logger.info(f"Processing {total_samples} test samples...")

        # Get dimensions from first batch
        first_batch = next(iter(test_loader))
        first_batch = {k: v.to(self.device) for k, v in first_batch.items()}

        with torch.no_grad():
            sample_outputs = self.model(
                enc_ids=first_batch["enc_input_ids"][:1],
                enc_mask=first_batch["enc_attention_mask"][:1],
                dec_ids=first_batch["dec_input_ids"][:1],
                dec_mask=first_batch["dec_attention_mask"][:1],
                style_tau=self.model_config.get("style_tau", 0.5),
                target_tau=self.model_config.get("target_tau", 0.5),
            )

            # Get dimensions
            zm_dim = sample_outputs["zm"].shape[-1]
            zu_dim = sample_outputs["zu"].shape[-1]
            zt_dim = sample_outputs["zt_onehot"].shape[-1]
            z_style_dim = sample_outputs["z_style_onehot"].shape[-1]

            logger.info(
                f"Embedding dimensions - zm: {zm_dim}, zu: {zu_dim}, "
                f"zt: {zt_dim}, z_style: {z_style_dim}"
            )

        # Setup paths
        embeddings_path = self.output_path / "models" / "embeddings.npz"
        latents_path = self.output_path / "models" / "latents.npz"

        temp_zm_path = self.output_path / "models" / "temp_zm.mmap"
        temp_zu_path = self.output_path / "models" / "temp_zu.mmap"
        temp_zt_path = self.output_path / "models" / "temp_zt.mmap"
        temp_z_style_path = self.output_path / "models" / "temp_z_style.mmap"

        # Create memory-mapped arrays for all latents
        zm_mmap = np.memmap(
            temp_zm_path, dtype=np.float32, mode="w+", shape=(total_samples, zm_dim)
        )
        zu_mmap = np.memmap(
            temp_zu_path, dtype=np.float32, mode="w+", shape=(total_samples, zu_dim)
        )
        zt_mmap = np.memmap(
            temp_zt_path, dtype=np.float32, mode="w+", shape=(total_samples, zt_dim)
        )
        z_style_mmap = np.memmap(
            temp_z_style_path, dtype=np.float32, mode="w+", shape=(total_samples, z_style_dim)
        )

        all_hate_labels = []
        all_hate_preds = []
        all_hate_probs = []
        sample_idx = 0

        logger.info("Streaming embeddings and latents to disk...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
                batch_size = len(batch["hate_label"])

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    enc_ids=batch["enc_input_ids"],
                    enc_mask=batch["enc_attention_mask"],
                    dec_ids=batch["dec_input_ids"],
                    dec_mask=batch["dec_attention_mask"],
                    style_tau=self.model_config.get("style_tau", 0.5),
                    target_tau=self.model_config.get("target_tau", 0.5),
                )

                # Stream latents directly to memory-mapped files
                end_idx = sample_idx + batch_size
                zm_mmap[sample_idx:end_idx] = outputs["zm"].cpu().numpy().astype(np.float32)
                zu_mmap[sample_idx:end_idx] = outputs["zu"].cpu().numpy().astype(np.float32)
                zt_mmap[sample_idx:end_idx] = outputs["zt_onehot"].cpu().numpy().astype(np.float32)
                z_style_mmap[sample_idx:end_idx] = (
                    outputs["z_style_onehot"].cpu().numpy().astype(np.float32)
                )

                # Get predictions
                hate_logits = outputs["hate_logits"]
                hate_probs = F.softmax(hate_logits, dim=-1)[:, 1]
                hate_preds = (hate_probs >= self.best_threshold).long()

                # Store predictions in memory (much smaller)
                all_hate_labels.extend(batch["hate_label"].cpu().numpy())
                all_hate_preds.extend(hate_preds.cpu().numpy())
                all_hate_probs.extend(hate_probs.cpu().numpy())

                sample_idx += batch_size

                # Periodic GPU memory cleanup
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Ensure all data is written to disk
        zm_mmap.flush()
        zu_mmap.flush()
        zt_mmap.flush()
        z_style_mmap.flush()
        logger.info("Embeddings and latents streaming completed.")

        # Save predictions (use standard column names for evaluator compatibility)
        predictions_df = pd.DataFrame(
            {
                "text_id": list(range(len(all_hate_labels))),
                "true_label": all_hate_labels,
                "pred_label": all_hate_preds,
                "prob": all_hate_probs,
                "threshold": self.best_threshold,
            }
        )

        predictions_file = self.output_path / "predictions" / "test_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to {predictions_file}")

        # Get dataset metadata
        target_style = self.data_loader.target_style
        source_style = self.data_loader.source_style
        dataset_name = getattr(self.data_loader, "dataset_name", "unknown")

        n_samples = len(all_hate_labels)
        style_labels = np.ones(n_samples) if target_style == "explicit" else np.zeros(n_samples)

        # Save embeddings.npz (zm only - for t-SNE visualization)
        logger.info("Saving embeddings.npz (zm for t-SNE)...")
        np.savez_compressed(
            embeddings_path,
            embeddings=zm_mmap[: len(all_hate_labels)],  # Only save actual samples
            labels=np.array(all_hate_labels),
            text_ids=np.arange(len(all_hate_labels)),
            style_labels=style_labels,
            # Metadata for t-SNE visualization
            model_name="cadet",
            target_style=target_style,
            source_style=source_style,
            dataset_name=dataset_name,
        )
        logger.info(f"Embeddings saved to {embeddings_path}")

        # Save latents.npz (all latents for causal analysis)
        logger.info("Saving latents.npz (all latents for causal analysis)...")
        np.savez_compressed(
            latents_path,
            zm=zm_mmap[: len(all_hate_labels)],
            zu=zu_mmap[: len(all_hate_labels)],
            zt=zt_mmap[: len(all_hate_labels)],
            z_style=z_style_mmap[: len(all_hate_labels)],
            hate_labels=np.array(all_hate_labels),
            text_ids=np.arange(len(all_hate_labels)),
            style_labels=style_labels,
            # Metadata
            target_style=target_style,
            source_style=source_style,
            dataset_name=dataset_name,
        )
        logger.info(f"Latents saved to {latents_path}")

        # Clean up temporary files
        del zm_mmap, zu_mmap, zt_mmap, z_style_mmap
        temp_zm_path.unlink(missing_ok=True)
        temp_zu_path.unlink(missing_ok=True)
        temp_zt_path.unlink(missing_ok=True)
        temp_z_style_path.unlink(missing_ok=True)

        logger.info(
            f"Final shapes - zm: ({len(all_hate_labels)}, {zm_dim}), "
            f"zu: ({len(all_hate_labels)}, {zu_dim}), "
            f"zt: ({len(all_hate_labels)}, {zt_dim}), "
            f"z_style: ({len(all_hate_labels)}, {z_style_dim})"
        )

        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "predictions_file": str(predictions_file),
            "embeddings_file": str(embeddings_path),
            "latents_file": str(latents_path),
            "num_samples": len(all_hate_labels),
        }

    def free_memory(self):
        """Free GPU memory."""
        if self.model is not None:
            self.model.cpu()
            self.model = None  # Release reference but keep attribute for type checking
        if hasattr(self, "optimizer"):
            del self.optimizer

        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Memory freed")
