"""Checkpoint management utilities."""

import os
import torch
from typing import Optional, Dict, Any


class CheckpointManager:
    """Manages model checkpoints with support for resuming training."""

    def __init__(self, ckpt_dir: str, model_name: str, logger=None):
        """
        Args:
            ckpt_dir (str): Directory to save checkpoints
            model_name (str): Base name for checkpoint files
            logger: Logger instance for logging
        """
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.logger = logger
        os.makedirs(ckpt_dir, exist_ok=True)

        # Define checkpoint paths
        self.best_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
        self.last_path = os.path.join(ckpt_dir, f"{model_name}_last.pt")

        # Track best performance
        self.best_loss = float('inf')

    def save_checkpoint(self, model, optimizer, epoch: int, loss: float,
                        is_best: bool = False, batch_idx: Optional[int] = None,
                        additional_info: Optional[Dict[str, Any]] = None):
        """Save a checkpoint with full training state.

        Args:
            model: The model to save
            optimizer: The optimizer state
            epoch (int): Current epoch number
            loss (float): Current loss value
            is_best (bool): Whether this is the best model so far
            batch_idx (int, optional): Current batch index within epoch (for intra-epoch saves)
            additional_info (dict): Additional information to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
        }

        # Add batch information if provided (for intra-epoch checkpoints)
        if batch_idx is not None:
            checkpoint['batch_idx'] = batch_idx

        if additional_info is not None:
            checkpoint.update(additional_info)

        # Always save last checkpoint
        torch.save(checkpoint, self.last_path)

        # Save best checkpoint if this is the best model
        if is_best:
            torch.save(checkpoint, self.best_path)
            self.best_loss = loss
            if self.logger:
                batch_str = f" (batch {batch_idx})" if batch_idx is not None else ""
                self.logger.info(f"✓ Best model saved at epoch {epoch}{batch_str} with loss {loss:.4f}")

        if self.logger and batch_idx is not None:
            # Log for intra-epoch saves
            if batch_idx % 100 == 0:  # Log every 100 batches to avoid spam
                self.logger.info(f"✓ Checkpoint saved at epoch {epoch}, batch {batch_idx}")

    def load_checkpoint(self, model, optimizer=None, load_best: bool = True,
                        device: Optional[str|torch.device] = 'cpu'):
        """Load a checkpoint and restore training state.

        Args:
            model: The model to load weights into
            optimizer: The optimizer to restore state (optional)
            load_best (bool): If True, load best checkpoint; otherwise load last
            device: Device to map the checkpoint to

        Returns:
            Dictionary with checkpoint information (epoch, loss, batch_idx, etc.)
            Returns None if checkpoint doesn't exist
        """
        ckpt_path = self.best_path if load_best else self.last_path

        if not os.path.exists(ckpt_path):
            if self.logger:
                self.logger.info(f"No checkpoint found at {ckpt_path}")
            return None

        if self.logger:
            self.logger.info(f"Loading checkpoint from {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Update best loss tracking
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']

        if self.logger:
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            batch_idx = checkpoint.get('batch_idx', None)
            batch_str = f", batch={batch_idx}" if batch_idx is not None else ""
            self.logger.info(f"✓ Checkpoint loaded: epoch={epoch}{batch_str}, loss={loss}")

        return checkpoint

    def has_checkpoint(self, load_best: bool = True) -> bool:
        """Check if a checkpoint exists.

        Args:
            load_best (bool): If True, check for best checkpoint; otherwise check last

        Returns:
            bool: True if checkpoint exists
        """
        ckpt_path = self.best_path if load_best else self.last_path
        return os.path.exists(ckpt_path)


class MultiStageCheckpointManager(CheckpointManager):
    """Checkpoint manager for multi-stage training."""

    def __init__(self, ckpt_dir: str, model_name: str, stages: list = None, logger=None):
        """
        Args:
            ckpt_dir (str): Directory to save checkpoints
            model_name (str): Base name for checkpoint files
            stages: List of stage names (e.g., ['pretrain', 'finetune'])
            logger: Logger instance for logging
        """
        super().__init__(ckpt_dir, model_name, logger)

        self.stages = stages or ['pretrain', 'finetune']
        self.stage_paths = {}
        self.best_losses = {}

        # Create paths for each stage
        for stage in self.stages:
            self.stage_paths[stage] = {
                'best': os.path.join(ckpt_dir, f"{model_name}_{stage}_best.pt"),
                'last': os.path.join(ckpt_dir, f"{model_name}_{stage}_last.pt")
            }
            self.best_losses[stage] = float('inf')

    def save_checkpoint(self, model, optimizer, epoch: int, loss: float,
                        stage: str = 'pretrain', is_best: bool = False,
                        batch_idx: Optional[int] = None,
                        additional_info: Optional[Dict[str, Any]] = None):
        """Save a checkpoint with full training state.

        Args:
            model: The model to save
            optimizer: The optimizer state
            epoch (int): Current epoch number
            loss (float): Current loss value
            stage (str): Training stage ('pretrain' or 'finetune')
            is_best (bool): Whether this is the best model so far
            batch_idx (int, optional): Current batch index within epoch
            additional_info (dict): Additional information to save
        """
        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}. Available stages: {self.stages}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'stage': stage,
            'best_loss': self.best_losses[stage],
        }

        if batch_idx is not None:
            checkpoint['batch_idx'] = batch_idx

        if additional_info is not None:
            checkpoint.update(additional_info)

        # Save last checkpoint for this stage
        last_path = self.stage_paths[stage]['last']
        torch.save(checkpoint, last_path)

        # Save best checkpoint if this is the best model
        if is_best:
            best_path = self.stage_paths[stage]['best']
            torch.save(checkpoint, best_path)
            self.best_losses[stage] = loss
            if self.logger:
                batch_str = f" (batch {batch_idx})" if batch_idx is not None else ""
                self.logger.info(
                    f"✓ Best {stage} model saved at epoch {epoch}{batch_str} with loss {loss:.4f}"
                )

    def load_checkpoint(self, model, optimizer=None, stage: str = 'pretrain',
                        load_best: bool = True, device: Optional[str|torch.device] = 'cpu'):
        """Load a checkpoint and restore training state.

        Args:
            model: The model to load weights into
            optimizer: The optimizer to restore state (optional)
            stage (str): Training stage ('pretrain' or 'finetune')
            load_best (bool): If True, load best checkpoint; otherwise load last
            device: Device to map the checkpoint to

        Returns:
            Dictionary with checkpoint information or None if not found
        """
        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}. Available stages: {self.stages}")

        ckpt_path = self.stage_paths[stage]['best' if load_best else 'last']

        if not os.path.exists(ckpt_path):
            if self.logger:
                self.logger.info(f"No {stage} checkpoint found at {ckpt_path}")
            return None

        if self.logger:
            self.logger.info(f"Loading {stage} checkpoint from {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Update best loss tracking
        if 'best_loss' in checkpoint:
            self.best_losses[stage] = checkpoint['best_loss']

        if self.logger:
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            batch_idx = checkpoint.get('batch_idx', None)
            batch_str = f", batch={batch_idx}" if batch_idx is not None else ""
            self.logger.info(f"✓ Checkpoint loaded: epoch={epoch}{batch_str}, loss={loss}")

        return checkpoint

    def has_checkpoint(self, stage: str = 'pretrain', load_best: bool = True) -> bool:
        """Check if a stage-specific checkpoint exists.

        Args:
            stage (str): Training stage ('pretrain' or 'finetune')
            load_best (bool): If True, check for best checkpoint; otherwise check last

        Returns:
            bool: True if checkpoint exists
        """
        if stage not in self.stages:
            return False

        ckpt_path = self.stage_paths[stage]['best' if load_best else 'last']
        return os.path.exists(ckpt_path)
