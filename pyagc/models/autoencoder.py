from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling

from pyagc.models.base import TrainableModel, LossOutput
from pyagc.utils import filter_kwargs

EPS = 1e-15
MAX_LOGSTD = 10


class GAE(TrainableModel):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Alias for :meth:`embed`."""
        return self.embed(*args, **kwargs)

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes node embeddings via the encoder."""
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""
        Computes the reconstruction loss for GAE.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices (positive edges).

        Returns:
            Reconstruction loss as a scalar tensor.
        """
        z = self.embed(x, edge_index, **kwargs)
        return self.recon_loss(z, edge_index)

    def loss_batch(self, batch: Data) -> Tensor:
        r"""
        Computes loss for a mini-batch with seed node slicing.

        Args:
            batch (Data): A mini-batch from the loader.

        Returns:
            Reconstruction loss as a scalar tensor.
        """
        z = self.embed(batch.x, batch.edge_index)
        z = z[:batch.batch_size]

        # Extract edges within the batch
        batch_mask = (batch.edge_index[0] < batch.batch_size) & (batch.edge_index[1] < batch.batch_size)
        batch_edge_index = batch.edge_index[:, batch_mask]

        return self.recon_loss(z, batch_edge_index)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)
        self.__mu__ = None
        self.__logstd__ = None

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        r"""Reparametrization trick for variational inference."""
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def embed(self, *args, **kwargs) -> Tensor:
        r"""
        Computes node embeddings via the variational encoder.

        The encoder outputs both :math:`\mu` and :math:`\log\sigma^2`,
        which are used for the reparametrization trick.
        """
        self.__mu__, self.__logstd__ = self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the VGAE loss with reconstruction and KL divergence components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices (positive edges).

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(x, edge_index, **kwargs)
        recon = self.recon_loss(z, edge_index)
        kl = self.kl_loss()

        return LossOutput(
            total=recon + kl,
            components={
                'recon': recon.item(),
                'kl': kl.item()
            }
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""
        Computes loss for a mini-batch with seed node slicing.

        Args:
            batch (Data): A mini-batch from the loader.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(batch.x, batch.edge_index)
        z = z[:batch.batch_size]
        mu = self.__mu__[:batch.batch_size]
        logstd = self.__logstd__[:batch.batch_size]

        # Extract edges within the batch
        batch_mask = (batch.edge_index[0] < batch.batch_size) & (batch.edge_index[1] < batch.batch_size)
        batch_edge_index = batch.edge_index[:, batch_mask]

        recon = self.recon_loss(z, batch_edge_index)
        kl = self.kl_loss(mu, logstd)

        return LossOutput(
            total=recon + kl,
            components={
                'recon': recon.item(),
                'kl': kl.item()
            }
        )


class ARGA(GAE):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.

    .. note::
        ARGA requires a two-phase training procedure (encoder + discriminator).
        Use :meth:`train_encoder` and :meth:`train_discriminator` separately,
        or implement a custom training loop.

    Args:
        encoder (torch.nn.Module): The encoder module.
        discriminator (torch.nn.Module): The discriminator module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(
            self,
            encoder: Module,
            discriminator: Module,
            decoder: Optional[Module] = None,
    ):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()
        reset(self.discriminator)

    def reg_loss(self, z: Tensor) -> Tensor:
        r"""Computes the regularization loss of the encoder.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    def discriminator_loss(self, z: Tensor) -> Tensor:
        r"""Computes the loss of the discriminator.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the ARGA encoder loss with reconstruction and regularization components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices (positive edges).

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(x, edge_index, **kwargs)
        recon = self.recon_loss(z, edge_index)
        reg = self.reg_loss(z)

        return LossOutput(
            total=recon + reg,
            components={
                'recon': recon.item(),
                'reg': reg.item()
            }
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""
        Computes encoder loss for a mini-batch with seed node slicing.

        Args:
            batch (Data): A mini-batch from the loader.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(batch.x, batch.edge_index)
        z = z[:batch.batch_size]

        # Extract edges within the batch
        batch_mask = (batch.edge_index[0] < batch.batch_size) & (batch.edge_index[1] < batch.batch_size)
        batch_edge_index = batch.edge_index[:, batch_mask]

        recon = self.recon_loss(z, batch_edge_index)
        reg = self.reg_loss(z)

        return LossOutput(
            total=recon + reg,
            components={
                'recon': recon.item(),
                'reg': reg.item()
            }
        )

    def train_encoder(self, data: Data, optimizer: torch.optim.Optimizer,
                      epoch: int, verbose: bool = True) -> float:
        r"""
        Trains the encoder for one epoch.

        This is equivalent to :meth:`train_full` but provided for clarity
        in the two-phase ARGA training procedure.

        Args:
            data (Data): The input full graph data.
            optimizer (torch.optim.Optimizer): The optimizer for encoder parameters.
            epoch (int): Current epoch number.
            verbose (bool, optional): If :obj:`True`, prints training progress.
                (default: :obj:`True`)

        Returns:
            Loss value of the epoch.
        """
        return self.train_full(data, optimizer, epoch, verbose)

    def train_discriminator(self, data: Data, optimizer: torch.optim.Optimizer,
                           epoch: int, verbose: bool = True) -> float:
        r"""
        Trains the discriminator for one epoch.

        Args:
            data (Data): The input full graph data.
            optimizer (torch.optim.Optimizer): The optimizer for discriminator parameters.
            epoch (int): Current epoch number.
            verbose (bool, optional): If :obj:`True`, prints training progress.
                (default: :obj:`True`)

        Returns:
            Discriminator loss value of the epoch.
        """
        self.train()
        optimizer.zero_grad()

        z = self.embed(**data)
        loss = self.discriminator_loss(z)

        loss.backward()
        optimizer.step()

        if verbose:
            print(f"Epoch: {epoch:02d} Discriminator Loss: {loss.item():.4f}")

        return float(loss.item())


class ARGVA(ARGA):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model from
    the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.

    .. note::
        ARGVA requires a two-phase training procedure (encoder + discriminator).
        Use :meth:`train_encoder` and :meth:`train_discriminator` separately,
        or implement a custom training loop.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        discriminator (torch.nn.Module): The discriminator module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        encoder: Module,
        discriminator: Module,
        decoder: Optional[Module] = None,
    ):
        # Note: We bypass ARGA's __init__ and call GAE's __init__ directly
        GAE.__init__(self, encoder, decoder)
        self.discriminator = discriminator
        self.__mu__ = None
        self.__logstd__ = None
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)
        reset(self.discriminator)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        r"""Reparametrization trick for variational inference."""
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def embed(self, *args, **kwargs) -> Tensor:
        r"""
        Computes node embeddings via the variational encoder.

        The encoder outputs both :math:`\mu` and :math:`\log\sigma^2`,
        which are used for the reparametrization trick.
        """
        self.__mu__, self.__logstd__ = self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logstd: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the ARGVA encoder loss with reconstruction, KL divergence,
        and regularization components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices (positive edges).

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(x, edge_index, **kwargs)
        recon = self.recon_loss(z, edge_index)
        kl = self.kl_loss()
        reg = self.reg_loss(z)

        return LossOutput(
            total=recon + kl + reg,
            components={
                'recon': recon.item(),
                'kl': kl.item(),
                'reg': reg.item()
            }
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""
        Computes encoder loss for a mini-batch with seed node slicing.

        Args:
            batch (Data): A mini-batch from the loader.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(batch.x, batch.edge_index)
        z = z[:batch.batch_size]
        mu = self.__mu__[:batch.batch_size]
        logstd = self.__logstd__[:batch.batch_size]

        # Extract edges within the batch
        batch_mask = (batch.edge_index[0] < batch.batch_size) & (batch.edge_index[1] < batch.batch_size)
        batch_edge_index = batch.edge_index[:, batch_mask]

        recon = self.recon_loss(z, batch_edge_index)
        kl = self.kl_loss(mu, logstd)
        reg = self.reg_loss(z)

        return LossOutput(
            total=recon + kl + reg,
            components={
                'recon': recon.item(),
                'kl': kl.item(),
                'reg': reg.item()
            }
        )


