import torch
import numpy as np
import torch.nn as nn

from particle_gnn.models.Interaction_Particle import Interaction_Particle
from particle_gnn.models.MLP import MLP
from particle_gnn.utils import to_numpy, fig_init, choose_boundary_values
from particle_gnn.fitting_models import linear_model


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the Apache License, Version 2.0
    # found in the LICENSE file in the root directory of this source tree.

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):

            # student_output = F.normalize(student_output, eps=eps, p=2, dim=0)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()

        return loss


def get_embedding(model_a=None, dataset_number=0):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    return embedding


def get_embedding_time_series(model=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):
    embedding = []
    embedding.append(model.a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    indexes = np.arange(n_frames) * n_particles + cell_id

    return embedding[indexes]


def get_type_time_series(new_labels=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):

    indexes = np.arange(n_frames) * n_particles + cell_id

    return new_labels[indexes]


def get_in_features(rr=None, embedding=None, model=[], model_name=[], max_radius=[]):

    match model_name:
        case 'PDE_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_A_bis':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_GS':
            in_features = torch.cat(
                (rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, 10 ** embedding), dim=1)
        case 'PDE_G':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None],
                                     0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)

    return in_features


def choose_training_model(model_config=None, device=None):
    """Create and return an Interaction_Particle model based on the configuration.

    Args:
        model_config: Configuration object containing simulation and graph model parameters.
        device: Torch device to place the model on.

    Returns:
        Tuple of (model, bc_pos, bc_dpos).
    """

    aggr_type = model_config.graph_model.aggr_type
    dimension = model_config.simulation.dimension

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    model = Interaction_Particle(
        aggr_type=aggr_type,
        config=model_config,
        device=device,
        bc_dpos=bc_dpos,
        dimension=dimension,
    )
    model.edges = []

    return model, bc_pos, bc_dpos
