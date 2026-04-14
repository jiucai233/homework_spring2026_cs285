"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        curr_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, chunk_size*action_dim))
        self.model = nn.Sequential(*layers)
        
        
        
    def compute_loss(
        self,
        state: torch.Tensor, # [batch_size, state_dim]
        action_chunk: torch.Tensor, # [batch_size, chunk_size, action_dim]
    ) -> torch.Tensor:
        model_action = self.model(state) # [batch_size, chunk_size * action_dim]
        model_action = model_action.view(-1, self.chunk_size, self.action_dim) # [batch_size, chunk_size, action_dim]
        loss = nn.functional.mse_loss(model_action, action_chunk) # scalar
        return loss

    def sample_actions(
        self,
        state: torch.Tensor, # [batch_size, state_dim]
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        actions = self.model(state) # [batch_size, chunk_size * action_dim]
        actions = actions.view(-1, self.chunk_size, self.action_dim) # [batch_size, chunk_size, action_dim]
        return actions


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        flatted_action_dim = chunk_size * action_dim
        curr_dim = state_dim + flatted_action_dim + 1
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, flatted_action_dim))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor, # [batch_size, state_dim]
        action_chunk: torch.Tensor, # [batch_size, chunk_size, action_dim]
    ) -> torch.Tensor:
        batch_size = state.shape[0] # int scalar
        time_step = torch.rand(batch_size, 1, 1, device=action_chunk.device) # [batch_size, 1, 1]
        noise = torch.normal(0, 1, size=action_chunk.shape, device=action_chunk.device) # [batch_size, chunk_size, action_dim]
        a_interpolation = time_step * action_chunk + ((1 - time_step) * noise) # [batch_size, chunk_size, action_dim]
        #flatten action chunk
        a_interpolation_flat = a_interpolation.view(batch_size, -1) # [batch_size, chunk_size * action_dim]
        time_step_flat = time_step.view(batch_size, 1) # [batch_size, 1]
        #concatenate state and a_interpolation_flat and time_step_flat
        input_tensor = torch.cat([state, a_interpolation_flat, time_step_flat], dim=1) # [batch_size, state_dim + chunk_size * action_dim + 1]
        #start of the forward pass
        pred_vel_flat = self.model(input_tensor) # [batch_size, chunk_size * action_dim]
        pred_vel = pred_vel_flat.view(batch_size, self.chunk_size, self.action_dim) # [batch_size, chunk_size, action_dim]
        target_vel = action_chunk - noise # [batch_size, chunk_size, action_dim]
        loss = torch.nn.functional.mse_loss(pred_vel, target_vel) # scalar

        return loss
    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor, # [batch_size, state_dim]
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0] # int scalar
        noise = torch.normal(0, 1, size=(batch_size, self.chunk_size, self.action_dim), device=state.device) # [batch_size, chunk_size, action_dim]
        A = noise # [batch_size, chunk_size, action_dim]
        for time_step in range(num_steps):

            curr_tau = time_step/num_steps # float scalar
            tau_tensor = torch.full((batch_size, 1, 1), curr_tau, device=state.device) # [batch_size, 1, 1]
            #flattening the var
            flattened_A = A.view(batch_size, -1) # [batch_size, chunk_size * action_dim]
            tau_tensor_flat = tau_tensor.view(batch_size, 1) # [batch_size, 1]
            #concatenate state and a_interpolation_flat and time_step_flat
            input_tensor = torch.cat([state, flattened_A, tau_tensor_flat], dim=1) # [batch_size, state_dim + chunk_size * action_dim + 1]

            pred_vel_flat = self.model(input_tensor) # [batch_size, chunk_size * action_dim]
            pred_vel = pred_vel_flat.view(batch_size, self.chunk_size, self.action_dim) # [batch_size, chunk_size, action_dim]
            A = A + 1/num_steps * pred_vel # [batch_size, chunk_size, action_dim]

        return A # [batch_size, chunk_size, action_dim]



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
