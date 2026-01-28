import argparse
import os
from types import SimpleNamespace

import torch

from legged_gym.envs import task_registry
from legged_gym.gym_utils.helpers import class_to_dict, get_load_path
from rsl_rl.modules.actor_critic import ActorCritic


class OnnxDeterministicPolicy(torch.nn.Module):
    def __init__(self, actor, normalizer=None):
        super().__init__()
        self.actor = actor
        self.has_normalizer = normalizer is not None
        if self.has_normalizer:
            self.register_buffer("obs_mean", normalizer.get_mean().detach().clone())
            self.register_buffer("obs_std", normalizer.get_std().detach().clone())
        else:
            self.obs_mean = None
            self.obs_std = None
        self.eps = 1e-4

    def forward(self, obs):
        if self.has_normalizer:
            obs = (obs - self.obs_mean) / (self.obs_std + self.eps)
        return self.actor(obs)


def parse_args():
    parser = argparse.ArgumentParser("Export stand_14dof_with_upper_csv policy to ONNX.")
    parser.add_argument(
        "--task",
        type=str,
        default="stand_14dof_with_upper_csv",
        help="Registered task name.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the training run directory that contains model_*.pt files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Checkpoint index to export. Defaults to the latest.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stand14dof.onnx",
        help="Output ONNX file.",
    )
    return parser.parse_args()


def load_policy(task_name: str, checkpoint_path: str, device: torch.device):
    env_cfg, train_cfg = task_registry.get_cfgs(task_name)
    actor_critic = ActorCritic(
        num_prop=env_cfg.env.n_proprio,
        num_critic_obs=env_cfg.env.num_observations,
        num_priv_latent=env_cfg.env.n_priv_latent,
        num_hist=env_cfg.env.history_len,
        num_actions=env_cfg.env.num_actions,
        **class_to_dict(train_cfg.policy),
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    actor_critic.load_state_dict(state_dict["model_state_dict"], strict=False)
    normalizer = state_dict.get("normalizer", None)
    actor_critic.eval()
    policy = OnnxDeterministicPolicy(actor_critic.actor.to(device), normalizer)
    policy.eval()
    return policy, env_cfg.env.num_observations


def main():
    args = parse_args()
    checkpoint_path = get_load_path(args.run_dir, load_run=-1, checkpoint=args.checkpoint)
    device = torch.device("cpu")
    policy, obs_dim = load_policy(args.task, checkpoint_path, device)
    dummy = torch.zeros(1, obs_dim, device=device)
    output_path = os.path.abspath(args.output)
    torch.onnx.export(
        policy,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    print(f"Exported ONNX policy to {output_path}")


if __name__ == "__main__":
    main()
