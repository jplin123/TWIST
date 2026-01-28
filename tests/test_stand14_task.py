import os
from types import SimpleNamespace

import pytest
import torch
from isaacgym import gymapi, gymutil

from legged_gym.envs import task_registry
from legged_gym.gym_utils.helpers import class_to_dict
from legged_gym.envs.g1.stand_14dof_with_upper_csv_config import Stand14DofWithUpperCSVCfg
from legged_gym.envs.g1.upper_body_csv_player import UpperBodyCSVPlayer
from legged_gym.scripts import export_stand14dof_onnx
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils.normalizer import Normalizer


def make_test_args(headless=True, device="cuda:0"):
    sim_device_type, compute_device_id = gymutil.parse_device_str(device)
    return SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        device=device,
        rl_device=device,
        sim_device=device,
        sim_device_type=sim_device_type,
        compute_device_id=compute_device_id,
        use_gpu=sim_device_type == "cuda",
        use_gpu_pipeline=sim_device_type == "cuda",
        subscenes=1,
        num_threads=0,
        headless=headless,
        num_envs=None,
        seed=None,
        rows=None,
        cols=None,
        record_video=False,
        no_rand=False,
        teleop_mode=False,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=-1,
        fix_action_std=False,
        teacher_exptid="",
        teacher_checkpoint=-1,
        eval_student=False,
        record_log=False,
        debug=False,
        action_delay=False,
        draw=False,
        save=False,
        web=False,
        proj_name="stand14_test",
        exptid="stand14_test",
        no_wandb=True,
        motion_file="",
    )


def test_csv_player_mapping_and_default_pose():
    cfg = Stand14DofWithUpperCSVCfg()
    csv_path = os.path.join(os.path.dirname(__file__), "..", "assets", "upper_body_csv", "wave.csv")
    csv_path = os.path.abspath(csv_path)
    mapping = cfg.upper_body_csv.joint_columns
    player = UpperBodyCSVPlayer(
        file_path=csv_path,
        joint_mapping=mapping,
        playback_rate_hz=cfg.upper_body_csv.playback_rate_hz,
        default_pose=cfg.init_state.default_joint_angles,
        loop=True,
        interpolate=True,
        device="cpu",
    )
    times = torch.tensor([0.0, 0.25])
    amplitude = torch.tensor([0.0, 1.0])
    speed = torch.ones_like(amplitude)
    targets = player.sample(times, amplitude, speed)
    assert targets.shape == (2, len(mapping))
    default = torch.tensor(
        [cfg.init_state.default_joint_angles[name] for name in mapping.keys()],
        dtype=torch.float,
    )
    assert torch.allclose(targets[0], default, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IsaacGym GPU runtime required")
def test_env_shapes_and_step():
    env_cfg, _ = task_registry.get_cfgs("stand_14dof_with_upper_csv")
    env_cfg.env.num_envs = 2
    env_cfg.upper_body_csv.file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "assets", "upper_body_csv", "wave.csv")
    )
    sim_args = make_test_args(headless=True, device="cuda:0")
    env, _ = task_registry.make_env("stand_14dof_with_upper_csv", args=sim_args, env_cfg=env_cfg)
    obs = env.get_observations()
    assert obs.shape[1] == env_cfg.env.num_observations
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    obs, _, _, _, _ = env.step(actions)
    assert obs.shape == (env.num_envs, env_cfg.env.num_observations)


def test_export_script_produces_onnx(tmp_path):
    env_cfg, train_cfg = task_registry.get_cfgs("stand_14dof_with_upper_csv")
    actor = ActorCritic(
        num_prop=env_cfg.env.n_proprio,
        num_critic_obs=env_cfg.env.num_observations,
        num_priv_latent=env_cfg.env.n_priv_latent,
        num_hist=env_cfg.env.history_len,
        num_actions=env_cfg.env.num_actions,
        **class_to_dict(train_cfg.policy),
    )
    normalizer = Normalizer(
        shape=env_cfg.env.num_observations, device=torch.device("cpu"), dtype=torch.float
    )
    state = {"model_state_dict": actor.state_dict(), "normalizer": normalizer}
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    ckpt_path = run_dir / "model_1.pt"
    torch.save(state, ckpt_path)
    policy, obs_dim = export_stand14dof_onnx.load_policy(
        "stand_14dof_with_upper_csv", str(ckpt_path), torch.device("cpu")
    )
    dummy = torch.zeros(1, obs_dim)
    out_path = tmp_path / "stand14dof.onnx"
    torch.onnx.export(
        policy,
        dummy,
        str(out_path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    assert out_path.exists()
