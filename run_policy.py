import argparse
import os
from types import SimpleNamespace

import onnxruntime as ort
import torch
from isaacgym import gymapi, gymutil

from legged_gym.envs import task_registry


def build_sim_args(headless: bool, device: str):
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
        proj_name="stand14_eval",
        exptid="stand14_eval",
        no_wandb=True,
        motion_file="",
    )


def parse_args():
    parser = argparse.ArgumentParser("Run stand_14dof_with_upper_csv ONNX policy.")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX policy path.")
    parser.add_argument("--csv", type=str, default=None, help="Upper-body CSV file to play.")
    parser.add_argument("--task", type=str, default="stand_14dof_with_upper_csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--headless", action="store_true", help="Disable viewer.")
    return parser.parse_args()


def main():
    args = parse_args()

    env_cfg, _ = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = args.num_envs
    if args.csv is not None:
        env_cfg.upper_body_csv.file = os.path.abspath(args.csv)

    sim_args = build_sim_args(args.headless, args.device)
    env, _ = task_registry.make_env(args.task, args=sim_args, env_cfg=env_cfg)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(args.onnx, providers=providers)

    obs = env.get_observations()
    for step in range(args.steps):
        obs_input = obs.cpu().numpy()
        action = session.run(None, {"obs": obs_input})[0]
        action_tensor = torch.from_numpy(action).to(env.device)
        obs, _, _, resets, extras = env.step(action_tensor)
        if (step + 1) % 200 == 0:
            mean_height = env.root_states[:, 2].mean().item()
            print(
                f"Step {step+1}: mean base height {mean_height:.3f}, resets {int(resets.sum().item())}"
            )


if __name__ == "__main__":
    main()
