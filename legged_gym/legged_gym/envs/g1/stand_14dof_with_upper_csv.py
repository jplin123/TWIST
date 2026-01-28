import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from .stand_14dof_with_upper_csv_config import Stand14DofWithUpperCSVCfg
from .upper_body_csv_player import UpperBodyCSVPlayer


class Stand14DofWithUpperCSV(LeggedRobot):
    def __init__(self, cfg: Stand14DofWithUpperCSVCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self._setup_joint_groups()
        csv_cfg = self.cfg.upper_body_csv
        self.upper_body_player = UpperBodyCSVPlayer(
            file_path=csv_cfg.file,
            joint_mapping=csv_cfg.joint_columns,
            playback_rate_hz=csv_cfg.playback_rate_hz,
            default_pose=self.cfg.init_state.default_joint_angles,
            loop=csv_cfg.loop,
            interpolate=csv_cfg.interpolate,
            device=self.device,
        )
        self.upper_body_targets = torch.zeros(
            (self.num_envs, len(self.upper_body_dof_indices)), device=self.device
        )
        self.upper_body_time = torch.zeros(self.num_envs, device=self.device)
        self.csv_amplitude = torch.full(
            (self.num_envs,), csv_cfg.default_amplitude, device=self.device
        )
        self.csv_speed = torch.full(
            (self.num_envs,), csv_cfg.default_speed, device=self.device
        )
        self.push_enabled = False
        self.pd_targets = torch.zeros_like(self.default_dof_pos_all)
        self.reach_goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self._update_upper_body_targets(0.0)

    def _setup_joint_groups(self):
        name_to_index = {name: idx for idx, name in enumerate(self.dof_names)}
        missing = [n for n in self.cfg.env.rl_dof_names if n not in name_to_index]
        if missing:
            raise ValueError(f"RL joint(s) {missing} not found in asset DOFs.")
        csv_names = list(self.cfg.upper_body_csv.joint_columns.keys())
        missing_csv = [n for n in csv_names if n not in name_to_index]
        if missing_csv:
            raise ValueError(f"CSV joint(s) {missing_csv} not found in asset DOFs.")
        self.rl_dof_indices = torch.tensor(
            [name_to_index[name] for name in self.cfg.env.rl_dof_names],
            device=self.device,
            dtype=torch.long,
        )
        self.upper_body_dof_indices = torch.tensor(
            [name_to_index[name] for name in csv_names],
            device=self.device,
            dtype=torch.long,
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(cfg.env.num_observations, device=self.device)
        idx = 0
        # projected gravity
        noise_vec[idx : idx + 3] = cfg.noise.noise_scales.gravity * cfg.noise.noise_level
        idx += 3
        # base ang vel
        noise_vec[idx : idx + 3] = cfg.noise.noise_scales.ang_vel * cfg.noise.noise_level
        idx += 3
        # base lin vel
        noise_vec[idx : idx + 3] = cfg.noise.noise_scales.lin_vel * cfg.noise.noise_level
        idx += 3
        # base height (no noise)
        idx += 1
        # joint pos
        pos_dim = len(self.cfg.env.rl_dof_names)
        noise_vec[idx : idx + pos_dim] = cfg.noise.noise_scales.dof_pos * cfg.noise.noise_level
        idx += pos_dim
        # joint vel
        noise_vec[idx : idx + pos_dim] = cfg.noise.noise_scales.dof_vel * cfg.noise.noise_level
        idx += pos_dim
        # previous action
        idx += pos_dim
        # upper body obs (leave noise off)
        return noise_vec

    def _post_physics_step_callback(self):
        self._update_curriculum()
        if (
            self.cfg.domain_rand.push_robots
            and self.push_enabled
            and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0)
        ):
            super()._push_robots()

    def _update_curriculum(self):
        csv_cfg = self.cfg.upper_body_csv
        cur_cfg = self.cfg.curriculum
        if not cur_cfg.enable:
            self.csv_amplitude[:] = csv_cfg.max_amplitude
            self.csv_speed[:] = csv_cfg.default_speed
            self.push_enabled = True
            return
        steps = float(self.total_env_steps_counter)
        amp_prog = max(0.0, (steps - cur_cfg.amp_warmup_steps) / max(cur_cfg.csv_motion_ramp_iters, 1.0))
        amp_prog = min(1.0, amp_prog)
        speed_prog = min(1.0, steps / max(cur_cfg.csv_speed_ramp_iters, 1.0))
        target_amp = csv_cfg.default_amplitude + (csv_cfg.max_amplitude - csv_cfg.default_amplitude) * amp_prog
        target_speed = csv_cfg.default_speed + (csv_cfg.max_speed - csv_cfg.default_speed) * speed_prog
        self.csv_amplitude[:] = target_amp
        self.csv_speed[:] = target_speed
        self.push_enabled = steps >= cur_cfg.push_start_iters

    def _update_upper_body_targets(self, delta_time: float):
        self.upper_body_time += delta_time
        targets = self.upper_body_player.sample(
            time_s=self.upper_body_time,
            amplitude=self.csv_amplitude,
            speed_scale=self.csv_speed,
        )
        self.upper_body_targets[:] = targets

    def _compute_torques(self, actions):
        self._update_upper_body_targets(self.sim_params.dt)
        self.pd_targets[:] = self.default_dof_pos_all
        self.pd_targets[:, self.rl_dof_indices] = (
            self.default_dof_pos_all[:, self.rl_dof_indices]
            + actions * self.cfg.control.action_scale
        )
        self.pd_targets[:, self.upper_body_dof_indices] = self.upper_body_targets
        pos_err = self.pd_targets - self.dof_pos
        if not self.cfg.domain_rand.randomize_motor:
            torques = self.p_gains * pos_err - self.d_gains * self.dof_vel
        else:
            torques = (
                self.motor_strength[0] * self.p_gains * pos_err
                - self.motor_strength[1] * self.d_gains * self.dof_vel
            )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        lower_pos = self.dof_pos[:, self.rl_dof_indices] - self.default_dof_pos_all[:, self.rl_dof_indices]
        lower_vel = self.dof_vel[:, self.rl_dof_indices]
        obs_parts = [
            self.projected_gravity,
            self.base_ang_vel,
            self.base_lin_vel,
            self.root_states[:, 2:3],
            lower_pos,
            lower_vel,
            self.last_actions,
        ]
        if self.cfg.env.include_upper_body_obs:
            obs_parts.append(self.upper_body_targets)
        self.obs_buf = torch.cat(obs_parts, dim=-1)
        if self.cfg.noise.add_noise:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        csv_cfg = self.cfg.upper_body_csv
        self.upper_body_time[env_ids] = 0.0
        if csv_cfg.random_start:
            rand_time = torch_rand_float(
                0.0,
                float(self.upper_body_player.total_duration),
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(-1)
            self.upper_body_time[env_ids] = rand_time
        self._update_upper_body_targets(0.0)

    def check_termination(self):
        tilt = torch.linalg.norm(self.projected_gravity[:, :2], dim=1)
        low_height = self.root_states[:, 2] < self.cfg.env.min_base_height
        tilt_fail = tilt > self.cfg.env.terminate_tilt
        contact_fail = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = low_height | tilt_fail | contact_fail | self.time_out_buf
        self.delta_yaw = torch.zeros_like(tilt)

    def _reward_upright(self):
        tilt = torch.linalg.norm(self.projected_gravity[:, :2], dim=1)
        return torch.exp(-tilt / (self.cfg.env.terminate_tilt + 1e-6))

    def _reward_base_height(self):
        height = self.root_states[:, 2]
        target = self.cfg.rewards.base_height_target
        return torch.exp(-torch.square(height - target))

    def _reward_low_lin_vel(self):
        return torch.exp(-torch.linalg.norm(self.base_lin_vel[:, :2], dim=1))

    def _reward_low_ang_vel(self):
        return torch.exp(-torch.linalg.norm(self.base_ang_vel[:, :2], dim=1))

    def _reward_contact_balance(self):
        if self.feet_indices.numel() < 2:
            return torch.ones(self.num_envs, device=self.device)
        contact_forces = torch.abs(self.contact_forces[:, self.feet_indices, 2])
        if contact_forces.ndim == 2:
            contact_forces = contact_forces.unsqueeze(-1)
        contact_forces = contact_forces.view(self.num_envs, -1)
        if contact_forces.shape[1] < 2:
            return torch.ones(self.num_envs, device=self.device)
        total = torch.sum(contact_forces[:, :2], dim=1) + 1e-3
        diff = torch.abs(contact_forces[:, 0] - contact_forces[:, 1])
        balance = 1.0 - diff / total
        return torch.clamp(balance, 0.0, 1.0)
