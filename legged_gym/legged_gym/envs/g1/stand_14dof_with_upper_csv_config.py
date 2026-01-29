from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Stand14DofWithUpperCSVCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_actions = 14
        include_upper_body_targets = True
        include_upper_body_obs = True
        upper_body_obs_dim = 9
        rl_dof_names = [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ]
        base_obs_dim = 3 + 3 + 3 + 1 + len(rl_dof_names) * 2 + len(rl_dof_names)
        num_observations = base_obs_dim + upper_body_obs_dim
        n_proprio = num_observations
        n_priv_latent = 0
        n_priv = 0
        num_privileged_obs = None
        history_len = 1
        history_encoding = True
        obs_type = "stand14"
        normalize_obs = True
        episode_length_s = 20.0
        send_timeouts = True
        contact_buf_len = 4
        include_foot_contacts = True
        randomize_start_pos = False
        randomize_start_yaw = False
        randomize_dof_pos = False
        min_base_height = 0.6
        terminate_tilt = 1.0
        terminate_on_fall = True

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.97,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.97,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {
            "hip_yaw": 100.0,
            "hip_roll": 100.0,
            "hip_pitch": 100.0,
            "knee": 150.0,
            "ankle": 40.0,
            "waist": 150.0,
            "shoulder": 40.0,
            "elbow": 40.0,
        }
        damping = {
            "hip_yaw": 2.,
            "hip_roll": 2.,
            "hip_pitch": 2.,
            "knee": 4.0,
            "ankle": 2.0,
            "waist": 4.0,
            "shoulder": 5.0,
            "elbow": 5.0,
        }
        action_scale = 0.5
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = f"{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision_with_fixed_hand.urdf"
        penalize_contacts_on = ["shoulder", "elbow"]
        terminate_after_contacts_on = ["torso_link"]
        fix_base_link = False
        disable_gravity = False
        collapse_fixed_joints = False
        feet_bodies = ["left_ankle_roll_link", "right_ankle_roll_link"]
        torso_name: str = "pelvis"
        chest_name: str = "imu_in_torso"
        upper_arm_name: str = "shoulder_roll_link"
        lower_arm_name: str = "elbow_link"
        thigh_name: str = "hip"
        shank_name: str = "knee"
        foot_name: str = "ankle_roll_link"
        waist_name: list = ["torso_link", "waist_roll_link", "waist_yaw_link"]
        hand_name: list = ["right_rubber_hand", "left_rubber_hand"]
        flip_visual_attachments = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False
        num_commands = 3
        resampling_time = 10.0

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.3]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
        randomize_base_com = True
        added_com_range = [-0.04, 0.04]
        push_robots = True
        push_interval_s = 6.0
        max_push_vel_xy = 0.4
        push_end_effector = False
        randomize_motor = True
        motor_strength_range = [0.9, 1.1]
        action_delay = False
        action_buf_len = 6

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            ang_vel = 1.5
            dof_pos = 1.0
            dof_vel = 0.05
            lin_vel = 1.0
        clip_observations = 5.0
        clip_actions = 2.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.5

        class noise_scales:
            ang_vel = 0.2
            lin_vel = 0.15
            gravity = 0.05
            dof_pos = 0.01
            dof_vel = 0.05

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        base_height_target = 1.0

        class scales:
            upright = 3.0
            base_height = 1.0
            low_lin_vel = 0.5
            low_ang_vel = 0.5
            contact_balance = 0.2
            action_rate = -0.05
            torques = -2.0e-4

    class viewer(LeggedRobotCfg.viewer):
        pos = [4.0, 0.0, 2.0]
        lookat = [0.0, 0.0, 0.9]

    class sim(LeggedRobotCfg.sim):
        dt = 0.002

    class upper_body_csv:
        file = f"{LEGGED_GYM_ROOT_DIR}/../assets/upper_body_csv/wave.csv"
        playback_rate_hz = 100.0
        loop = True
        interpolate = True
        random_start = True
        default_amplitude = 0.0
        max_amplitude = 0.5
        default_speed = 1.0
        max_speed = 1.0
        joint_columns = {
            "waist_yaw_joint": "waist_yaw_joint",
            "left_shoulder_pitch_joint": "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint": "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint": "left_shoulder_yaw_joint",
            "left_elbow_joint": "left_elbow_joint",
            "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint": "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint": "right_shoulder_yaw_joint",
            "right_elbow_joint": "right_elbow_joint",
        }

    class curriculum:
        enable = False
        csv_motion_ramp_iters = 10000
        csv_speed_ramp_iters = 10000
        push_start_iters = 20000
        amp_warmup_steps = 5000


class Stand14DofWithUpperCSVCfgPPO(LeggedRobotCfgPPO):
    seed = 7

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 256, 128]
        critic_hidden_dims = [256, 256, 128]
        init_noise_std = 0.8
        activation = "elu"
        tanh_encoder_output = False

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 2.5e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        normalizer_update_iterations = 3000

    class runner(LeggedRobotCfgPPO.runner):
        runner_class_name = "OnPolicyRunner"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 30000
        experiment_name = "stand_14dof_with_upper_csv"
        run_name = "baseline"
        save_interval = 200
