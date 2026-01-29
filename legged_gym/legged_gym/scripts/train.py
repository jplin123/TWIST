# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry

import torch
import wandb

from legged_gym.scripts.play import apply_debug_init

def train(args):
    args.headless = True
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.exptid}" if args.exptid else timestamp
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + run_id
    try:
        os.makedirs(log_pth)
    except:
        pass
    
    disable_wandb = args.no_wandb or os.environ.get("WANDB_DISABLED", "true").lower() in ("1", "true", "yes", "on")
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 5
        args.num_envs = 32
        args.headless = False
    else:
        mode = "online"
    
    if disable_wandb:
        mode = "disabled"
        os.environ["WANDB_MODE"] = "disabled"
        
    robot_type = args.task.split("_")[0]
    
    if not disable_wandb:
        wandb_project = f"{robot_type}_mimic"
        wandb.init(project=wandb_project, name=run_id, mode=mode, dir="../../logs")
        if robot_type == "g1":
            wandb.save(LEGGED_GYM_ENVS_DIR + "/g1/g1_mimic_distill_config.py", policy="now")
    
    if args.debug_init:
        env_cfg, _ = task_registry.get_cfgs(name=args.task)
        apply_debug_init(env_cfg)
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    else:
        env, _ = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root=log_pth,
        env=env,
        name=args.task,
        args=args,
        init_wandb=not disable_wandb,
    )
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    

if __name__ == "__main__":
    args = get_args()
    train(args)
