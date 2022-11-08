import multiprocessing as mp
import os
import time
from typing import Union, Tuple

import numpy as np
import torch

from rl.gh_config import build_env
from rl.gh_evaluator import Evaluator
from rl.gh_replay_buffer import ReplayBuffer
from rl.gh_config import Arguments
#from elegantrl.train.config import build_env
#from elegantrl.train.evaluator import Evaluator
#from elegantrl.train.replay_buffer import ReplayBuffer, ReplayBufferList
#from elegantrl.train.config import Arguments

## initiate buffer
def init_buffer(args: Arguments, gpu: int) -> ReplayBuffer:
    buffer = ReplayBuffer(gpu=gpu,
                          max_capacity=args.max_memo,
                          state_dim=args.state_dim,
                          action_dim=1 if args.if_discrete else args.action_dim, args=args)
    buffer.save_or_load_history(args.cwd, if_save=False)
    return buffer

## initiate evaluator
def init_evaluator(args: Arguments, gpu: int) -> Evaluator:
    eval_func = args.eval_env_func if getattr(args, "eval_env_func") else args.env_func
    eval_args = args.eval_env_args if getattr(args, "eval_env_args") else args.env_args
    eval_env = build_env(args.env, eval_func, eval_args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu, eval_env=eval_env, args=args)
    return evaluator