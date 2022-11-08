import os
import torch
import numpy as np
from copy import deepcopy
from pprint import pprint

'''config for agent'''


class Arguments:
    def __init__(self, agent_class=None, actor_args: dict = None, critic_args: dict = None, env=None, env_func=None, env_args: dict = None, **kwargs):
        self.env = env  # the environment for training
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)
        self.actor_args = actor_args
        self.critic_args = critic_args if critic_args else actor_args
        if kwargs.get("lookback") is not None:
            self.lookback = kwargs.get("lookback")

        if kwargs.get("act_class") is not None:
            self.act_class = kwargs.get("act_class")
        if kwargs.get("cri_class") is not None:
            self.cri_class = kwargs.get("cri_class")

        self.env_num = self.update_attr('env_num')  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.update_attr('max_step')  # the max step of an episode
        self.env_name = self.update_attr('env_name')  # the env name. Be used to set 'cwd'.
        self.state_dim = self.update_attr('state_dim')  # vector dimension (feature number) of state
        self.action_dim = self.update_attr('action_dim')  # vector dimension (feature number) of action
        self.if_discrete = self.update_attr('if_discrete')  # discrete or continuous action space
        self.target_return = self.update_attr('target_return')  # target average episode return

        self.agent_class = agent_class  # the class of DRL algorithm
        self.net_dim = 2 ** 4  # the network width
        self.num_layer = 3  # layer number of MLP (Multi-layer perception, `assert layer_num>=2`)
        self.horizon_len = kwargs.get("horizon_len", 32)  # number of steps per exploration
        if self.if_off_policy:  # off-policy
            self.max_memo = int(kwargs.get("max_memo", 2 ** 21))  # capacity of replay buffer, 2 ** 21 ~= 2e6
            self.batch_size = kwargs.get("batch_size", self.net_dim)  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # epoch num
            self.if_use_per = False  # use PER (Prioritized Experience Replay) for sparse reward
            self.num_seed_steps = 2  # the total samples for warm-up is num_seed_steps * env_num * num_steps_per_episode
            self.num_steps_per_episode = 128
            self.n_step = 1  # multi-step TD learning
        else:  # on-policy
            self.max_memo = 2 ** 12  # capacity of replay buffer
            self.target_step = self.max_memo  # repeatedly update network to keep critic's loss small
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 4  # collect target_step, then update network
            self.if_use_gae = False  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.lambda_critic = 2 ** 0  # the objective coefficient of critic network
        self.learning_rate = 2 ** -15  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.clip_grad_norm = 3.0  # 0.1 ~ 4.0, clip the gradient after normalization
        self.if_off_policy = self.if_off_policy()  # agent is on-policy or off-policy
        self.if_use_old_traj = False  # save old data to splice and get a complete trajectory (for vector env)

        '''Arguments for device'''
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.gpu = kwargs.get("gpu", False)  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = kwargs.get("cwd", None)  # current working directory to save model. None means set automatically
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = kwargs.get("break_step", +np.inf)  # break training if 'total_step > break_step'
        self.if_over_write = False  # overwrite the best policy network (actor.pth)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        '''Arguments for evaluate'''
        self.save_gap = 2  # save the policy network (actor.pth) for learning curve, +np.inf means don't save
        self.eval_gap = 2 ** 4  # evaluate the agent per eval_gap seconds
        self.eval_steps = kwargs.get("eval_steps", self.horizon_len*500) ## how often to run Evaluator (after every eval_steps number of steps)
        self.eval_times = kwargs.get("eval_times", 4)  # number of times that get episode return
        self.eval_times2 = kwargs.get("eval_times", self.eval_times)
        self.eval_env_func = None  # eval_env = eval_env_func(*eval_env_args)
        self.eval_env_args = None  # eval_env = eval_env_func(*eval_env_args)

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        '''auto set'''
        if self.cwd is None:
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_{self.gpu}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def update_attr(self, attr: str):
        try:
            attribute_value = getattr(self.env, attr) if self.env_args is None else self.env_args[attr]
        except Exception as error:
            print(f"| Argument.update_attr() Error: {error}")
            attribute_value = None
        return attribute_value

    def if_off_policy(self) -> bool:
        name = self.agent_class.__name__
        if_off_policy = all((name.find('PPO') == -1, name.find('A2C') == -1))
        return if_off_policy

    def print(self):
        # prints out args in a neat, readable format
        pprint(vars(self))


def build_env(env=None, env_func=None, env_args=None):
    if env is not None:
        env = deepcopy(env)
    elif env_func.__module__ == 'gym.envs.registration':
        import gym
        gym.logger.set_level(40)  # Block warning
        env = env_func(id=env_args['env_name'])
    else:
        env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    for attr_str in ('state_dim', 'action_dim', 'max_step', 'if_discrete', 'target_return'):
        if (not hasattr(env, attr_str)) and (attr_str in env_args):
            setattr(env, attr_str, env_args[attr_str])
    return env

## create config class
class Config:
    def __init__(self):
        # parameters for data sources
        self.ALPACA_API_KEY = "PKDQULGJ1BBMJDLNO211"  # your ALPACA_API_KEY
        self.ALPACA_API_SECRET = "KC7eJ691Mok89tKAnKVguyOk03FWpdzwHAmAT0gZ"  # your ALPACA_API_SECRET
        self.ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url

        self.INDICATORS = [
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ]