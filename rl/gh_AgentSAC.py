import numpy as np
import torch
from torch import Tensor
from copy import deepcopy
from typing import List, Tuple
from rl.gh_net import ActorSAC, CriticTwin
from rl.gh_AgentBase import AgentBase
from rl.gh_config import Arguments
#from elegantrl.agents.net import ActorSAC, ActorFixSAC, CriticTwin, ShareSPG
#from elegantrl.agents.AgentBase import AgentBase
#from elegantrl.train.config import Arguments


class AgentSAC(AgentBase):
    def __init__(self, action_dim, state_dim, actor_args: dict, critic_args: dict, gpu = False, args: Arguments = None):
        self.if_off_policy = True
        self.act_class = getattr(args, 'act_class', ActorSAC)
        self.cri_class = getattr(args, 'cri_class', CriticTwin)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        super().__init__(action_dim, state_dim, actor_args, critic_args, gpu, args)

        self.alpha_log = torch.tensor(
            (-np.log(action_dim),), dtype=torch.float32, requires_grad=True, device=self.device
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=0.005)
        self.target_entropy = getattr(args, 'target_entropy', 0.5 * -action_dim)

    def update_net(self, buffer):
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)

        for _ in range(int(self.repeat_times)):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            a_noise_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (-log_prob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)

            q_value_pg = self.cri.get_q_min(state, a_noise_pg)
            obj_actor = (q_value_pg - log_prob * alpha).mean() #-(q_value_pg - log_prob * alpha).mean() ## should be positive??
            self.optimizer_update(self.act_optimizer, obj_actor)
            if self.if_act_target:
                self.soft_update(self.act_target, self.act, self.soft_update_tau)

        return obj_critic.item(), -obj_actor.item(), self.alpha_log.exp().detach().item()

    def get_obj_critic(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            reward, mask, action, state, next_state = buffer.sample_batch(batch_size)  ## state, action, reward, next_state, done = buffer.sample_batch(batch_size)
            #mask = (1 - done) * self.gamma

            next_action, next_logprob = self.act_target.get_action_logprob(next_state)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_state, next_action)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q - next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2
        return obj_critic, state