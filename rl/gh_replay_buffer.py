import os
from typing import List, Tuple, Union
import numpy as np
import numpy.random as rd
import torch
from torch import Tensor
from rl.gh_config import Arguments
#from elegantrl.train.config import Arguments



class ReplayBuffer:  # for off-policy
    def __init__(self, max_capacity: int, state_dim: int, action_dim: int, gpu=False, if_use_per=False, args:Arguments=None):
        self.prev_p = 0  # previous pointer
        self.next_p = 0  # next pointer
        self.if_full = False
        self.cur_capacity = 0  # current capacity
        self.max_capacity = max_capacity
        self.add_capacity = 0  # update in self.update_buffer

        #self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.device = torch.device("mps" if gpu else "cpu")

        self.buf_action = torch.empty((max_capacity, action_dim), dtype=torch.float32, device=self.device)
        self.buf_reward = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)
        self.buf_mask = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)

        if args.env.lookback >= 1:
            buf_state_size = (self.max_capacity, args.env.lookback, state_dim) if isinstance(state_dim, int) else (max_capacity, *state_dim)
        else:
            buf_state_size = (max_capacity, state_dim) if isinstance(state_dim, int) else (max_capacity, *state_dim)
        self.buf_state = torch.empty(buf_state_size, dtype=torch.float32, device=self.device)

        self.if_use_per = if_use_per
        if if_use_per:
            self.per_tree = BinarySearchTree(max_capacity)
            self.sample_batch = self.sample_batch_per

    def update_buffer(self, traj_list: List[List]):
        #traj_items = list(map(list, zip(*traj_list)))

        #states, rewards, masks, actions = [torch.cat(item, dim=0) for item in traj_items]
        states, rewards, masks, actions = [item for item in traj_list]
        self.add_capacity = rewards.shape[0]
        p = self.next_p + self.add_capacity  # update pointer

        if self.if_use_per:
            self.per_tree.update_ids(data_ids=np.arange(self.next_p, p) % self.max_capacity)

        if p > self.max_capacity:
            self.buf_state[self.next_p:self.max_capacity] = states[:self.max_capacity - self.next_p]
            self.buf_reward[self.next_p:self.max_capacity] = rewards[:self.max_capacity - self.next_p]
            self.buf_mask[self.next_p:self.max_capacity] = masks[:self.max_capacity - self.next_p]
            self.buf_action[self.next_p:self.max_capacity] = actions[:self.max_capacity - self.next_p]
            self.if_full = True

            p = p - self.max_capacity
            self.buf_state[0:p] = states[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_mask[0:p] = masks[-p:]
            self.buf_action[0:p] = actions[-p:]
        else:
            self.buf_state[self.next_p:p] = states
            self.buf_reward[self.next_p:p] = rewards
            self.buf_mask[self.next_p:p] = masks
            self.buf_action[self.next_p:p] = actions

        self.next_p = p  # update pointer
        self.cur_capacity = self.max_capacity if self.if_full else self.next_p

        steps = rewards.shape[0]
        r_exp = rewards.mean().item()
        return steps, r_exp

    def sample_batch(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        indices = torch.randint(self.cur_capacity - 1, size=(batch_size,), device=self.device)

        '''replace indices using the latest sample'''
        i1 = self.next_p
        i0 = self.next_p - self.add_capacity
        num_new_indices = 1
        new_indices = torch.randint(i0, i1, size=(num_new_indices,)) % (self.max_capacity - 1)
        indices[0:num_new_indices] = new_indices

        return (
            self.buf_reward[indices],
            self.buf_mask[indices],
            self.buf_action[indices],
            self.buf_state[indices],
            self.buf_state[indices + 1]  # next state
        )

    def sample_batch_per(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        beg = -self.max_capacity
        end = (self.cur_capacity - self.max_capacity) if (self.cur_capacity < self.max_capacity) else None

        indices, is_weights = self.per_tree.get_indices_is_weights(batch_size, beg, end)

        return (
            self.buf_reward[indices],
            self.buf_mask[indices],
            self.buf_action[indices],
            self.buf_state[indices],
            self.buf_state[indices + 1],  # next state
            torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)  # important sampling weights
        )

    def td_error_update(self, td_error: Tensor):
        self.per_tree.td_error_update(td_error)

    def save_or_load_history(self, cwd: str, if_save: bool):
        obj_names = (
            (self.buf_reward, "reward"),
            (self.buf_mask, "mask"),
            (self.buf_action, "action"),
            (self.buf_state, "state"),
        )

        if if_save:
            print(f"| {self.__class__.__name__}: Saving in cwd {cwd}")
            for obj, name in obj_names:
                if self.cur_capacity == self.next_p:
                    buf_tensor = obj[:self.cur_capacity]
                else:
                    buf_tensor = torch.vstack((obj[self.next_p:self.cur_capacity], obj[0:self.next_p]))

                torch.save(buf_tensor, f"{cwd}/replay_buffer_{name}.pt")

            print(f"| {self.__class__.__name__}: Saved in cwd {cwd}")

        elif os.path.isfile(f"{cwd}/replay_buffer_state.pt"):
            print(f"| {self.__class__.__name__}: Loading from cwd {cwd}")
            buf_capacity = 0
            for obj, name in obj_names:
                buf_tensor = torch.load(f"{cwd}/replay_buffer_{name}.pt")
                buf_capacity = buf_tensor.shape[0]

                obj[:buf_capacity] = buf_tensor
            self.cur_capacity = buf_capacity

            print(f"| {self.__class__.__name__}: Loaded from cwd {cwd}")

    def concatenate_state(self) -> Tensor:
        if self.prev_p <= self.next_p:
            buf_state = self.buf_state[self.prev_p:self.next_p]
        else:
            buf_state = torch.vstack((self.buf_state[self.prev_p:], self.buf_state[:self.next_p],))
        self.prev_p = self.next_p
        return buf_state

    def concatenate_buffer(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.prev_p <= self.next_p:
            buf_state = self.buf_state[self.prev_p:self.next_p]
            buf_action = self.buf_action[self.prev_p:self.next_p]
            buf_reward = self.buf_reward[self.prev_p:self.next_p]
            buf_mask = self.buf_mask[self.prev_p:self.next_p]
        else:
            buf_state = torch.vstack((self.buf_state[self.prev_p:], self.buf_state[:self.next_p],))
            buf_action = torch.vstack((self.buf_action[self.prev_p:], self.buf_action[:self.next_p],))
            buf_reward = torch.vstack((self.buf_reward[self.prev_p:], self.buf_reward[:self.next_p],))
            buf_mask = torch.vstack((self.buf_mask[self.prev_p:], self.buf_mask[:self.next_p],))
        self.prev_p = self.next_p
        return buf_state, buf_action, buf_reward, buf_mask


class BinarySearchTree:
    """Binary Search Tree for PER
    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """
    def __init__(self, memo_len):
        self.memo_len = memo_len  # replay buffer len
        self.prob_ary = np.zeros((memo_len - 1) + memo_len)  # parent_nodes_num + leaf_nodes_num
        self.max_capacity = len(self.prob_ary)
        self.cur_capacity = self.memo_len - 1  # pointer
        self.indices = None
        self.depth = int(np.log2(self.max_capacity))

        # PER.  Prioritized Experience Replay. Section 4
        # alpha, beta = 0.7, 0.5 for rank-based variant
        # alpha, beta = 0.6, 0.4 for proportional variant
        self.per_alpha = 0.6  # alpha = (Uniform:0, Greedy:1)
        self.per_beta = 0.4  # beta = (PER:0, NotPER:1)

    def update_id(self, data_id, prob=10):  # 10 is max_prob
        tree_id = data_id + self.memo_len - 1
        if self.cur_capacity == tree_id:
            self.cur_capacity += 1

        delta = prob - self.prob_ary[tree_id]
        self.prob_ary[tree_id] = prob

        while tree_id != 0:  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.prob_ary[tree_id] += delta

    def update_ids(self, data_ids, prob=10):  # 10 is max_prob
        ids = data_ids + self.memo_len - 1
        self.cur_capacity += (ids >= self.cur_capacity).sum()

        upper_step = self.depth - 1
        self.prob_ary[ids] = prob  # here, ids means the indices of given children (maybe the right ones or left ones)
        p_ids = (ids - 1) // 2

        while upper_step:  # propagate the change through tree
            ids = p_ids * 2 + 1  # in this while loop, ids means the indices of the left children
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]
        # because we take depth-1 upper steps, ps_tree[0] need to be updated alone

    def get_leaf_id(self, v):
        """Tree structure and array storage:
        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node
            r_idx = l_idx + 1  # the leaf's right node
            if l_idx >= (len(self.prob_ary)):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx
                else:
                    v -= self.prob_ary[l_idx]
                    parent_idx = r_idx
        return min(leaf_idx, self.cur_capacity - 2)  # leaf_idx

    def get_indices_is_weights(self, batch_size, start, end):
        self.per_beta = min(1., self.per_beta + 0.001)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (self.prob_ary[0] / batch_size)

        # get proportional prioritization
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[start:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights
        return self.indices, is_weights

    def td_error_update(self, td_error):  # td_error = (q-q).detach_().abs()
        prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)
        prob = prob.cpu().numpy()
        self.update_ids(self.indices, prob)