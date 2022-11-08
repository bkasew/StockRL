import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import numpy as np
import numpy.random as rd
from typing import Tuple
from collections import OrderedDict





"""Actor (policy network)"""


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, num_layer, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )
        self.net_a_avg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        )  # the average of action
        self.net_a_std = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        )  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        tmp = self.net_state(state.squeeze())
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net_state(state.squeeze())
        a_avg = self.net_a_avg(t_tmp).clamp(-2, 2)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()

        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        t_tmp = self.net_state(state.squeeze())
        a_avg = self.net_a_avg(t_tmp).clamp(-2, 2)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

        neg_logprob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # logprob = logprob + (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        epsilon = 1e-6
        logprob = -(neg_logprob + (1 - a_tan.pow(2) + epsilon).log()) ## fix neg_logbprob, then take negative to get log_prob
        return a_tan, logprob.sum(1, keepdim=True)  # todo negative logprob

# class ActorSAC(nn.Module):
#     def __init__(self, state_dim, layers, avg_layers, std_layers, action_dim):
#         super().__init__()

#         ## create OrderedDict for each NN; initialize
#         od_net_state = OrderedDict([("input_layer", nn.Linear(state_dim, layers[0])), ("input_relu", nn.ReLU())])
#         od_net_a_avg = OrderedDict([("input_layer", nn.Linear(layers[-1], avg_layers[0])), ("input_relu", nn.ReLU())])
#         od_net_a_std = OrderedDict([("input_layer", nn.Linear(layers[-1], std_layers[0])), ("input_relu", nn.ReLU())])

#         ## add middle layers to OrderedDict for each NN
#         for i in range(len(layers) - 1):
#             od_net_state["linear_" + str(i+1)] = nn.Linear(layers[i], layers[i+1])
#             od_net_state["relu_" + str(i+1)] = nn.ReLU()
#         for i in range(len(avg_layers) - 1):
#             od_net_a_avg["linear_" + str(i+1)] = nn.Linear(avg_layers[i], avg_layers[i+1])
#             od_net_a_avg["relu_" + str(i+1)] = nn.ReLU()
#         for i in range(len(std_layers) - 1):
#             od_net_a_std["linear_" + str(i+1)] = nn.Linear(std_layers[i], std_layers[i+1])
#             od_net_a_std["relu_" + str(i+1)] = nn.ReLU()

        
#         ## add final layer to OrdereDict for each NN
#         od_net_a_avg["output_layer"] = nn.Linear(avg_layers[-1], action_dim)
#         od_net_a_std["output_layer"] = nn.Linear(std_layers[-1], action_dim)

#         ## create NN's using OrderedDict's from above; create log_sqrt_2pi
#         self.net_state = nn.Sequential(od_net_state)
#         self.net_a_avg = nn.Sequential(od_net_a_avg) ## average of the action
#         self.net_a_std = nn.Sequential(od_net_a_std) ## log_std of the action
#         self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

#     def forward(self, state):
#         tmp = self.net_state(state.squeeze())
#         return self.net_a_avg(tmp).tanh()  # action

#     def get_action(self, state):
#         t_tmp = self.net_state(state.squeeze())
#         a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
#         a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()

#         return torch.normal(a_avg, a_std).tanh()  # re-parameterize

#     def get_action_logprob(self, state):
#         t_tmp = self.net_state(state.squeeze())
#         a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
#         a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
#         a_std = a_std_log.exp()

#         noise = torch.randn_like(a_avg, requires_grad=True)
#         a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

#         neg_logprob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
#         # logprob = logprob + (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
#         epsilon = 1e-6
#         logprob = -(neg_logprob + (1 - a_tan.pow(2) + epsilon).log()) ## fix neg_logbprob, then take negative to get log_prob
#         return a_tan, logprob.sum(1, keepdim=True)  # todo negative logprob


## NEW: ACTOR NET FOR GRU
class GRUNetActor(nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, hidden_dim3, mid_dim, action_dim, n_layers, drop_prob=0.2):
        super().__init__()

        ## set up GRU
        self.gru_0 = nn.GRU(state_dim, hidden_dim1, n_layers, batch_first=True, dropout=drop_prob)
        self.gru_1 = nn.GRU(hidden_dim1, hidden_dim2, n_layers, batch_first=True, dropout=drop_prob)
        self.gru_2 = nn.GRU(hidden_dim2, hidden_dim3, n_layers, batch_first=True, dropout=drop_prob)

        ## set up all other neural nets
        self.net_a_avg = nn.Sequential(
            nn.Linear(hidden_dim3, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        ) # the average of action
        self.net_a_std = nn.Sequential(
            nn.Linear(hidden_dim3, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        ) # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        out, h = self.gru_0(state)
        out, h = self.gru_1(out) #,h
        out, h = self.gru_2(out) #,h
        
        return self.net_a_avg(h[-1, :, :]).tanh()  # action
        
    def get_action(self, state):
        out, h = self.gru_0(state)
        out, h = self.gru_1(out) #,h
        out, h = self.gru_2(out) #,h

        a_avg = self.net_a_avg(h[-1, :, :])  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(h[-1, :, :]).clamp(-20, 2).exp()

        return torch.normal(a_avg, a_std).tanh()  # re-parameterize
    
    def get_action_logprob(self, state):
        out, h = self.gru_0(state)
        out, h = self.gru_1(out) #,h
        out, h = self.gru_2(out) #,h

        a_avg = self.net_a_avg(h[-1, :, :])  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(h[-1, :, :]).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

        neg_logprob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # logprob = logprob + (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        epsilon = 1e-6
        logprob = -(neg_logprob + (1 - a_tan.pow(2) + epsilon).log()) ## fix neg_logbprob, then take negative to get log_prob
        return a_tan, logprob.sum(1, keepdim=True)  # todo negative logprob





"""Critic (value network)"""


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, num_layer,  state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )  # concat(state, action)
        self.net_q1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )  # q2 value

    def forward(self, state, action):
        return torch.add(*self.get_q1_q2(state.squeeze(), action)) / 2.0  # mean Q value

    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state.squeeze(), action))  # min Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state.squeeze(), action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values

# class CriticTwin(nn.Module):  # shared parameter
#     def __init__(self, state_dim, action_dim, layers, q_layers):
#         super().__init__()
        
#         ## create OrderedDict for each NN; initialize
#         od_net_sa = OrderedDict([("input_layer", nn.Linear(state_dim + action_dim, layers[0])), ("input_relu", nn.ReLU())])
#         od_net_q = OrderedDict([("input_layer", nn.Linear(layers[-1], q_layers[0])), ("input_relu", nn.ReLU())])

#         ## add middle layers to OrderedDict for each NN
#         for i in range(len(layers) - 1):
#             od_net_sa["linear_" + str(i+1)] = nn.Linear(layers[i], layers[i+1])
#             od_net_sa["relu_" + str(i+1)] = nn.ReLU()
#         for i in range(len(q_layers) - 1):
#             od_net_q["linear_" + str(i+1)] = nn.Linear(q_layers[i], q_layers[i+1])
#             od_net_q["relu_" + str(i+1)] = nn.ReLU()

#         ## add final layer to OrdereDict for each NN
#         od_net_q["output_layer"] = nn.Linear(q_layers[-1], 1)

#         ## create NN's using OrderedDict's from above
#         self.net_sa = nn.Sequential(od_net_sa) # concat(state, action)
#         self.net_q1 = nn.Sequential(od_net_q) # q1 value
#         self.net_q2 = nn.Sequential(od_net_q) # q2 value

#     def forward(self, state, action):
#         return torch.add(*self.get_q1_q2(state.squeeze(), action)) / 2.0  # mean Q value

#     def get_q_min(self, state, action):
#         return torch.min(*self.get_q1_q2(state.squeeze(), action))  # min Q value

#     def get_q1_q2(self, state, action):
#         tmp = self.net_sa(torch.cat((state.squeeze(), action), dim=1))
#         return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


## NEW: CRITIC NET FOR GRU
class GRUNetCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim, mid_dim, n_layers, drop_prob=0.2):
        super().__init__()

        ## set up GRU for state
        self.gru_0 = nn.GRU(state_dim, hidden_dim1, n_layers, batch_first=True, dropout=drop_prob)
        self.gru_1 = nn.GRU(hidden_dim1, hidden_dim2, n_layers, batch_first=True, dropout=drop_prob)
        self.gru_2 = nn.GRU(hidden_dim2, hidden_dim3, n_layers, batch_first=True, dropout=drop_prob)
        
        ## set up Linear for action
        self.linear = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim), nn.ReLU()
        )

        ## set up Q-nets
        self.net_q1 = nn.Sequential(
            nn.Linear(hidden_dim3 + mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(hidden_dim3 + mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )  # q2 value
        
    def forward(self, state, action):
        return torch.add(*self.get_q1_q2(state, action)) / 2.0  # mean Q value
    
    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state, action))  # min Q value
    
    def get_q1_q2(self, state, action):
        ## get state values
        gru_out, h = self.gru_0(state)
        gru_out, h = self.gru_1(gru_out) #,h
        gru_out, h = self.gru_2(gru_out) #,h

        ## get action values
        linear_out = self.linear(action)

        ## concatenate to get Q values
        values_in = torch.cat((h[-1,:,:], linear_out), dim=1)
        return self.net_q1(values_in), self.net_q2(values_in)