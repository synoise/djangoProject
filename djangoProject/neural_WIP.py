# import math
# import random
#
# import torch
# import numpy as np
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
#
# from collections import namedtuple, deque
#
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))
#
#
# class ReplayMemory(object):
#
#     def __init__(self, capacity):
#         self.memory = deque([], maxlen=capacity)
#
#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# memory = ReplayMemory(10000)
#
#
# def resetState(x=np.zeros(400, dtype=int)):
#     print("reset state")
#     return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
#
#
# state = resetState()
#
#
# # def addMoveOLD(area, gamer, award: int, winner: int):
# # for i in range(len(area)):
# #     if not (area[i]):
# #         area[i] = 1
# #         break
# # return learn(area, gamer, award, winner)
# # return [area.tolist(), i, gamer]
#
#
# # def addTransition(area: np.ndarray, gamer, award: int, winner: int):
# #     if not winner:
# #         addMove(area, gamer, award, winner)
# #     else:
# #         resetState()
#
#
# def addMove(area, gamer, award, winner):
#     state = torch.tensor(area, dtype=torch.float32, device=device).unsqueeze(0)
#     action = select_action(state)
#     # observation, reward, terminated, truncated, _ = env.step(action.item())
#     observation = area
#     reward = torch.tensor([award], device=device)
#     next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
#     memory.push(state, action, next_state, reward)
#     state = next_state
#     area[steps_done] = 1
#     return [area.tolist(),1, gamer]
#
#
# class DQN(nn.Module):
#
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)
#
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)
#
#
# BATCH_SIZE = 128
# GAMMA = 0.99
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 1000
# TAU = 0.005
# LR = 1e-4
#
# print(22,state)
#
# n_observations = len(state)
# n_actions = 2
#
# policy_net = DQN(n_observations, n_actions).to(device)
#
# steps_done = 0
#
#
# def select_action(state):
#     global steps_done
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#                     math.exp(-1. * steps_done / EPS_DECAY)
#     steps_done += 1
#     if sample > eps_threshold:
#         with torch.no_grad():
#             # t.max(1) will return the largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return policy_net(state).max(1)[1].view(1, 1)
#     else:
#         return torch.tensor([[33]], device=device, dtype=torch.long)
