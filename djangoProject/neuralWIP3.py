import random

import numpy as np

from . import RL


def resetState(x=np.zeros(400, dtype=int)):
    # print("reset state", x)
    return 0


def periodicMove(area, gamer):
    if gamer:
        x = -1
    else:
        x = 1
    for i in range(len(area)):
        if not (area[i]):
            area[i] = x
            break
    return area, i


def randomMove(area, gamer):
    # for i in range(len(area)):
    if gamer:
        x = -1
    else:
        x = 1
    index_agent0 = get_index_of(area, 0)
    i = random.choice(index_agent0)
    area[i] = x
    return area, i


def addMove(_area, gamer, award: int, winner: int, aivsai: bool):
    area, i = randomMove(_area, gamer)
    if aivsai:
        areaAi, pos = randomMove(area, not gamer)
        return {"area": area.tolist(), "possition": i, "gamer": gamer, "aivsai": aivsai, "areaai": areaAi.tolist(),
                "pos": pos}
    # return learn(area, gamer, award, winner)
    return {"area": area.tolist(), "possition": i, "gamer": gamer, "aivsai": aivsai}


def get_index_of(lst, element):
    return list(map(lambda x: x[0], (list(filter(lambda x: x[1] == element, enumerate(lst))))))


import torch, numpy as np
from torch import nn

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

def seed(self, seed):
    np.random.seed(seed)

state_shape = 400
action_shape = 400
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
import tianshou as ts
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

import envpool
train_envs = envpool.make_gymnasium("CartPole-v0", num_envs=10)
test_envs = envpool.make_gymnasium("CartPole-v0", num_envs=100)

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

# import tianshou as ts
# from torch.utils.tensorboard import SummaryWriter
# from tianshou.utils.net.common import Net
#
# lr, epoch, batch_size = 1e-3, 10, 64
# train_num, test_num = 10, 100
# gamma, n_step, target_freq = 0.9, 3, 320
# buffer_size = 20000
# eps_train, eps_test = 0.1, 0.05
# step_per_epoch, step_per_collect = 10000, 10
# logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))

# def train_agent(
#         args: RL.argparse.Namespace = RL.get_args(),
#         agent_learn: RL.Optional[RL.BasePolicy] = None,
#         agent_opponent: RL.Optional[RL.BasePolicy] = None,
#         optim: RL.Optional[RL.torch.optim.Optimizer] = None,
# ) -> RL.Tuple[dict, RL.BRL.asePolicy]:
#     # ======== environment setup =========
#     train_envs = RL.DummyVectorEnv([RL.get_env for _ in range(args.training_num)])
#     test_envs = RL.DummyVectorEnv([RL.get_env for _ in range(args.test_num)])
#     # seed
#     np.random.seed(args.seed)
#     RL.torch.manual_seed(args.seed)
#     train_envs.seed(args.seed)
#     test_envs.seed(args.seed)
#
#     # ======== agent setup =========
#     policy, optim, agents = RL.get_agents(
#         args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
#     )
#
#     # ======== collector setup =========
#     train_collector = RL.Collector(
#         policy,
#         train_envs,
#         RL.VectorReplayBuffer(args.buffer_size, len(train_envs)),
#         exploration_noise=True
#     )
#     test_collector = RL.Collector(policy, test_envs, exploration_noise=True)
#     # policy.set_eps(1)
#     train_collector.collect(n_step=args.batch_size * args.training_num)
#
#     # ======== tensorboard logging setup =========
#     log_path = RL.os.path.join(args.logdir, 'tic_tac_toe', 'dqn')
#     writer = RL.SummaryWriter(log_path)
#     writer.add_text("args", str(args))
#     logger = RL.TensorboardLogger(writer)
#
#     # ======== callback functions used during training =========
#     def save_best_fn(policy):
#         if hasattr(args, 'model_save_path'):
#             model_save_path = args.model_save_path
#         else:
#             model_save_path = RL.os.path.join(
#                 args.logdir, 'tic_tac_toe', 'dqn', 'policy.pth'
#             )
#         RL.torch.save(
#             policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
#         )
#
#     def stop_fn(mean_rewards):
#         return mean_rewards >= args.win_rate
#
#     def train_fn(epoch, env_step):
#         policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)
#
#     def test_fn(epoch, env_step):
#         policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
#
#     def reward_metric(rews):
#         return rews[:, args.agent_id - 1]
#
#     # trainer
#     result = RL.offpolicy_trainer(
#         policy,
#         train_collector,
#         test_collector,
#         args.epoch,
#         args.step_per_epoch,
#         args.step_per_collect,
#         args.test_num,
#         args.batch_size,
#         train_fn=train_fn,
#         test_fn=test_fn,
#         stop_fn=stop_fn,
#         save_best_fn=save_best_fn,
#         update_per_step=args.update_per_step,
#         logger=logger,
#         test_in_train=False,
#         reward_metric=reward_metric
#     )
#
#     return result, policy.policies[agents[args.agent_id - 1]]
#
#
# # ======== a test function that tests a pre-trained agent ======
# def watch(
#         args: RL.argparse.Namespace = RL.get_args(),
#         agent_learn: RL.Optional[RL.BasePolicy] = None,
#         agent_opponent: RL.Optional[RL.BasePolicy] = None,
# ) -> None:
#     env = RL.get_env(render_mode="human")
#     env = RL.DummyVectorEnv([lambda: env])
#     policy, optim, agents = RL.get_agents(
#         args, agent_learn=agent_learn, agent_opponent=agent_opponent
#     )
#     policy.eval()
#     policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
#     collector = RL.Collector(policy, env, exploration_noise=True)
#     result = collector.collect(n_episode=1, render=args.render)
#     rews, lens = result["rews"], result["lens"]
#     print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")
#
#
#
# # train the agent and watch its performance in a match!
# args = RL.get_args()
# result, agent = train_agent(args)
# # watch(args, agent)
