import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
import nn.functional as F
from agent_dir.agent import Agent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PGNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PGNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        ##################
        pass

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        x = F.relu(self.fc1(inputs))
        x = F.softmax(self.fc2(x),dim = 1)
        ##################
        pass

class ReplayBuffer:
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        self.buffer_size = buffer_size
        self.buffer = []
        ##################
        # pass

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        return len(self.buffer)
        ##################
        # pass

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        if len(self.buffer) == self.buffer_size: #buffer not full
            self.buffer.pop(0)
        self.buffer.append(transition)
        ##################
        # pass

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        index = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i][0] for i in index]
        return zip(*batch)
        ##################
        # pass

    def sampleall(self):#sample all experience
        ##################
        # YOUR CODE HERE #
        index = np.range(0,len(self.buffer))
        batch = [self.buffer[i][0] for i in index]
        return zip(*batch)
        ##################
        # pass

    def clean(self):
        ##################
        # YOUR CODE HERE #
        self.buffer.clear()
        ##################
        # pass

class AgentPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentPG, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        self.env = env
        self.h = self.env.observation_space.shape[0]  # observation_dim,input size
        self.w = self.env.observation_space.shape[1]
        self.c = self.env.observation_space.shape[2]
        self.max_episode = args.max_episode
        self.action_dim = self.env.action_space.n  # output_size
        self.policy_net = PGNetwork(self.h, args.hidden_size,
                                    self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=args.lr)  # 使用Adam优化器
        self.gamma = args.gamma  # 折扣因子
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        obs, actions, rewards, _ , _ = self.replay_buffer.sample(self.replay_buffer.__len__())
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(device)

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(rewards.size)):  # 从最后一步算起
            reward = rewards[i]
            state = torch.tensor([obs[i]],
                                 dtype=torch.float).to(device)
            action = torch.tensor([actions[i]]).view(-1, 1).to(device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
        ##################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        # YOUR CODE HERE #
        obs = torch.tensor([observation], dtype=torch.float).to(self.device)
        probs = self.policy_net(obs)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
        ##################
        # pass

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        return_list = []
        for i_episode in range(int(self.max_episodes / 10)):
            episode_return = 0

            obs = self.env.reset()
            done = False
            while not done:
                action = self.make_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.store_transition = (obs, action, reward, next_obs, done)  # 存储记忆
                self.replay_buffer.push(self.store_transition)
                obs = next_obs
                episode_return += reward
            return_list.append(episode_return)
            self.train()
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
        ##################
        pass
