import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F
from agent_dir.agent import Agent
from collections import namedtuple
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

MODEL_STORE_PATH = "/home/cyl/DQN/model"
modelname = 'DQN_Pong'
model_path = MODEL_STORE_PATH + '/' + 'model/' + 'DQN_Pong_episode900.pt'

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        # ##################
        # # YOUR CODE HERE #
        # """
        # Initialize Deep Q Network
        # Args:
        #     in_channels (int): number of input channels
        #     n_actions (int): number of outputs
        # """
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(hidden_size*2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(hidden_size*2)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, output_size)
        ##################
        # pass

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #

        # x = inputs[np.newaxis, :]
        # x = torch.Tensor(x)
        # x = x.float() / 255
        # x = x.to(device)

        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)
        ##################
        # pass


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

    def clean(self):
        ##################
        # YOUR CODE HERE #
        self.buffer.clear()
        ##################
        # pass


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        self.env = env
        self.h = self.env.observation_space.shape[0]#observation_dim,input size
        self.w = self.env.observation_space.shape[1]
        self.c = self.env.observation_space.shape[2]
        self.action_dim = self.env.action_space.n#output_size
        self.action_space = []
        self.hidden_size = args.hidden_size
        self.seed = args.seed
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.grad_norm_clip = args.grad_norm_clip
        self.max_episode = args.max_episode
        self.eps = args.eps
        self.eps_min = args.eps_min
        self.eps_decay = args.eps_decay
        self.eps_anneal = (self.eps - self.eps_min) / self.eps_decay
        self.update_target = args.update_target
        self.test = args.test
        self.use_cuda = args.use_cuda

        self.buffer_size = args.buffer_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        #构建网络
        self.eval_dqn = QNetwork(self.h, self.hidden_size, self.action_dim).to(device)
        self.target_dqn = QNetwork(self.h, self.hidden_size, self.action_dim).to(device)
        self.target_dqn.load_state_dict(self.eval_dqn.state_dict())
        self.parameters = list(self.eval_dqn.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
        
        ##################
        # pass
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(device)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(device)

        q_eval = self.eval_dqn(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_dqn(next_obs).detach()

        q_target = rewards + self.gamma * (1-dones) * torch.max(q_next, dim = -1)[0]
        Loss = self.loss_fn(q_eval, q_target)
        self.optim.zero_grad()
        Loss.backward()
        self.optim.step()
        return Loss.item()
        ##################
        # pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        if np.random.uniform() <= self.eps:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
            action_value = self.eval_dqn(observation)
            action = torch.max(action_value, dim = -1)[1].cpu().numpy()
        return int(action)
        ##################
        # pass

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        step = 0 #记录现在走到第几步了
        for i_episode in range(self.max_episode):
            obs = self.env.reset() #获得初始观测值
            episode_reward = 0
            done = False
            loss = 0
            while not done:
                loss_ = []
                action = self.make_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.store_transition = (obs, action, reward, next_obs, done) #存储记忆
                self.replay_buffer.push(self.store_transition)
                episode_reward += reward
                obs = next_obs
                if step >= self.batch_size*300:
                    loss_.append(self.train())
                if done:
                    if loss_.__len__():
                        loss = sum(loss_) / loss_.__len__()
                    break
                
                step += 1
                
                if self.eps > self.eps_min:
                    self.eps -= self.eps_anneal
                else:
                    self.eps = self.eps_min
               
                if step % self.update_target == 0:
                    self.target_dqn.load_state_dict(self.eval_dqn.state_dict())
            
            print("step:", step, "episode:", i_episode, "epsilon:",  self.eps, "loss:", loss, "reward:", episode_reward)

        ##################
        # pass
