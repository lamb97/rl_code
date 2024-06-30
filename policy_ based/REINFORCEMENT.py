import gym
import torch
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim, hidden_dim, action_dim):
        super(PolicyNet,self).init()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return F.softmax(self.fc2(x),dim=1)

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_bet = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_bet.parameters(), lr=learning_rate) #使用Adam优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):  #根据概率分布随即采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_bet(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  #从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.sensor([action_list[i]]).view(-1,1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1,action))
            G = self.gamma * G + reward
            loss = -log_prob * G #每一步的损失函数
            loss.backward() #反向传播计算梯度
        self.optimizer.step() #梯度下降

