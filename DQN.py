import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
class ReplayBuffer:
    "经验回放池"
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity) #队列，先进先出
    def __add__(self, state,action,reward,next_state,done): #将数据加入buffer
        self.buffer.append((state,action,reward,next_state,done))

    def __sample__(self,batch_size):  #从buffer采样数据，数量为batch_size
        transitions = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, done = zip(*transitions)

    def size(self):
        return len(self.buffer) #目前buffer中的数据的数量

class Qnet(torch.nn.Module):
    "只有"

