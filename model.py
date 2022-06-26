import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class ActorCritic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=4, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        # print('0',inputs.shape)
        # print('1',hx.shape)
        # print('2',cx.shape)
        
        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        # print('3',x.shape)  

        x = x.view(-1, 32 * 3 * 3)
        # print('4',x.shape)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        # print('5',self.critic_linear(x).shape, self.actor_linear(x).shape, hx.shape, cx.shape)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
