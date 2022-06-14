import os
from envs import create_atari_env
from model import ActorCritic
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train
import argparse

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')






def run(rank):
    print("Hi, I'm", rank)
    

if __name__ == "__main__":
    size = 4
    processes = []
    # os.environ['OMP_NUM_THREADS'] = '4'
    # os.environ['CUDA_VISIBLE_DEVICES'] = "" #! 
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name)
    
    shared_model= ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()
    
    
    for rank in range(size):
        # mp.spawn(run, args=(rank,), nprocs=size, join=True)
        
        p = mp.Process(target=train, args=(rank, args, shared_model))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()