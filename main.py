from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
# from envs import create_atari_env
from envs_hm import make_atari, wrap_deepmind

from model import ActorCritic
from test import test
from train import train



# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
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
parser.add_argument('--num-processes', type=int, default=12,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongDeterministic-v4)') #SpaceInvadersNoFrameskip-v4
parser.add_argument('--no-shared', default=True,
                    help='use an optimizer without shared momentum.')


if __name__ == '__main__':
    # wandb.init(project='Pong_NoGAE_4frame+LSTM', entity = "polcom",name='test1_Pong_NoGAE_4frame+LSTM', config=None, sync_tensorboard=True, settings=wandb.Settings(start_method='thread', console="off"))
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()
    print('main')

    torch.manual_seed(args.seed)
    env = make_atari(args.env_name,render_mode= None)
    env = wrap_deepmind(env)
    print('main')
    

    
    img_h, img_w, img_c = env.observation_space.shape
    state_size = [1 * img_c, img_h, img_w]
    print('main', state_size)
    
    
    shared_model = ActorCritic(state_size[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        pass
        # optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        # optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
