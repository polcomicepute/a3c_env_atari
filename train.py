import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# from envs import create_atari_env
from envs_hm import make_atari,wrap_deepmind

from model import ActorCritic
from torch.utils.tensorboard import SummaryWriter 


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        # print('None')
        


def train(rank, args, shared_model, counter, lock, optimizer=None):
    if rank==0:
        writer = SummaryWriter('./log_dir')

    
    torch.manual_seed(args.seed + rank)
    

    env = make_atari(args.env_name, render_mode=None)
    env = wrap_deepmind(env)
    env.seed(args.seed + rank)
    
    img_h, img_w, img_c = env.observation_space.shape
    state_size = [1*img_c, img_h, img_w]
    # print('train', state_size)

    model = ActorCritic(state_size[0], env.action_space)

    # if optimizer is None:
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 0.95 ** step, last_epoch=-1,
    #                             verbose=True)#


    model.train()

    state = env.reset()
    state = torch.from_numpy(state.transpose(2, 0, 1)/255).float()
    done = True

    episode_length = 0
    while True:
        
        model.load_state_dict(shared_model.state_dict())
        
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []
        # print('train')

        for step in range(args.num_steps):
            # print('train')
            
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach() #!왜 랜덤하게 샘플링하지? max값이 아니고
            #! num sample개의 인덱스 나옴 
            log_prob = log_prob.gather(1, action) #! 해당 action의 인덱스를 logprobd에서 가져와 

            #! perform action, get next state and reward
            # print('action',action[0].numpy())
            
            state, reward, done, _ = env.step(int(action[0]))
            # env.render()
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state.transpose(2, 0, 1)/255).float()

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            policy_loss = policy_loss - \
                log_probs[i] * advantage.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()

        ensure_shared_grads(model, shared_model)
        optimizer.step()
            
        
        if rank==0:
            writer.add_scalar('LR/train', get_lr(optimizer), counter.value)
        
        if counter.value > 5000000: #! Done 안에 넣어야하나 ? 
            break
        
        
            
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']