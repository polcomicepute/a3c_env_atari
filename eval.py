import time
from collections import deque
import os
import torch
import torch.nn.functional as F
import numpy as np

# from envs import create_atari_env
from envs_hm import make_atari,wrap_deepmind

from model import ActorCritic
import wandb
wandb.login()
os.environ["WANDB_API_KEY"] = "41d70fa07c07644beca7655a81d7af49afccf2dd"

def eval(args, n_episodes):
    n_epi = 0
    
    # wandb.init(project=str(args.env_name) + '_NoGAE_1frame+LSTM', entity = "polcom",name='eval1_best_'+ str(args.env_name) +'_NoGAE_1frame+LSTM', config=None, sync_tensorboard=True, settings=wandb.Settings(start_method='thread', console="off"))
    wandb.init(project=str(args.env_name) + '_NoGAE_1frame+LSTM', entity = "polcom",name='eval1_'+ str(args.env_name) +'_NoGAE_1frame+LSTM', config=None, sync_tensorboard=True, settings=wandb.Settings(start_method='thread', console="off"))
    wandb.config.update(args)
    
    
    torch.manual_seed(args.seed)
    env = make_atari(args.env_name, render_mode='human')
    env = wrap_deepmind(env)
    env.seed(args.seed)
    
    img_h, img_w, img_c = env.observation_space.shape
    state_size = [1*img_c, img_h, img_w]
    print('test', state_size)

    model = ActorCritic(state_size[0], env.action_space)
    load(model,'gym-results-hm/','a3c_'+str(args.env_name)+'.pth.tar')
    # load(model,'gym-results-hm/','best_a3c_'+str(args.env_name)+'.pth.tar')
    
    wandb.watch(model)

    model.eval()

    state = env.reset()
    state = np.array(state,copy=False)
    state = torch.from_numpy(state.transpose(2, 0, 1)/255).float()

    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    
    best_reward =0
    ####
    all_rewards =[]


    # for i_ep in range(n_episodes):
    while True:
    
    # print('test' + str(counter.value))
        episode_length += 1
        # Sync with the shared model
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()   

        state, reward, done, _ = env.step(action[0, 0])
        # env.render()
        # done = done or episode_length >= args.max_episode_length
        # print(episode_length)
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])

        if done:
            n_epi +=1
            
            print('[Test] episode: %3d, episode_reward: %5f' % (n_epi, reward_sum))  
            wandb.log({
                "Episode Reward": reward_sum ,
                # "num steps": counter.value,
                "episode length": episode_length})
            all_rewards.append(reward_sum)
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            if n_epi > n_episodes:
                break
            
        state = np.array(state,copy=False)
        state = torch.from_numpy(state.transpose(2, 0, 1)/255).float()
        
    print("avg reward: %5f" % (np.mean(all_rewards)))
    env.reset()
        
def save(model, save_dir, name):
        modelpath =os.path.join(save_dir, name)
        torch.save(model.state_dict(), modelpath)
        print("Saved to model to {}".format(modelpath))

def load(model, save_dir, name):
    modelpath =os.path.join(save_dir, name)
    state_dict = torch.load(modelpath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    #     state = np.zeros(self.state_size)
    #     obs = self.env.reset()
    #     state = torch.tensor(np.vstack([state[1:], (obs.transpose(2, 0, 1))/255])).to('cpu')#, dtype = torch.float)

    #     for t in count():
    #         action = self.get_action(0, state)
    #         next_obs, reward, done, _ = self.env.step(action)
    #         self.env.render()
    #         next_state = torch.vstack([state[1:], (torch.from_numpy(next_obs.transpose(2, 0, 1)))/255]).to('cpu')

    #         epi_reward += reward

    #         if done:
    #             all_rewards.append(epi_reward)
    #             print('[Test] episode: %3d, episode_reward: %5f' % (i_ep, epi_reward))
    #             epi_reward = 0
    #             break
    #         else:
    #             state = next_state  
        

    # print("avg reward: %5f" % (np.mean(all_rewards)))
    # self.env.reset()
    
    
   
