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

def test(rank, args, shared_model, counter):
    wandb.init(project=str(args.env_name) + '_NoGAE_1frame+LSTM', entity = "polcom",name='test2_'+ str(args.env_name) +'_NoGAE_4frame+LSTM', config=None, sync_tensorboard=True, settings=wandb.Settings(start_method='thread', console="off"))
    wandb.config.update(args)
    
    torch.manual_seed(args.seed + rank)

    env = make_atari(args.env_name, render_mode='human')
    env = wrap_deepmind(env)
    env.seed(args.seed + rank)
    
    img_h, img_w, img_c = env.observation_space.shape
    state_size = [1*img_c, img_h, img_w]
    print('test', state_size)

    model = ActorCritic(state_size[0], env.action_space)
    wandb.watch(shared_model)
    
    
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
    
    best_reward = -10000
    while True:
        # print('test' + str(counter.value))
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
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
        done = done or episode_length >= args.max_episode_length
        # print(episode_length)
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            wandb.log({
                "Episode Reward": reward_sum ,
                "num steps": counter.value,
                "episode length": episode_length})
             
            if best_reward < reward_sum:
                best_reward = reward_sum
                save(shared_model,'gym-results-hm/','best_a3c_'+str(args.env_name)+'.pth.tar')
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            if counter.value > 5000000: 
                break
        if counter.value % args.save_freq == 0:
                save(shared_model,'gym-results-hm/','a3c_'+str(args.env_name)+'.pth.tar')
                
        state = np.array(state,copy=False)
        state = torch.from_numpy(state.transpose(2, 0, 1)/255).float()
        
        
        
def save(model, save_dir, name):
        modelpath =os.path.join(save_dir, name)
        torch.save(model.state_dict(), modelpath)
        print("Saved to model to {}".format(modelpath))

def load(model, save_dir, name):
    modelpath =os.path.join(save_dir, name)
    state_dict = torch.load(modelpath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

