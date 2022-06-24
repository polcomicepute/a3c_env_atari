import time
from collections import deque
import os
import torch
import torch.nn.functional as F

# from envs import create_atari_env
from envs_hm import make_atari,wrap_deepmind

from model import ActorCritic
import wandb
wandb.login()
os.environ["WANDB_API_KEY"] = "41d70fa07c07644beca7655a81d7af49afccf2dd"

def test(rank, args, shared_model, counter):
    wandb.init(project='Pong_NoGAE_4frame+LSTM', entity = "polcom",name='test1_Pong_NoGAE_4frame+LSTM', config=None, sync_tensorboard=True, settings=wandb.Settings(start_method='thread', console="off"))
    wandb.config.update(args)
    
    torch.manual_seed(args.seed + rank)

    env = make_atari(args.env_name, render_mode='human')
    env = wrap_deepmind(env)
    env.seed(args.seed + rank)
    
    img_h, img_w, img_c = env.observation_space.shape
    state_size = [1*img_c, img_h, img_w]
    print('train', state_size)

    model = ActorCritic(state_size[0], env.action_space)
    wandb.watch(model)
    

    # model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state.transpose(2, 0, 1)/255).float()

    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    
    while True:
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
        
        
        # action = prob.multinomial(num_samples=1).detach() #!왜 랜덤하게 샘플링하지? max값이 아니고
        #     #! num sample개의 인덱스 나옴 
        # log_prob = log_prob.gather(1, action) #! 해당 action의 인덱스를 logprobd에서 가져와 

        #     #! perform action, get next state and reward
        #     # print('action',action[0].numpy())
            
        # state, reward, done, _ = env.step(int(action[0]))
        

        state, reward, done, _ = env.step(action[0, 0])
        # env.render()
        done = done or episode_length >= args.max_episode_length
        # print(episode_length)
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        # if actions.count(actions[0]) == actions.maxlen:
        #     done = True

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
            # data = [[x, y] for (x, y) in zip(recall_micro, precision_micro)]
            # table = wandb.Table(data=data, columns = ["recall_micro", "precision_micro"])
            # wandb.log({"my_lineplot_id" : wandb.plot.line(table, "recall_micro", "precision_micro", stroke=None, title="Episode Reward")})
            
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            # time.sleep(60)

        state = torch.from_numpy(state.transpose(2, 0, 1)/255).float()
        
