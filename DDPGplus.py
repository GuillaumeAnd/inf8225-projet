# Import
!apt-get install x11-utils > /dev/null 2>&1 
!pip install pyglet > /dev/null 2>&1 
!apt-get install -y xvfb python-opengl > /dev/null 2>&1

!pip install gym pyvirtualdisplay > /dev/null 2>&1

!pip install gym[box2d]
!pip install pyvirtualdisplay
!pip install PyOpenGL
!pip install PyOpenGL-accelerate

import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay

from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()

# Import
import torch
import numpy as np
import random, math

import gym
from gym.wrappers import Monitor

import base64
from pathlib import Path
from gym.wrappers import Monitor
def show_video(directory):
    html = []
    for mp4 in Path(directory).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
    
### DDPG ameliore : ###

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_maxi):
        super(Actor, self).__init__()

        self.lin1 = nn.Linear(state_dim, 400)
        self.lin2 = nn.Linear(400, 300)
        self.lin3 = nn.Linear(300, action_dim)

        self.action_maxi = action_maxi

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        out = torch.tanh(self.lin3(x))
        out = self.action_maxi * out
        return out
    

class Critic(nn.Module):  #pour les 2 critics
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.lin1_1 = nn.Linear(state_dim + action_dim, 400)  #Critic1
        self.lin1_2 = nn.Linear(400, 300)   #400 & 300 provient d'un papier de recherche
        self.lin1_3 = nn.Linear(300, 1)

        self.lin2_1 = nn.Linear(state_dim + action_dim, 400) #Critic2
        self.lin2_2 = nn.Linear(400, 300)
        self.lin2_3 = nn.Linear(300, 1)


    def forward(self, x, u):
        x_cat = torch.cat([x, u], 1)

        x1 = F.relu(self.lin1_1(x_cat))
        x1 = F.relu(self.lin1_2(x1))
        x1 = self.lin1_3(x1)

        x2 = F.relu(self.lin2_1(x_cat))
        x2 = F.relu(self.lin2_2(x2))
        x2 = self.lin2_3(x2)

        return x1, x2  #Renvoie 2 estimations de Q (pour ensuite prendre la min)



    def Q1(self, x, u):
        x_cat = torch.cat([x, u], 1)

        out = F.relu(self.lin1_1(x_cat))
        out = F.relu(self.lin1_2(out))
        out = self.lin1_3(out)
        return out 
    

class ReplayBuffer(object):
    def __init__(self):
        self.storage = []
        self.size = 1e6
        self.pos = 0

    def add(self, data):
        if len(self.storage) == self.size:
            self.storage[int(self.pos)] = data
            self.pos = (self.pos + 1) % self.size  #on ajoute à la suite modulo max size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))  #autrement ça bug
            y.append(np.array(Y, copy=False))  # idée reprise d'un github externe
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)   
    
class DDPGplus(object): 

    def __init__(self, state_dim, action_dim, action_maxi):

        self.actor = Actor(state_dim, action_dim, action_maxi)
        self.actor_target = Actor(state_dim, action_dim, action_maxi)
        self.actor_target.load_state_dict(self.actor.state_dict())   #Initialisation du dico des params
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())  #on utilise adam

        self.critic = Critic(state_dim, action_dim)   #idem
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.action_maxi = action_maxi

    def choose_action(self, state):   #Pi(S)
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()  # .data pour save 

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, bruit_policy=0.2, bruit=0.5, policy_freq=2): 
      # hyperparamètres à modifier pour toruver l'optimale
        
        for i in range(iterations):

            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x) 
            action = torch.FloatTensor(u) 
            next_state = torch.FloatTensor(y) 
            done = torch.FloatTensor(1 - d) 
            reward = torch.FloatTensor(r) 

          
            bruit = torch.FloatTensor(u).data.normal_(0, bruit_policy) # .data essentiel
            bruit = bruit.clamp(-bruit, bruit)   #restreint le bruit
            
    # Amélioration 3 :on rajoute le bruit à l'action cible

            next_action = (self.actor_target(next_state) + bruit).clamp(-self.action_maxi, self.action_maxi)


            #Calcul des estimations de Q
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            current_Q1, current_Q2 = self.critic(state, action)

    # Amélioration 1 : prendre le plus petit target_Q

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Fonction de perte critic
            Q_loss = 0.5*(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

            # backward de critic
            self.critic_optimizer.zero_grad()
            Q_loss.backward()
            self.critic_optimizer.step()

    # Amélioration 2 : actualisation de actor 1 fois sur policy freq
            if i % policy_freq == 0:

                # actualisation de actor avec la perte de critic
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Actualisation des modèles à la fin pour la stabilité

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    # utilsation de data

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    
def save(agent, name, dir):  #on sauvegarde le modèle pour éviter de tj le refaire
  torch.save(agent.actor.state_dict(), '%s/%s_actor.pth' % (dir, name))
  torch.save(agent.critic.state_dict(), '%s/%s_critic.pth' % (dir, name))
  torch.save(agent.actor_target.state_dict(), '%s/%s_actor_tgt.pth' % (dir, name))
  torch.save(agent.critic_target.state_dict(), '%s/%s_critic_tgt.pth' % (dir, name))  
 
 # Paramètres
start_timestep=1e4

std_noise=0.1

env = gym.make('BipedalWalker-v3')

# Set seeds
seed = 2164904
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state = env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
action_maxi = float(env.action_space.high[0])

agent = DDPGplus(state_dim, action_dim, action_maxi)

### DDPG train + video
def DDPGplus_train(n_episodes=5):

    scores_deque = deque(maxlen=100)
    scores_list = []
    avg_scores_list = []    
    time_start = time.time()           
    replay_buf = ReplayBuffer()   
    
    timestep_after_last_save = 0
    total_timesteps = 0
    low = env.action_space.low
    high = env.action_space.high
    
            
    for i_episode in range(1, n_episodes):
        
        timestep = 0
        total_reward = 0
        
        
        state = env.reset()
        done = False
        
        while not done:

            if total_timesteps < start_timestep:  #1e4
                action = env.action_space.sample()   #Actions initiales
            else:
                action = agent.choose_action(np.array(state))
                shift_action = np.random.normal(0, std_noise, size=action_dim)
                action = (action + shift_action).clip(low, high)
            
            new_state, reward, done, _ = env.step(action) 
            done_bool = 0 if timestep + 1 == env._max_episode_steps else float(done)
            total_reward += reward                       

            # Store every timestep in replay buffer
            replay_buf.add((state, new_state, action, reward, done_bool))
            state = new_state

            timestep += 1     
            total_timesteps += 1
            timestep_after_last_save += 1

        scores_deque.append(total_reward)
        scores_list.append(total_reward)

        avg_score = np.mean(scores_deque)
        avg_scores_list.append(avg_score)
        
        # train_by_episode(time_start, i_episode) 
        
        print('Ep. {}, Score: {:.2f}, Avg.Score: {:.2f}'.format(i_episode, total_reward, avg_score))     

        agent.train(replay_buf, timestep)

        # save tous les 10 ep
        if timestep_after_last_save >= 10:

            timestep_after_last_save %= 10            
            save(agent, 'test_1', 'DDPGplus')  
    
    ### Dernier épisode pour vidéo

    i_episode = n_episodes +1

    monitor_env = Monitor(env, "./gym-results", force=True, video_callable=lambda episode: True)
    timestep = 0
    total_reward = 0
        
    state = monitor_env.reset()
    done = False
    while not done:
      if total_timesteps < start_timestep:  #1e4
        action = monitor_env.action_space.sample()
      else:
        action = agent.choose_action(np.array(state))
        shift_action = np.random.normal(0, std_noise, size=action_dim)
        action = (action + shift_action).clip(low, high)
            
        new_state, reward, done, _ = monitor_env.step(action) 
        done_bool = 0 if timestep + 1 == monitor_env._max_episode_steps else float(done)
        total_reward += reward                       

        # Store every timestep in replay buffer
        replay_buf.add((state, new_state, action, reward, done_bool))
        state = new_state

        timestep += 1     
        total_timesteps += 1
        timestep_after_last_save += 1

    scores_deque.append(total_reward)
    scores_list.append(total_reward)

    avg_score = np.mean(scores_deque)
    avg_scores_list.append(avg_score)
        
        # train_by_episode(time_start, i_episode) 
        
    print('Ep. {}, Score: {:.2f}, Avg.Score: {:.2f}'.format(i_episode, total_reward, avg_score))     

    agent.train(replay_buf, timestep)

        # save tous les 10 ep
    if timestep_after_last_save >= 10:

        timestep_after_last_save %= 10            
        save(agent, 'test_1', 'DDPGplus')  
    monitor_env.close()
    show_video("./gym-results")

    return scores_list, avg_scores_list

scores, avg_scores = DDPGplus_train()

import pandas as pd
import matplotlib.pyplot as plt


res = pd.read_csv('DDPGplus_res.txt', delimiter = ",", header=None)
res.head()
score = res[3].apply(lambda x : float(x.split(":")[1]))
avg_score = res[4].apply(lambda x : float(x.split(":")[1]))
Ep = res[0].apply(lambda x : x.split(".")[1])

plt.plot(range(len(score)),score,range(len(score)),avg_score)
