import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from grid_world import GridWorld
import random
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.utils.data as dataf
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision 
from copy import deepcopy
from gw import GridWord

def update_P(policy): #update martrix by policy
    P = np.zeros((16,16))
    for i in range(16):
        if i in [6,9,14]: #pass forbidden
            continue
        if policy[i] == 1:
            if ((i//4)==3)|((i+1) in [6,9,14]):
                P[i][i] = 1
            else:
                P[i][i+1] = 1
        elif policy[i] == 0:
            if (i>11)|((i+4) in [6,9,14]):
                P[i][i] = 1
            else:
                P[i][i+4] = 1
        elif policy[i] == 3:
            if ((i//4)==0)|((i-1) in [6,9,14]):
                P[i][i] = 1
            else:
                P[i][i-1] = 1
        elif policy[i] == 2:
            if (i<4)|((i-4) in [6,9,14]):
                P[i][i] = 1
            else:
                P[i][i-4] = 1
        elif policy[i] == 4:
            P[i][i] = 1

    P = np.delete(P, [6,9,14], axis=0)  
    P = np.delete(P, [6,9,14], axis=1)

    return P

alpha  =0.001
discount=0.9

class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.fc1 = nn.Linear(3, 160)
        self.fc2 = nn.Linear(160, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":   
    nsteps=10000 # number of steps
    state_space = [(j,i) for i in range(4) for j in range(4)]
    action_space = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
    state_name = ["s1", "s2", "s3", "s4", "s5", "s6", None,"s7", "s8", None,"s9", "s10", "s11", "s12", None,"s13"]
    policy = np.ones((len(state_space), len(action_space))) / 5
    policy[[6, 9, 14],:] = 0

    samples = []
    env = GridWord(action_space=action_space, state_name=state_name, start_state=(0,0)) # start from s1
    state, _ = env.reset()
    for _ in trange(nsteps):
        sample = [state[0], state[1], None, None, None, None]
        action = action_space[np.random.choice(np.arange(len(action_space)), p=policy[state[0] + state[1] * 4])] 
        sample[2] = action_space.index(action) # map the action to index
        state, reward, _, _ = env.step(action)
        sample[3] = reward
        sample[4] = state[0]
        sample[5] = state[1]
        samples.append(sample)
    samples = np.array(samples)
    env.render()
    plt.savefig("../plots/behavior_policy.png", dpi=600) # plot the trajectory
    plt.close()

    target_nn = MyNN()
    LR = 0.005
    main_nn = deepcopy(target_nn)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(main_nn.parameters(), lr=LR)
    state_space = [(j,i) for i in range(4) for j in range(4)]
    loss_history = []
    BATCH_SIZE = 200
    EPOCH = 5000
    value_store = []
    err = []

    #model train
    for epoch in range(EPOCH):
        # Calculate the q values for all states and actions
        qvalues = np.zeros([16,5])
        for sindex in range(16):
            for aindex in range(5):
                qvalues[sindex, aindex] = main_nn.forward(
                torch.tensor([state_space[sindex][0], state_space[sindex][1], aindex], dtype=torch.float32)).item()
        batch = samples[np.random.choice(a=np.arange(len(samples)), size=BATCH_SIZE)]
        value_store.append(np.max(qvalues, axis=1))
        # Calculate the target values
        x = torch.tensor(batch[:, 0:3], dtype=torch.float32)
        next_state = batch[:, 4:6]
        q_next = []
        for n in range(BATCH_SIZE):
            q_next.append(max([target_nn.forward(torch.tensor([next_state[n, 0], next_state[n, 1], a],
                                                            dtype=torch.float32)) for a in range(5)]))
        q_values_next = torch.tensor(q_next, dtype=torch.float32)
        rewards = torch.tensor(batch[:, 3], dtype=torch.float32)
        y_t = rewards + discount * q_values_next

        # Update the main network
        y_t = y_t.reshape(BATCH_SIZE, 1)
        y = main_nn.forward(x)
        loss = loss_fn(y, y_t)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
        # Print loss 
        if epoch % 10 == 0:
            print(f"Iteration: {epoch}, Loss: {loss.item()}")
        loss_history.append(loss.item())

        # Update W_T
        if epoch % 25 == 0:
            target_nn = deepcopy(main_nn)

    plt.plot([i*10 for i in range(len(loss_history))], loss_history, 'b', linewidth=1, label='train data')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("../plots/modelerr.png")
    plt.show()

    err = []
    for i in range(len(value_store)-1):
        err.append(np.linalg.norm(value_store[i] - value_store[-1]))
    plt.figure()
    plt.plot([i*10 for i in range(len(err))],err)
    plt.xlabel('epoch')
    plt.ylabel('State value error')
    plt.show()
    plt.savefig('../plots/err.png')

    #get fianl policy
    qtabel2 = np.zeros((16,5))
    piT = np.zeros((16,5))
    for a in range(5):
        for s in range(16):
            tmepinput = torch.tensor([float(s%4),float(s//4),float(a)])
            qtabel2[s,a] = target_nn(tmepinput)

    for s in range(len(piT)):
        for index in range(5):
            piT[s,index] = 0
        piT[s,np.argmax(qtabel2[s,:])] = 1

    #plot final policy
    env = GridWorld(env_size = (4,4),start_state = (0,0),target_state = (2,2),
                    forbidden_states = [(2,1),(1,2),(2,3)],reward_target=1,
                    reward_forbidden=-1,reward_step=0,)
    state = env.reset()
    next_state = state[0] 
    final_policy = np.zeros(16,dtype=int)
    for i in range(len(piT)):
        for j in range(5):
            if piT[i,j] == 1:
                final_policy[i] =j
        
    #print(final_policy)
    #plot episode
    for t in range(10):
        env.render()
        action = env.action_space[final_policy[next_state[1]*4+next_state[0]]]
        next_state, reward, done, info = env.step(action)
        reward = reward*(discount**t)
        if done:
                break

    #closed form calculate state value for final policy
    rpi = np.zeros((13,1))
    rpi[8] = 1

    final_P = update_P(final_policy)
    Penv=np.linalg.inv(np.identity(13, dtype="int")-0.9*final_P)@rpi
    Penv = np.insert(Penv,6,0)
    Penv = np.insert(Penv,9,0)
    Penv = np.insert(Penv,14,0)
    env.add_state_values(Penv)
    env.add_policy(piT)
    env.save_fig("DQlearning")