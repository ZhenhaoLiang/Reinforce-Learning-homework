
import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np
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

if __name__ == "__main__":      
    discount=0.9

    policy = np.ones(16,dtype=int)
    P = update_P(policy)
  
    state_value = np.zeros((16,1)) #initial values
    for i in range(61): #iteration
        env = GridWorld(env_size = (4,4),start_state = (0,0),target_state = (2,2),
                    forbidden_states = [(2,1),(1,2),(2,3)],reward_target=1,
                    reward_forbidden=-1,reward_step=0)
        state = env.reset()
        rpi = []
        for s in range(16):  
            if s in [6,9,14]:
                continue
            q_table = []
            for k in range(5): #k is action
                (x, y), reward = env._get_next_state_and_reward((s%4,s//4),env.action_space[k])
                q_table.append(reward+discount*(state_value[y*4+x]))  
            max_value = max(q_table) #get q_table and choose firdt max valueas update action
            
            policy[s] = q_table.index(max_value)
            (x, y), reward = env._get_next_state_and_reward((s%4,s//4),env.action_space[policy[s]])

            rpi.append(reward) #update return
        #state_value update
        P = update_P(policy)
        state_value = np.delete(state_value, [6,9,14])
        state_value = 0.9*P@state_value + rpi
        state_value = np.insert(state_value,6,0)
        state_value = np.insert(state_value,9,0)
        state_value = np.insert(state_value,14,0)
        #plot policy and sate
        policy_matrix=np.zeros((env.num_states,len(env.action_space)))                                            
        for k in range(len(policy)):
            policy_matrix[k,policy[k]] = 1
        env.render()
        env.add_policy(policy_matrix)
        env.add_state_values(state_value)
        env.save_fig(f"PolicyAndState_{i}")

    next_state = state[0]      
    print("Start optimal policy.")       
    for t in range(10):
        env.render()
        action = env.action_space[policy[next_state[1]*4+next_state[0]]]
        next_state, reward, done, info = env.step(action)
        reward = reward*(discount**t)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
             break
    # Add policy
    policy_matrix=np.zeros((env.num_states,len(env.action_space)))                                            
    for i in range(len(policy)):
        policy_matrix[i,policy[i]] = 1
    env.add_policy(policy_matrix)
    env.save_fig("policy1")