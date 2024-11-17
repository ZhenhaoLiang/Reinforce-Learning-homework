import matplotlib.pyplot as plt
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
err = []
alpha  =0.001
discount=0.9

def qtabel():
    policy = np.ones(16,dtype=int)
    P = update_P(policy)
  
    state_value = np.zeros((16,1)) #initial values
    for i in range(201): #iteration
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
        new_state_value = 0.9*P@state_value + rpi
        now_error = (state_value - new_state_value)
        state_value = new_state_value
        state_value = np.insert(state_value,6,0)
        state_value = np.insert(state_value,9,0)
        state_value = np.insert(state_value,14,0)
        if np.linalg.norm(now_error) < 0.001:
            break
    #plot policy and sate
    policy_matrix=np.zeros((env.num_states,len(env.action_space)))                                            
    for k in range(len(policy)):
        policy_matrix[k,policy[k]] = 1
    env.render()
    env.add_policy(policy_matrix)
    env.add_state_values(state_value)
    env.save_fig(f"qtableoptimal")
    return state_value

def behavior_policy(env):
    policy_matrix=np.ones((env.num_states,len(env.action_space)))                          
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
    print("Start plot") 
    state = env.reset()
    next_state = state[0]      
    env.render()
    env.add_policy(policy_matrix)
    env.save_fig("pi_b_expericenve_policy")
    #print(np.array(policy_matrix[next_state[1]*4+next_state[0],:]))
    #To know the development of pi_b policy
    for t in range(10000):
        env.render()
        action = env.action_space[int(np.random.choice(a=np.array([0,1,2,3,4]), 
                                size=1, replace=True, 
                                p=np.array(policy_matrix[next_state[1]*4+next_state[0],:])))]
        next_state, reward, done, info = env.step(action)
        reward = reward*(discount**t)
    env.save_fig("pi_b_expericenve")
    return policy_matrix

if __name__ == "__main__":      
    env = GridWorld(env_size = (4,4),start_state = (0,0),target_state = (2,2),
                    forbidden_states = [(2,1),(1,2),(2,3)],reward_target=1,
                    reward_forbidden=-1,reward_step=0,)
    state_value1 = qtabel() #optimal state values from last assignment 
    policy_matrix = behavior_policy(env) #pi_b policy matrix
    env.render()
    print("Start iteration") 
    policy = np.ones(16,dtype=int)
    action_dic = {(0,1):0,(1,0):1,(0,-1):2,(-1,0):3,(0,0):4}
    qtabel2 = np.zeros((16,5))
    piT = np.zeros((16,5))
    state = env.reset()
    next_state = state[0]

    #update q-table_t by pi_b
    for t in range(5000):
        env.render()
        action = env.action_space[int(np.random.choice(a=np.array([0,1,2,3,4]), 
                                size=1, replace=True, 
                                p=np.array(policy_matrix[next_state[1]*4+next_state[0],:])))]
        #get a_t,s_t,s_t+1
        at = action_dic[action]
        st = next_state[1]*4+next_state[0]
        next_state, reward, done, info = env.step(action)
        st1 = next_state[1]*4+next_state[0]
        # update q-table_t+1
        qtabel2[st,at] = qtabel2[st,at] - alpha*(qtabel2[st,at] - (reward+discount*max(qtabel2[st1,:])))
        
        #calculate error from current PiT
        if t%10 == 0:
            for s in range(len(piT)):
                for index in range(5):
                    piT[s,index] = 0
                piT[s,np.argmax(qtabel2[s,:])] = 1
            temp_policy = np.zeros(16,dtype=int)
            for i in range(len(piT)):
                for j in range(5):
                    #print(piT[i,j])
                    if piT[i,j] == 1:
                        temp_policy[i] =j
                    final_P = update_P(temp_policy)
            rpi = np.zeros((13,1))
            rpi[8] = 1
            Penv=np.linalg.inv(np.identity(13, dtype="int")-0.9*final_P)@rpi
            Penv = np.insert(Penv,6,0)
            Penv = np.insert(Penv,9,0)
            Penv = np.insert(Penv,14,0)
            err.append(np.linalg.norm(Penv - state_value1))
    #Get final pi_T
    for s in range(len(piT)):
        for index in range(5):
            piT[s,index] = 0
        piT[s,np.argmax(qtabel2[s,:])] = 1

    
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
    env.save_fig("Qlearning")

    #plot error with evolvement process
    plt.figure()
    plt.plot([i*10 for i in range(len(err))],err)
    plt.xlabel('t')
    plt.ylabel('State value error')
    plt.show()
    plt.savefig('../plots/err.png')

    