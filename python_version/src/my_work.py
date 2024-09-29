
import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":      
    # policy 1       
    env = GridWorld(env_size = (4,4),start_state = (0,0),target_state = (2,2),
                    forbidden_states = [(2,1),(1,2),(2,3)],reward_target=1,
                    reward_forbidden=-1,reward_step=0,)
    policy = [1,1,1,0,1,2,4,0,2,4,4,3,2,3,4,2]
    P = np.zeros((13,13))
    P[0][1] = P[1][2] = P[2][3] = P[3][6] = P[4][5] = P[5][1] = P[6][9] = P[7][4] = P[8][8] = P[9][8] =1
    P[10][7] =P[11][10] = P[12][9] =1
    
    discount=0.9
    state = env.reset()
    next_state = state[0]      
    print("Start policy 1")       
    for t in range(50):
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
    ######closed form
    env = GridWorld(env_size = (4,4),start_state = (0,0),target_state = (2,2),
                    forbidden_states = [(2,1),(1,2),(2,3)],reward_target=1,
                    reward_forbidden=-1,reward_step=0,)
    state = env.reset()
    next_state = state[0] 
    for t in range(50):
        env.render()
        action = env.action_space[policy[next_state[1]*4+next_state[0]]]
        next_state, reward, done, info = env.step(action)
        reward = reward*(discount**t)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
             break
    rpi = np.zeros((13,1))
    rpi[8] = 1
    Penv=np.linalg.inv(np.identity(13, dtype="int")-0.9*P)@rpi
    Penv = np.insert(Penv,6,0)
    Penv = np.insert(Penv,9,0)
    Penv = np.insert(Penv,14,0)
    env.add_state_values(Penv)
    print(Penv)
    env.save_fig("policy1_closed")
    #iterative solution
    env = GridWorld(env_size = (4,4),start_state = (0,0),target_state = (2,2),
                    forbidden_states = [(2,1),(1,2),(2,3)],reward_target=1,
                    reward_forbidden=-1,reward_step=0,)
    state = env.reset()
    next_state = state[0] 
    for t in range(50):
        env.render()
        action = env.action_space[policy[next_state[1]*4+next_state[0]]]
        next_state, reward, done, info = env.step(action)
        reward = reward*(discount**t)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
             break
    v = np.ones((13,1))
    for i in range(100):
        v = rpi + 0.9*P@v
    env.add_state_values(Penv)
    print(Penv)
    env.save_fig("policy1_interative")

    # policy 2
    env = GridWorld(env_size = (4,4),start_state = (0,0),target_state = (2,2),
                    forbidden_states = [(2,1),(1,2),(2,3)],reward_target=1,
                    reward_forbidden=-1,reward_step=0,)
    policy_matrix=np.ones((env.num_states,len(env.action_space)))        
    policy_matrix[:,4] = 0 #zeors stay 
    #zeors boundary    
    for i in range(4):
        policy_matrix[i,2] = 0     
    for i in range(4):
        policy_matrix[i*4+3,1] = 0    
    for i in range(4):
        policy_matrix[i*4,3] = 0  
    for i in range(12,16):
        policy_matrix[i,0] = 0                    
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
    print("Start policy 2") 
    state = env.reset()
    next_state = state[0]      
    #print(np.array(policy_matrix[next_state[1]*4+next_state[0],:]))
    for t in range(50):
        env.render()
        action = env.action_space[int(np.random.choice(a=np.array([0,1,2,3,4]), 
                                size=1, replace=True, 
                                p=np.array(policy_matrix[next_state[1]*4+next_state[0],:])))]
        next_state, reward, done, info = env.step(action)
        reward = reward*(discount**t)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
             break
    
    env.add_policy(policy_matrix)
    env.save_fig("policy2")
    P = np.zeros((13,13))
    P[0][1] = P[0][4] = 0.5
    P[1][0] = P[1][2] = P[0][5] = 0.33
    P[2][1] = P[2][2] = P[2][3] = 0.33
    P[3][2] = P[3][6] = 0.5
    P[4][0] = P[4][5] = P[4][7] = 0.33
    P[5][1] = P[5][4] = 0.25
    P[5][5] = 0.5
    P[6][3] = P[6][6] = P[6][9] = 0.33
    P[7][4] = P[7][7] = P[7][10] = 0.33
    P[8][8] = 0.75
    P[8][9] = 0.25
    P[9][7] = P[9][9] = P[9][12] = 0.33
    P[10][8] = P[10][10] = 0.5
    P[11][10] = 0.33
    P[11][12] = 0.67
    P[12][9] = P[10][12] = 0.5
    ### Plot state
    #closed
    rpi = np.array([0,0,-0.33,0,0,-0.5,-0.33,-0.33,-0.75,0.33,0,-0.67,-0.5])
    Penv=np.linalg.inv(np.identity(13, dtype="int")-0.9*P)@rpi
    Penv = np.insert(Penv,6,0)
    Penv = np.insert(Penv,9,0)
    Penv = np.insert(Penv,14,0)
    state = env.reset()
    next_state = state[0]      
    #print(np.array(policy_matrix[next_state[1]*4+next_state[0],:]))
    for t in range(50):
        env.render()
        action = env.action_space[int(np.random.choice(a=np.array([0,1,2,3,4]), 
                                size=1, replace=True, 
                                p=np.array(policy_matrix[next_state[1]*4+next_state[0],:])))]
        next_state, reward, done, info = env.step(action)
        reward = reward*(discount**t)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
             break
    env.add_state_values(Penv)
    print(Penv)
    env.save_fig("policy2_closed")
    #iterative
    state = env.reset()
    next_state = state[0]      
    #print(np.array(policy_matrix[next_state[1]*4+next_state[0],:]))
    for t in range(50):
        env.render()
        action = env.action_space[int(np.random.choice(a=np.array([0,1,2,3,4]), 
                                size=1, replace=True, 
                                p=np.array(policy_matrix[next_state[1]*4+next_state[0],:])))]
        next_state, reward, done, info = env.step(action)
        reward = reward*(discount**t)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
             break
    v = np.ones((13,1))
    for i in range(100):
        v = rpi + 0.9*P@v
    env.add_state_values(Penv)
    print(Penv)
    env.save_fig("policy2_interative")
