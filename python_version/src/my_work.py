
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