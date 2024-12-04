import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches      
from copy import deepcopy

class GridWord():
    def __init__(self, env_size: tuple = (4, 4),
                 start_state: tuple = (0, 0),
                 target_state: tuple = (2, 2),
                 forbidden_states: list = [(2, 1), (1, 2), (2, 3)],
                 reward_target: float = 1,
                 reward_forbidden: float = -1,
                 reward_step: float = 0,
                 action_space: list = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)], # right, down, left, up, stay
                 debug: bool = False,
                 animation_interval: float = 0.2,
                 state_name = None,
                 ax = None):
        '''
        Initialize the environment

        Parameters:
            env_size: tuple, the size of the environment
            start_state: tuple, the start state of the agent
            target_state: tuple, the target state of the agent
            forbidden_states: list, the forbidden states cannot be visited by the agent
            reward_target: float, the reward for reaching the target state
            reward_forbidden: float, the reward for visiting the forbidden states or hitting the wall
            reward_step: float, the reward for each step
            action_space: list, the action space of the agent
            debug: bool, whether to debug, if True, the agent will wait for the user to press Enter to continue
            animation_interval: float, the interval for rendering the environment, unit is second
        '''
        
        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states
        self.debug = debug

        self.agent_state = start_state
        self.action_space = action_space
        self.reward_target = reward_target
        self.reward_forbidden = reward_forbidden
        self.reward_step = reward_step
        self.state_name = state_name

        self.ax = ax
        self.animation_interval = animation_interval

        self.color_forbid = (0.9290,0.6940,0.125)
        self.color_target = (0.3010,0.7450,0.9330)
        self.color_policy = (0.4660,0.6740,0.1880)
        self.color_trajectory = (0, 1, 0)
        self.color_agent = (0,0,1)

    def reset(self):
        '''
        Reset the environment to the start state
        '''
        self.agent_state = self.start_state # reset the agent
        self.traj = [self.agent_state] # record the trajectory of the agent
        return self.agent_state, {}

    def step(self, action: tuple[int]) -> tuple[tuple[int], float, bool, dict]:
        '''
        Do the action and return the next state, reward, done and info

        Parameters:
            action: tuple, the action to take

        Returns:
            next_state: tuple, the next state
            reward: float, the reward for the action
            done: bool, whether the episode is done
            empty_dict: dict, an empty dictionary
        '''

        assert action in self.action_space, "Invalid action"

        next_state, reward  = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store,  y_store)) + 0.2 * np.array(action)) # For rendering
        state_store_2 = (next_state[0], next_state[1]) # the actual next state

        self.agent_state = next_state

        self.traj.append(state_store)   
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}  

    def _get_next_state_and_reward(self, state: tuple[int], action: tuple[int]) -> tuple[tuple, float]:
        '''
        Get the next state and reward for the action

        Parameters:
            state: tuple, the current state
            action: tuple, the action to take
        
        Returns:
            next_state: tuple, the next state
            reward: float, the reward for the action
        '''
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))

        # check if the agent hits the wall
        if y + 1 > self.env_size[1] - 1 and action == (0,1):    # down
            y = self.env_size[1] - 1
            reward = self.reward_forbidden  
        elif x + 1 > self.env_size[0] - 1 and action == (1,0):  # right
            x = self.env_size[0] - 1
            reward = self.reward_forbidden  
        elif y - 1 < 0 and action == (0,-1):   # up
            y = 0
            reward = self.reward_forbidden  
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden 
        elif new_state == self.target_state:  # stay
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # stay
            x, y = state
            reward = self.reward_forbidden        
        else:
            x, y = new_state
            reward = self.reward_step
            
        return (x, y), reward
    
    def _is_done(self, state: tuple[int]) -> bool:
        '''
        Whether the episode is done

        Parameters:
            state: tuple, the current state
        
        Returns:
            bool, whether the episode is done
        '''

        return state == self.target_state

    def render(self):
        # initialize the canvas if it is None
        if self.ax is None:
            plt.ion()                        # turn on the interactive mode     
            self.canvas, self.ax = plt.subplots() 
        
        self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
        self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
        self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))     
        self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))     
        self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')          
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()                           
        self.ax.xaxis.set_ticks_position('top')           
            
        idx_labels_x = [i for i in range(self.env_size[0])]
        idx_labels_y = [i for i in range(self.env_size[1])]
        for lb in idx_labels_x:
            self.ax.text(lb, -0.75, str(lb+1), size=10, ha='center', va='center', color='black')           
        for lb in idx_labels_y:
            self.ax.text(-0.75, lb, str(lb+1), size=10, ha='center', va='center', color='black')
        self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,labeltop=False)   

        self.target_rect = patches.Rectangle( (self.target_state[0]-0.5, self.target_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_target, facecolor=self.color_target)
        self.ax.add_patch(self.target_rect)     

        for forbidden_state in self.forbidden_states:
            rect = patches.Rectangle((forbidden_state[0]-0.5, forbidden_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_forbid, facecolor=self.color_forbid)
            self.ax.add_patch(rect)

        self.agent_star, = self.ax.plot([], [], marker = '*', color=self.color_agent, markersize=20, linewidth=0.5) 
        self.traj_obj, = self.ax.plot([], [], color=self.color_trajectory, linewidth=0.5)

        # state_name
        if self.state_name is not None:
            for i, state in enumerate(self.state_name):
                if state is not None:
                    x = i % self.env_size[0]
                    y = i // self.env_size[0]
                    self.ax.text(x, y, state, ha='center', va='center', fontsize=10, color='black')

        # self.agent_circle.center = (self.agent_state[0], self.agent_state[1])
        self.agent_star.set_data([self.agent_state[0]],[self.agent_state[1]])       
        traj_x, traj_y = zip(*self.traj)         
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(self.animation_interval)
        if self.debug:
            input('press Enter to continue...')    
    
    def add_policy(self, policy_matrix: np.ndarray):
        '''
        Add the policy to the environment

        Parameters:
            policy_matrix: np.ndarray, the policy matrix. The shape of the matrix should be (num_states, num_actions)
        '''
        for state, state_action_group in enumerate(policy_matrix):
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability !=0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0,0):
                        self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability/2)*dx, dy=(0.1+action_probability/2)*dy, color=self.color_policy, width=0.001, head_width=0.05))
                        if action_probability != 1:
                            self.ax.text(x + (0.1 + action_probability / 2) * dx + 0.1*dx, y + (0.1 + action_probability / 2) * dy - 0.1*dy, f'P={action_probability:.1f}', color='black', fontsize=8, ha='center', va='center')
                    else:
                        self.ax.add_patch(patches.Circle((x, y), radius=0.07, facecolor=self.color_policy, edgecolor=self.color_policy, linewidth=1, fill=False))

    def add_state_values(self, values, precision=1):
        '''
        add the state values to the environment

        Parameters:
            values: iterable, the state values
            precision: int, the precision of the values
        '''
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            if value is not None:
                self.ax.text(x, y+0.15, str(np.round(value, precision)), ha='center', va='center', fontsize=10, color='black')