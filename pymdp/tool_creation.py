#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:43:12 2023

Tool use with actinf

Based on tool3_5_extrapolate_cut (Paul's code)

Note: requires sparse_likelihoods_111 branch of pymdp                                  
                                
@author: poppy
"""

import pymdp
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs import Env

# paul's imports
#from pymdp import utils
#from pymdp.envs import Tool_create_single_cut as Tool_create

# my imports
import sys
sys.path.append('/home/poppy/Documents/Active Inference/pymdp projects/pauls_files')
from tool_create_single_cut import Tool_create_single_cut as Tool_create

import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as image
import seaborn as sns
from scipy.stats import rankdata

from timeit import default_timer as timer
import time


"""Define dimensionality of hidden states, control states and observation modalities"""

#env parameters
init_room=0
init_tool=0
prob_A= 1. # confidence in A matrix (so eg 90% confident that a state maps to an observation)
prob_B= 1. # need sum over s of p(o=O|s)=1 so columns need to add  up to 1 

#agent parameter
policy_len=5
policies=None # give all possible policies use Paul's code for it
#policies_restriction= "single_action"

# states
state_room = ["Left_room", "Right_room"]
state_tool = ["None", "V", "H", "HV", "VH"] # what is Dummy1? ["None", "V", "H", "HV", "Dummy1"]
num_states = [len(state_tool),len(state_room)] # hidden state factor dimensions

## observations
obs_room = ["Left_room", "Right_room"]
obs_tool = ["None", "V", "H", "HV", "VH"]
obs_reward = ["Punish", "Reward"]
num_obs = [len(obs_tool), len(obs_room),len(obs_reward)] # observation modality dimensions

# controls
room_action = ["Null", "Move", "Pick-up", "Drop"] # 2 dims must be the same for policy restriction TO DOOOO
tool_action = ["Null", "Move", "Pick-up", "Drop"]
combined_action_list=[room_action, tool_action]
num_controls = [len(tool_action),len(room_action)] # control state factor dimensions
nulls=[0,0] # indices of the null actions for each control dimension


"""
Build A Matrix:
    Four arrays, one for each observation dimension of size: 
        obs[i].size * state[0].size * state[1].size
        e.g. 5*5*2, 2*5*2,  2*5*2,  s[0]=tool. s[1]=room.
"""


def build_A_tool(A, prob=1.0):
    """
    Returns
    -------
    A : A[0] filled of A matrix

    """
    # A[0].shape = 2 * 5 * 2
    for i in range(A[0].shape[2]): # 2
        non_prob=(1-prob)/(A[0].shape[0]-1)
        A[0][:,:,i] = np.ones((A[0].shape[0], A[0].shape[1]))*non_prob
        np.fill_diagonal(A[0][:,:,i], prob)
        
    # added in so that any dummy states are mapped to the dummy observaiotn
    for i in range(A[0].shape[2]):
        for j in range(A[0].shape[1]):
            if np.sum(A[0][:,j:,i]) == 0:
                A[0][-1,j,i]=1.
                
    return A

def build_A_room(A, prob=1.):
    """
    Returns
    -------
    A : A[1] filled of A matrix

    """
    for i in range(A[1].shape[1]):
        non_prob=(1-prob)/(A[1].shape[0]-1)
        A[1][:,i,:] = np.ones((A[1].shape[0], A[1].shape[2]))*non_prob
        np.fill_diagonal(A[1][:,i,:], prob)
    return A

def build_A_reward(A, reward_location, prob=1.):
    """
    Returns
    -------
    A : A[2] filled of A matrix

    """
    # if we want VH and HV to differ then 5:[0,4], else: 5:[0,3]
    lookup = {2:[1,3], 3:[1,2], 4:[0,2], 5:[0,4], 6:[0,1], 7:[1,1]} # [room, tool]
    reward_room = lookup[reward_location][0]
    reward_tool = lookup[reward_location][1]
    # if reward location fixed, tell agent what room and tool states combinations will get reward
    non_prob=(1-prob)/(A[2].shape[0]-1)
    A[2][0,:,:] = prob; 
    A[2][1,:,:] = non_prob;
    A[2][0,reward_tool,reward_room] = non_prob; 
    A[2][1,reward_tool,reward_room] = prob_A
    
    return A
    
def fill_A(A, reward_location, prob=prob_A):
    build_A_room(A, prob=prob_A)
    build_A_tool(A, prob=prob_A)
    build_A_reward(A, reward_location)
    return A



"""
Build B Matrix:
    Dimensions are [s[t+1] s[t], action]
    There are 2 states. Size of matrices = 5*5*2*4 (the 2 is the room state influencing the next tool state) 
    and 2*2*2 (next room state is only affected by current room state - current tool state has no influence)
"""

def build_B_tool(B, fully_known, prob=1.):
    """
    Note
    -------
    Tool state changes from 2nd dmiension to first
    e.g. B[0][3,2,0,1] is: action 1 (pickup) in room 0 changes tool state 2 to tool state 3
    
    Returns
    -------
    B : fills in B[0]
    """
    if fully_known:
        # tool state null action:
        # in room 0
        non_prob = 1-prob
        B[0][:,:,0,0] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,0,0], prob)
        # in room 1
        B[0][:,:,1,0] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,1,0], prob)
    
        # tool state move action:
        # in room 0
        B[0][:,:,0,1] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,0,1], prob)
        # in room 1
        B[0][:,:,1,1] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        np.fill_diagonal(B[0][:,:,1,1], prob)    
            
        # tool state pick-up action:
        # in room 0
        B[0][:,:,0,2] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][1,0,0,2] = prob
        B[0][1,1,0,2] = prob
        B[0][3,2,0,2] = prob
        B[0][3,3,0,2] = prob
        B[0][4,4,0,2] = prob
        # in room 1
        B[0][:,:,1,2] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][2,0,1,2] = prob
        B[0][4,1,1,2] = prob
        B[0][2,2,1,2] = prob
        B[0][3,3,1,2] = prob
        B[0][4,4,1,2] = prob 
                
        # tool state drop action:
        # drops everything and they go back to original starting place
        # in room 0
        B[0][:,:,0,3] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][0,:,0,3] = prob
        # in room 1
        B[0][:,:,1,3] = np.ones((B[0].shape[0], B[0].shape[1]))*non_prob
        B[0][0,:,1,3] = prob
        
    else:
        # all actions are totally unknown 
        # i.e. B[:,:,: for B[0] - each factor list factor is the same
        for i in range(B[0].shape[2]):
            for j in range(B[0].shape[3]):
                B[0][:,:,i,j] = np.ones((B[0].shape[0],B[0].shape[1])) / B[0][:,:,i,j].shape[0]   
                
    return B

def build_B_room(B, fully_known,  prob=1.):
    """
    Returns
    -------
    B : fills in B[1]
    """
    if fully_known:
        non_prob = (1-prob)/(B[0].shape[0]-1)
        # room state, null action
        B[1][:,:,0] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,0], prob)
        # room state, move action
        B[1][:,:,1] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        B[1][0,1,1] = prob
        B[1][1,0,1] = prob
        # room state, pick-up action
        B[1][:,:,2] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,2], prob)
        # room state, drop-off action
        B[1][:,:,3] = np.ones((B[1].shape[0], B[1].shape[1]))*non_prob
        np.fill_diagonal(B[1][:,:,3], prob)
        
    else:
        # all actions are totally unknown 
        # i.e. each B[:,:,i] is the same everywhere adding up to 1.0 for B[1]
        for i in range(B[1].shape[2]):
            B[1][:,:,i] = np.ones((B[1].shape[0],B[1].shape[1])) / B[1][:,:,i].shape[0]   
        
        
    return B

        
def fill_B(B, fully_known=True, prob=1):
    B = build_B_tool(B, fully_known, prob=1.)
    B = build_B_room(B, fully_known, prob=1.)
    return B


# set reward location 
def set_reward_location():
    reward_location=input("what room is reward in (Original experiment=2):")
    if int(reward_location) not in {2,3,4,5,6,7}:
        raise TypeError('Not a valid reward room. Allowable rooms {2,3,4,5,6,7}')
    reward_location = int(reward_location)
    orig_reward_location = reward_location
    return reward_location, orig_reward_location

reward_location, orig_reward_location = set_reward_location()
#reward_location, orig_reward_location = 5,5 # manually set to skip inputting

# A MATRIX
A_empty = utils.initialize_empty_A(num_obs, num_states)
A = fill_A(A_empty, reward_location, prob=prob_A)
pA = utils.dirichlet_like(A,scale=1.)
A_init, pA_init = np.copy(A), np.copy(pA)

# B MATRIX
B_factor_list=[[0,1],[1]] # B_tool depends on tool and room, B_room only depends on room
B_empty = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list)
B = fill_B(B_empty, fully_known = True, prob=prob_B)
pB = utils.dirichlet_like(B,scale=1.)
B_init, pB_init = np.copy(B), np.copy(pB)


# C MATRIX
punishment_value, reward_value = 0, 20
C = utils.obj_array_uniform(num_obs) #C[0] and C[1] don't care so uniform
C[2][:] = [punishment_value, reward_value] # we care about C[2], the reward observation

# D MATRIX
#D = utils.obj_array(len(num_states))
#D[0] = utils.onehot(init_state, num_states[0])
#D[1] = utils.onehot(init_tool, num_states[1])


"""Construct policies to limit the potential policies to one action at a time"""

def construct_policies(num_states, num_controls=None, policy_len=1, control_fac_idx=None, restriction=None):
    """
    Generate a ``list`` of policies.
    Parameters
    ----------
    num_states: ``list`` of ``int``
        ``list`` of the dimensionalities of each hidden state factor
    num_controls: ``list`` of ``int``, default ``None``
         ``list`` of the dimensionalities of each control state factor. If ``None``, then is automatically computed as the dimensionality of each hidden state factor that is controllable
    policy_len: ``int``, default 1
         temporal depth ("planning horizon") of policies
    control_fac_idx: ``list`` of ``int``
         ``list`` of indices of the hidden state factors that are controllable (i.e. those state factors ``i`` where ``num_controls[i] > 1``)

     Returns
     ----------
     policies: ``list`` of 2D ``numpy.ndarray``
        The returned array ``policies`` is a ``list`` that stores one policy per entry.
        A particular policy (``policies[i]``) has shape ``(num_timesteps, num_factors)`` 
        where ``num_timesteps`` is the temporal depth of the policy and ``num_factors`` is the number of control factors.
    """
    num_factors = len(num_states)

    # produce list of controllable hidden state factors 
    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1] 
        else:
            control_fac_idx = list(range(num_factors)) 

    # produce list of dimensionalities of each control state factor
    if num_controls is None:
        num_controls = [num_states[c_idx] if c_idx in control_fac_idx else 1 for c_idx in range(num_factors)]

    x = num_controls * policy_len 

    if restriction == None:
        policies = None

    elif restriction == "single_action":
        
        x = [num_controls[0]] * policy_len
        policies = list(itertools.product(*[list(range(i)) for i in x]))
        for pol_i in range(len(policies)):
            policies[pol_i] = np.tile(np.array(policies[pol_i]).reshape(policy_len, 1),num_factors)
            
    # if the restriction variable is an integer, this restricts the policies to this number
    elif isinstance(restriction,int):
        
        if restriction < np.product(x):
            policies = np.empty((restriction,len(x)))
            for i in range(len(x)):
                policies[:,i] = np.random.choice(np.arange(x[i]), restriction)   
            # we make sure that, all possible actions at time t=1 and t=2 are included (to prevent stupid misses when get close to target)
            first_step = np.unique(policies[:,:2], axis=0).tolist()
            for i in range(x[0]):
                for ii in range(x[1]):
                    if ([i,ii] in first_step) == False:
                        new_row = np.append(np.array([i,ii]), policies[-1,2:], axis=0).reshape(1,-1)
                        policies = np.append(policies, new_row, axis=0)
            policies = policies.tolist() 
        
        else:
            print("Warning - policy restriction size is greater than actual number of policies. Restriction Ignored.")
            policies = None
    
    # if the restriction variable is a list, this restricts the policies to this number
    elif isinstance(restriction, list):
        all_policies = list(itertools.product(*[list(range(i)) for i in x]))
        policies = []
        num_control_dims = len(num_controls)
        for pol_i in range(len(all_policies)):
            policy = np.array(all_policies[pol_i]).reshape(policy_len, num_factors)
            at_most_one_non_null = True
            for step in range(len(policy)):
                if sum(policy[step,:]==restriction)<num_control_dims-1:
                    at_most_one_non_null = False
            if at_most_one_non_null:
                policies.append(policy)  
    else:
        print("Policy restriction not understood. Provide an integer (numer of allowed policies) or a list (location of null actions).")
        policies = None
    
    return policies


"""Define agent and environment"""

# agent
policies = construct_policies(num_states, num_controls, policy_len, restriction="single_action")
my_agent = Agent(A=A, pA=pA, B = B, pB=pB, C = C, policy_len=policy_len, policies=policies, B_factor_list=B_factor_list)

# environment
my_env = Tool_create(init_state=init_room, init_tool=init_tool, reward_location=reward_location) # reduced_obs if using tool3.py 
num_runs = 1

def run_AIF_loop(agent, env, A_init, pA_init, reward_location, num_runs=1):
    
    filecount = 0
    
    for i in range(num_runs):
        
        # iter counter for a single run
        j = 0
        finish = False
                
        # reset A
        A = A_init 
        pA = pA_init
        my_agent.A=A
        my_agent.pA=pA
        my_agent.reset()
        env.reset(reward_location=reward_location, init_state=init_room, init_tool=init_tool)  
        
        # initial action of do nothing
        next_action=(0,0)
        my_agent.action=np.array(next_action)
        next_observation = env.step(next_action)
        qs_prev = my_agent.infer_states(next_observation)

        # make explicit
        print("Actions:", next_action, ":", room_action[int(next_action[1])], tool_action[int(next_action[0])], ", Returned Observations:", next_observation, ":", obs_tool[next_observation[0]],obs_room[next_observation[1]])
        print("Returned reward:", obs_reward[next_observation[2]])
        env.render(title="Run" + str(i+1)+ ". Start", save_in="/home/poppy/Documents/Active Inference/pymdp projects/pauls_files"+ str(filecount))
        
        # step the filenumber to save in right place
        filecount += 1
        
        # inner loop for the agent acting in the world during a single run
        while j < 100 and finish == False:
            
            qs = my_agent.infer_states(next_observation)
            q_pi, G, G1, G2, G3 = my_agent.infer_policies_factorized() # this is new method for factorised B
            
            qA=my_agent.update_A(next_observation)   
            qB=my_agent.update_B(qs_prev)
            qs_prev=qs
            
            next_action = my_agent.sample_action()
            print(next_action)
            next_observation = env.step(next_action) # IndexError: invalid index to scalar variable

            # make explicit
            print("Actions:", next_action, ":", room_action[int(next_action[1])], tool_action[int(next_action[0])], ", Returned Observations:", next_observation, ":", obs_tool[next_observation[0]],obs_room[next_observation[1]])
            print("Returned reward:", obs_reward[next_observation[2]])
            env.render(title="Run" + str(i+1)+ ".", save_in="/home/poppy/Documents/Active Inference/pymdp projects/pauls_files"+ str(filecount))
            
            # step the filenumber to save in right place
            filecount += 1
            j += 1
            
            if next_observation[2]==1:
                finish = True
                print("Reward found. Inference finished after",i, "steps.")
            
run_AIF_loop(my_agent, my_env, A_init, pA_init, reward_location, num_runs)
