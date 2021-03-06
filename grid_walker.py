#!/usr/bin/env python

#gridwalker.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#31 January 2018
#
#this was adapted from http://outlace.com/rlpart3.html  
#to execute:    ./gridwalker.py

#imports
import numpy as np
import copy
import random

#initialize the environment = dict containing all constants that describe the system
def initialize_environment(grid_size, init):
    actions = [0, 1, 2, 3]
    acts = ['up', 'down', 'left', 'right']
    objects = ['agent', 'goal', 'pit', 'wall']
    max_moves = grid_size**2
    environment = {'actions':actions, 'acts':acts, 'objects':objects, 'grid_size':grid_size,
        'max_moves':max_moves, 'init':init}
    return environment

#initialize state = dict containing x,y coordinates of all objects in the system,
#with agent's location fixed or random
def initialize_state(environment):
    wall = {'x':1, 'y':4}
    pit  = {'x':4, 'y':2}
    goal = {'x':4, 'y':4}
    grid_size = environment['grid_size']
    init = environment['init']
    if (init == 'fixed'):
        agent = {'x':0, 'y':0}
    if (init == 'random_agent'):
        while (True):
            agent = {'x':np.random.randint(0, grid_size), 'y':np.random.randint(0, grid_size)}
            if (agent != wall):
                if (agent != pit):
                    if (agent != goal):
                        break
    state = {'agent':agent, 'wall':wall, 'pit':pit, 'goal':goal}
    return state

#move agent...agent only moves if it doesnt hit wall or boundary
def move_agent(state, action, environment):
    state_next = copy.deepcopy(state)
    agent = copy.deepcopy(state['agent'])
    grid_size = environment['grid_size']
    act = environment['acts'][action]
    if (act == 'up'):
        if (agent['y'] < grid_size-1):
            agent['y'] += 1
    if (act == 'down'):
        if (agent['y'] > 0):
            agent['y'] -= 1
    if (act == 'left'):
        if (agent['x'] > 0):
            agent['x'] -= 1
    if (act == 'right'):
        if (agent['x'] < grid_size-1):
            agent['x'] += 1
    wall = state['wall']
    if (agent != wall):
        state_next['agent'] = agent
    return state_next

#generate 2D string array showing locations of all objects
def make_grid(state, environment):
    grid_size = environment['grid_size']
    grid = np.zeros((grid_size, grid_size), dtype='string')
    objects = environment['objects']
    for object in objects:
        xy = state[object]
        x = xy['x']
        y = xy['y']
        grid[y, x] = object[0].upper()
        if (object == 'goal'):
            if (state['agent'] == state['goal']):
                grid[y, x] = '*'
        if (object == 'pit'):
            if (state['agent'] == state['pit']):
                grid[y, x] = '@'
    return grid

#get reward
def get_reward(current_state, previous_state):
    if (current_state['agent'] == current_state['goal']):
        #agent is at goal
        return 10
    if (current_state['agent'] == current_state['pit']):
        #agent is in pit
        return -10
    if (current_state == previous_state):
        #agent was blocked by a wall or boundary
        return -3
    return -1

#check game state = running. hit goal, hit pit, too many moves
def get_game_state(state, N_moves, environment):
    agent = state['agent']
    goal = state['goal']
    pit = state['pit']
    max_moves = environment['max_moves']
    game_state = 'running'
    if (agent == goal):
        game_state = 'goal'
    if (agent == pit):
        game_state = 'pit'
    if (N_moves > max_moves):
        game_state = 'max_moves'
    return game_state

#convert state into a numpy array of agents' x,y coordinates
def state2vector(state, environment):
    agent = state['agent']
    x = agent['x']
    y = agent['y']
    xy = np.array([x, y])
    return xy.reshape(1, len(xy))

#initialize the memories queue with a buncha random moves
def initialize_memories(environment, memories_size):
    from collections import deque
    memories = deque(maxlen=memories_size)
    state = initialize_state(environment)
    N_moves = 0
    while (len(memories) < memories_size):
        state_vector = state2vector(state, environment)
        actions = environment['actions']
        action = np.random.choice(actions)
        state_next = move_agent(state, action, environment)
        reward = get_reward(state_next, state)
        game_state = get_game_state(state_next, N_moves, environment)
        memories.append((state, action, reward, state_next, game_state))
        if (game_state == 'running'):
            state = state_next
            N_moves += 1
        else:
            state = initialize_state(environment)
            N_moves = 0
    return memories

#build neural network
def build_model(N_inputs, grid_size, N_outputs):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.optimizers import RMSprop
    model = Sequential()
    layer_size = grid_size**2
    model.add(Dense(layer_size, input_shape=(N_inputs,)))
    model.add(Activation('relu'))
    model.add(Dense(layer_size))
    model.add(Activation('relu'))
    model.add(Dense(N_outputs))
    model.add(Activation('linear'))
    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)
    return model

#train model
def train(environment, model, N_training_games, gamma, memories_size, batch_size, debug=False):
    epsilon = 1.0
    for N_games in range(N_training_games):
        state = initialize_state(environment)
        state_vector = state2vector(state, environment)
        N_inputs = state_vector.shape[1]
        #initialize memory of random movements by agent
        memories = initialize_memories(environment, memories_size)
        experience_replay = True
        N_moves = 0
        if (N_games > N_training_games/10):
            #agent random walks for first 100 games, after which epsilon slowly down to 0.1
            if (epsilon > 0.1):
                epsilon -= 1.0/(N_training_games/2)
        game_state = get_game_state(state, N_moves, environment)
        while (game_state == 'running'):
            state_vector = state2vector(state, environment)
            #predict this turn's possible rewards Q
            Q = model.predict(state_vector, batch_size=1)
            #choose best action
            if (np.random.random() < epsilon):
                #choose random action
                action = np.random.choice(environment['actions'])
            else:
                #choose best action
                action = np.argmax(Q)
            #get next state
            state_next = move_agent(state, action, environment)
            state_vector_next = state2vector(state_next, environment)
            #predict next turn's possible rewards
            Q_next = model.predict(state_vector_next, batch_size=1)
            max_Q_next = np.max(Q_next)
            reward = get_reward(state_next, state)
            game_state = get_game_state(state_next, N_moves, environment)
            #add next turn's discounted reward to this turn's predicted reward
            Q[0, action] = reward
            if (game_state == 'running'):
                Q[0, action] += gamma*max_Q_next
                grid = make_grid(state_next, environment)
            else:
                if (debug):
                    print '======================='
                    print 'game number = ', N_games
                    print 'move number = ', N_moves
                    print 'action = ', environment['acts'][action]
                    grid = make_grid(state_next, environment)
                    print np.rot90(grid.T)
                    print 'reward = ', reward
                    print 'epsilon = ', epsilon
                    print 'game_state = ', game_state
            if (experience_replay):
                #train model on randomly selected past experiences
                memories.append((state, action, reward, state_next, game_state))
                memories_sub = random.sample(memories, batch_size)
                statez = [m[0] for m in memories_sub]
                actionz = [m[1] for m in memories_sub]
                rewardz = [m[2] for m in memories_sub]
                statez_next = [m[3] for m in memories_sub]
                game_onz = [m[4] for m in memories_sub]
                state_vectorz = np.array([state2vector(s, environment) for s in statez]).reshape(batch_size, N_inputs)
                Qz = model.predict(state_vectorz, batch_size=batch_size)
                state_vectorz_next = np.array([state2vector(s, environment) for s in statez_next]).reshape(batch_size, N_inputs)
                Qz_next = model.predict(state_vectorz_next, batch_size=batch_size)
                for idx in range(batch_size):
                    reward = rewardz[idx]
                    max_Q_next = np.max(Qz_next[idx])
                    action = actionz[idx]
                    Qz[idx, action] = reward
                    if (game_onz[idx] == 'running'):
                        Qz[idx, action] += gamma*max_Q_next
                model.fit(state_vectorz, Qz, batch_size=batch_size, epochs=1, verbose=0)
            else:
                #teach model about current action & reward
                model.fit(state_vector, Q, batch_size=1, epochs=1, verbose=0)
            state = state_next
            N_moves += 1
    return model

#test model
def test_model(model, environment, display_stats=False):
    acts = environment['acts']
    initial_state = initialize_state(environment)
    if (display_stats):
        grid = make_grid(initial_state, environment)
        print 'initial state:'
        print np.rot90(grid.T)
        print '======================='
    N_moves = 0
    state = initial_state.copy()
    game_state = get_game_state(state, N_moves, environment)
    while (game_state == 'running'):
        state_vector = state2vector(state, environment)
        Q = model.predict(state_vector, batch_size=1)
        action = np.argmax(Q)
        state_next = move_agent(state, action, environment)
        N_moves += 1
        grid = make_grid(state_next, environment)
        reward = get_reward(state_next, state)
        game_state = get_game_state(state_next, N_moves, environment)
        if (display_stats):
            print(' move : %s    action: %s' %(N_moves, acts[action]))
            print('reward: %s' %reward)
            print np.rot90(grid.T)
            if (game_state != 'running'):
                print('game_state: %s' %(game_state))
        state = state_next
    final_state = state
    return initial_state, final_state, N_moves, game_state
