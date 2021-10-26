import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_10traj():
    PATH ='/home/giovani/dril/dril/trained_results/dril/'
    FILE_BETA =     'dril_LunarLanderContinuous-v2_ntraj=10_ensemble_lr=0.00025_lr=0.00025_bcep=2001_shuffle=sample_w_replace_quantile=0.98_cost_-1_to_1_seed=1Beta.perf'
    FILE_GAUSSIAN = 'dril_LunarLanderContinuous-v2_ntraj=10_ensemble_lr=0.00025_lr=0.00025_bcep=2001_shuffle=sample_w_replace_quantile=0.98_cost_-1_to_1_seed=1Gaussian.perf'

    data_beta = pd.read_csv(PATH + FILE_BETA)
    data_gaussian = pd.read_csv(PATH + FILE_GAUSSIAN)
    print(data_beta.columns)

    fig, ax = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(f'DRIL with {data_beta["num_trajs"][0]} demo trajectories')


    LIM = 27
    ax[0, 0].plot(data_beta['total_num_steps'][0:LIM], data_beta['train_reward'][0:LIM])
    ax[1, 0].plot(data_beta['total_num_steps'][0:LIM], data_beta['test_reward'][0:LIM])
    ax[2, 0].plot(data_beta['total_num_steps'][0:LIM], data_beta['u_reward'][0:LIM])

    ax[0, 0].set_title('Beta distribution')
    ax[0, 0].set_ylabel('train_reward')
    ax[1, 0].set_ylabel('test_reward')
    ax[2, 0].set_ylabel('u_reward')

    ax[0, 1].plot(data_gaussian['total_num_steps'][0:LIM], data_gaussian['train_reward'][0:LIM])
    ax[1, 1].plot(data_gaussian['total_num_steps'][0:LIM], data_gaussian['test_reward'][0:LIM])
    ax[2, 1].plot(data_gaussian['total_num_steps'][0:LIM], data_gaussian['u_reward'][0:LIM])

    ax[0, 1].set_title('Gaussian distribution')
    ax[0, 1].set_ylabel('train_reward')
    ax[1, 1].set_ylabel('test_reward')
    ax[2, 1].set_ylabel('u_reward')

    MIN_Y = 0
    MAX_Y = 300
    ax[0, 0].set_ylim([MIN_Y, MAX_Y])
    ax[1, 0].set_ylim([MIN_Y, MAX_Y])
    ax[0, 1].set_ylim([MIN_Y, MAX_Y])
    ax[1, 1].set_ylim([MIN_Y, MAX_Y])
    ax[2, 0].set_ylim([0, 1])
    ax[2, 1].set_ylim([0, 1])



    for i in range(3):
        for j in range(2):
            ax[i, j].grid()

    plt.show()


def plot_3traj():
    PATH ='/home/giovani/dril/dril/trained_results/dril/'
    #FILE_BETA =     'dril_LunarLanderContinuous-v2_ntraj=10_ensemble_lr=0.00025_lr=0.00025_bcep=2001_shuffle=sample_w_replace_quantile=0.98_cost_-1_to_1_seed=1Beta.perf'
    FILE_GAUSSIAN = 'dril_LunarLanderContinuous-v2_ntraj=3_ensemble_lr=0.00025_lr=0.00025_bcep=2001_shuffle=sample_w_replace_quantile=0.98_cost_-1_to_1_seed=1Gaussian.perf'

    #data_beta = pd.read_csv(PATH + FILE_BETA)
    data_gaussian = pd.read_csv(PATH + FILE_GAUSSIAN)

    fig, ax = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(f'DRIL with {data_gaussian["num_trajs"][0]} demo trajectories')


    LIM = 27
    ax[0, 1].plot(data_gaussian['total_num_steps'][0:LIM], data_gaussian['train_reward'][0:LIM])
    ax[1, 1].plot(data_gaussian['total_num_steps'][0:LIM], data_gaussian['test_reward'][0:LIM])
    ax[2, 1].plot(data_gaussian['total_num_steps'][0:LIM], data_gaussian['u_reward'][0:LIM])

    ax[0, 1].set_title('Gaussian distribution')
    ax[0, 1].set_ylabel('train_reward')
    ax[1, 1].set_ylabel('test_reward')
    ax[2, 1].set_ylabel('u_reward')

    MIN_Y = 0
    MAX_Y = 300
    ax[0, 0].set_ylim([MIN_Y, MAX_Y])
    ax[1, 0].set_ylim([MIN_Y, MAX_Y])
    ax[0, 1].set_ylim([MIN_Y, MAX_Y])
    ax[1, 1].set_ylim([MIN_Y, MAX_Y])
    ax[2, 0].set_ylim([0, 1])
    ax[2, 1].set_ylim([0, 1])

    for i in range(3):
        for j in range(2):
            ax[i, j].grid()

    plt.show()


def new_env():
    import gym
    env_name = 'BipedalWalkerHardcore-v3'
    env = gym.make(env_name)
    s = env.reset()
    n_games = 10
    for game in range(n_games):
        done = False
        score = 0
        env.reset()
        while not done:
            action = env.action_space.sample()
            s_, reward, done, info = env.step(action)
            score +=  reward

        print(f'Game {game:5} Score{score}')

    env.close()



def dril_bip_walker_aws():
    '''
    ,x,total_num_steps,train_loss,test_loss,train_reward,test_reward,num_trajs,u_reward
    0,,2048,0,0,-74.93485533333333,87.8016158,10,0.946501733870302
    1,,411648,0,0,77.0462851,25.827703500000002,10,0.9801137683718373
    2,,821248,0,0,112.6138792,158.09153099999997,10,0.980527641118573
    3,,1230848,0,0,151.1100403,170.3609024,10,0.9864231825399724



    '''
    pass

if __name__ == '__main__':
    #plot_10traj()
    #plot_3traj()
    new_env()
