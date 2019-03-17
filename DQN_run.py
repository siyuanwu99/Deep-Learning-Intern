import gym
import numpy as np 
import tensorflow as tf 
from DQN_brain import DQN 

def run():
    step = 0
    for i in range(0,10):
        # initial
        observation = env.reset()

        while True:
            env.render()
            action = RL._choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            print("Episode:\b", i,"  Action:\b", action, "\b   Reward:\b", reward)
            RL._store_transition(observation, action, reward, observation_)
            RL.learn()
            observation = observation_
            if done:
                break
            step += 1
        print("\n\n>>  Game over!!  <<\n\n")
        



if __name__ == "__main__":
    # maze game
    env = gym.make("Pendulum-v0")
    print("N_actions:\t", env.action_space)
    print("N_observation:\t", env.observation_space)
   
    RL = DQN(2, 4
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iteration=200,
                      memory_size=2000,
                      is_output_graph=True
                      )
    run()
    RL.view_graph()




