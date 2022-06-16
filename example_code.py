import math
import pprint
import time

import gym
from gym import wrappers
from matplotlib import pyplot as plt

import rsoccer_gym

# Using VSS Single Agent env
env = gym.make('VSSMA-v0')
env = wrappers.RecordVideo(env, './videos/' + str(time.time()) + '/')
# Run for 1 episode and print reward at the end
v_list = []
max_step=100
for i in range(10):
    env.reset()
    done = False
    episode_step=0
    while not done and episode_step<max_step:
        episode_step+=1
        # Step using random actions
        action = [[0,0.],[1,1],[0,0]]
        # action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        #print("=============================")
        #pprint.pprint(_)
        #pprint.pprint(env.reward_shaping_total)
        # pprint.pprint(env.individual_reward["robot_0"]["robot_to_robot"])
        #print("!!!!!!!!!!!!!!!!!!!!")
        #pprint.pprint(env.individual_reward["robot_0"])
        pprint.pprint(env.individual_reward)
        # pprint.pprint(reward)

        # ball_x = next_state[0][0]
        # ball_y = next_state[0][1]
        # print(f"{ball_x},{ball_y}")

        for i in range(3):
            vx=next_state[0][8+7*i]
            vy=next_state[0][9+7*i]
            v = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
            #print("V{}:{}".format(i,v))
            if v < 0.1:
                v=-0.1
            v_list.append(v)
        env.render()

#print(v_list)
# plt.scatter(range(len(v_list)), v_list, s=1)
# plt.show()