import math
import pprint

import gym
from matplotlib import pyplot as plt

import rsoccer_gym

# Using VSS Single Agent env
env = gym.make('VSSMA-v0')
env.reset()
# Run for 1 episode and print reward at the end
v_list = []
for i in range(5):
    done = False
    while not done:
        # Step using random actions
        action = [[0,0],[0,0],[0.5,0.5]]
        # action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print("=============================")
        pprint.pprint(_)
        pprint.pprint(env.reward_shaping_total)
        pprint.pprint(env.individual_reward)
        pprint.pprint(reward)


        for i in range(3):
            vx=next_state[0][8+7*i]
            vy=next_state[0][9+7*i]
            v = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
            print("V{}:{}".format(i,v))
            if v < 0.1:
                v=-0.1
            v_list.append(v)
        env.render()
print(v_list)
plt.scatter(range(len(v_list)), v_list, s=1)
plt.show()