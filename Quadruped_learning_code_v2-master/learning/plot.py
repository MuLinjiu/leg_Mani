import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

label1 = "Avg. Episode Reward"
label2 = "Avg. Episode Length"

host.set_xlabel("Iterations")
host.set_ylabel(label1)
par1.set_ylabel(label2)

color1 = 'tab:red'
color2 = 'tab:blue'
color3 = 'tab:green'

progress = pd.read_csv(
    'ray_results_ppo/End2End-1024-21-34/PPO/PPO_DummyQuadrupedEnv_59383_00000_0_2022-10-24_21-34-44/progress.csv')

reward = progress['episode_reward_mean']
length = progress['episode_len_mean']
ratio = reward / length

p1, = host.plot(progress['training_iteration'], reward, color=color1, label=label1)
p2, = par1.plot(progress['training_iteration'], length, color=color2, label=label2)

par1.yaxis.label.set_color(p2.get_color())

host.tick_params(axis='y', labelcolor=color1)
par1.tick_params(axis='y', labelcolor=color2)

plt.savefig("summary.png", bbox_inches='tight')
plt.show()
