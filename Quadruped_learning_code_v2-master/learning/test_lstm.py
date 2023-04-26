import os, ray, sys, time
import numpy as np
import pickle
import gym
import json
from ray import tune
from ray.rllib import SampleBatch
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.tune.trial import ExportFormat
# from envs.pmtg_task import QuadrupedGymEnv

from envs.running_task_cartesian import QuadrupedGymEnv
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ray.rllib.models import ModelCatalog

# from learning.rllib_helpers.fcnet_me import MyFullyConnectedNetwork
from learning.rllib_helpers.tf_lstm_net import LSTMModel

from matplotlib import pyplot as plt

model_dir = "ray_results_ppo/End2End-1024-22-19/PPO/PPO_DummyQuadrupedEnv_925ad_00000_0_2022-10-24_22-19-17"


# model_dir = "ray_results_ppo/End2End-1012-14-54/PPO/PPO_DummyQuadrupedEnv_6f020_00000_0_2022-10-12_14-54-23"

def run_sim(env, agent, count):
    rewards = []
    episode_lengths = []

    model_configs = config["model"]["custom_model_config"]
    d_model = config["model"]["lstm_cell_size"]

    for i in range(count):
        print("Current episode: {}, remaining episodes: {}".format(i + 1, count - i - 1))
        obs = env.reset()

        foot_pos = []
        # obs_arr = [np.copy(obs)]
        memory_states = [np.zeros([d_model], np.float32) for _ in range(2)]
        episode_reward = 0
        num_steps = 0
        infer_time = 0
        actions = []
        while True:
            start_time = time.time()
            action, memory_states, _ = agent.compute_single_action(obs, memory_states, explore=False)
            actions.append(action)
            infer_time += time.time() - start_time
            # action = agent.compute_action(obs)
            # action = np.array(env.action_space)
            obs, reward, done, info = env.step(action)
            foot_pos.append(np.clip(action, env.action_space.low, env.action_space.high))
            # obs_arr.append(np.copy(obs))
            # time.sleep(0.01)

            episode_reward += reward
            num_steps += 1
            if num_steps % 100 == 0:
                a = 0
            #     env.env.save_robot_cam_view()
            if done:
                print('episode reward:', episode_reward, "num_steps:", num_steps)
                print("---Avg exec time per second: %.4f seconds ---" % (infer_time / num_steps))
                episode_lengths.append(num_steps)
                rewards.append(episode_reward)
                print(info)
                # plot_foot_pos(foot_pos)
                # if num_steps < 1001:
                #     time.sleep(5)
                # df = pd.DataFrame(np.array(actions))
                # df.to_csv("/home/zhuochen/actions" + str(i) + ".csv", index=False)
                break
        # df = pd.DataFrame(np.array(obs_arr))
        # df.to_csv("/home/zhuochen/obs_vision_" + str(i) + ".csv", index=False)
    return rewards, episode_lengths


class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """

    def __init__(self, dummy_env_config):
        # self.env = QuadrupedGymEnv(render=True, time_step=0.001, action_repeat=20, obs_hist_len=3)
        self.env = QuadrupedGymEnv(render=True, time_step=0.001, action_repeat=10, obs_hist_len=1)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # print('\n', '*' * 80)
        # print(self.observation_space)

    def reset(self):
        """reset env """
        obs = self.env.reset()
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # print('step obs, rew, done, info', obs, rew, done, info)
        # NOTE: it seems pybullet torque control IGNORES joint velocity limits..
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if (obs < self.observation_space.low).any() or (obs > self.observation_space.high).any():
            print(obs)
            sys.exit()
        return obs, rew, done, info


register_env("quadruped_env", lambda _: DummyQuadrupedEnv(_))

# ModelCatalog.register_custom_model("my_model", MyFullyConnectedNetwork)
ModelCatalog.register_custom_model("my_model", LSTMModel)

config_path = os.path.join(model_dir, "params.pkl")

# config_path = "/home/zhuochen/params.pkl"

with open(config_path, "rb") as f:
    config = pickle.load(f)
config["num_workers"] = 0
config["num_gpus"] = 0
# config["model"] = {"custom_model": "my_model", "fcnet_activation": "tanh", "vf_share_layers": True,
#                    "custom_model_config": {"cam_seq_len": 10, "sensor_seq_len": 30, "action_seq_len": 1,
#                                            "cam_dim": (36, 36),
#                                            "is_training": False}}

config["num_envs_per_worker"] = 1
config["evaluation_config"] = {
    "explore": False,
    "env_config": {
        # Use test set to evaluate
        'mode': "test"}
}

ray.init()
agent = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv)

latest = 'checkpoint_000160/checkpoint-160'
# checkpoint = get_latest_model_rllib(model_dir)
checkpoint = os.path.join(model_dir, latest)
# checkpoint = "/home/zhuochen/checkpoint_120/checkpoint-120"
agent.restore(checkpoint)
env = agent.workers.local_worker().env

rewards, lengths = run_sim(env, agent, 5)
# generate(env, 5000)
print("Average reward: {}, average episode length: {}".format(np.mean(rewards), np.mean(lengths)))

ray.shutdown()
