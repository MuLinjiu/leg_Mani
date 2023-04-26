import os, pickle
from datetime import datetime
import numpy as np
import gym
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.rllib.agents import ppo
from envs.MPC_Nav_env import QuadrupedGymEnv
from ray.rllib.models.tf.tf_action_dist import SquashedGaussian
from ray.rllib.models import ModelCatalog

# from usc_learning.learning.rllib_helpers.tf_lstm_net import LSTMModel
from learning.rllib_helpers.mpc_vision_net import LSTMModel
# from learning.rllib_helpers.fcnet_me import MyFullyConnectedNetwork
# from usc_learning.learning.rllib_helpers.partial_filter import PartialFilter, PartialFilter2

import sys

difficulty = 1


def on_train_result(info):
    global difficulty
    result = info["result"]

    target = 12

    if result["episode_reward_mean"] > target and result["episode_len_mean"] > 800:
        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.env.env.env.increase_difficulty()))
        difficulty += 1
        with open("record.txt", "a") as record:
            record.write("Difficulty increase to {} at iteration {}".format(difficulty, result["training_iteration"]))


"""
Notes:
Make sure to pass the following in config (or significantly increases training time with lower returns)
    "observation_filter": "MeanStdFilter",
    -(this is similar to stable-baseliens vectorized environment, keeps running mean and std of observations)
*OR*
VecNormalize env first, and use that in rllib (this will fix the observation/reward scaling like in SB)
"""
monitor_dir = datetime.now().strftime("MPC-%m%d-%H-%M") + '/'
SAVE_RESULTS_PATH = os.path.join('./ray_results_ppo', monitor_dir)

ALG = "PPO"

USE_IMITATION_ENV = False
# use VecNormalize
USING_VEC_ENV = False

script_dir = os.path.dirname(os.path.abspath(__file__))

save_path = os.path.join(script_dir, SAVE_RESULTS_PATH)
vec_stats = os.path.join(save_path, "vec_normalize.pkl")
os.makedirs(save_path, exist_ok=True)


# copy_files(save_path)


class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """

    def __init__(self, env_config):
        self.env = QuadrupedGymEnv(time_step=0.001, action_repeat=30, obs_hist_len=1)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        """reset env """
        obs = self.env.reset()
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
        # return obs

    def step(self, action):
        """step env """
        obs, rew, done, info = self.env.step(action)
        # print(np.isnan(action).any())
        # print('step obs, rew, done, info', obs, rew, done, info)
        # NOTE: it seems pybullet torque control IGNORES joint velocity limits..
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if (obs < self.observation_space.low).any() or (obs > self.observation_space.high).any():
            print("!!!!!!!!!!!")
            print(obs, self.observation_space.low, self.observation_space.high)
            sys.exit()
        return obs, rew, done, info


ray.init()

ModelCatalog.register_custom_action_dist("SquashedGaussian", SquashedGaussian)

ModelCatalog.register_custom_model("my_model", LSTMModel)
# model = {"custom_model": "my_model", "fcnet_activation": "tanh", "vf_share_layers": True,
#          "custom_model_config": {"cam_seq_len": 10, "sensor_seq_len": 30, "action_seq_len": 1, "cam_dim": (36, 36),
#                                  "is_training": True}}
# model = {"fcnet_hiddens": [32, 32, 32, 32], "fcnet_activation": "relu", "vf_share_layers": True}
# model = {"fcnet_hiddens": [512, 512], "fcnet_activation": "tanh", "vf_share_layers": True}
model = {"custom_model": "my_model", "custom_model_config": {}}

#################################################################################################
### PPO (currently data stays in buffer for a while?)
#################################################################################################
config = ppo.DEFAULT_CONFIG.copy()
"""
Notes:

seems simple_optimizer should be False

# most recent test was 128
"""
NUM_CPUS = 15

num_samples_each_worker = int(6000 / NUM_CPUS)
# num_samples_each_worker = 48

train_batch_size = NUM_CPUS * num_samples_each_worker

config_PPO = {"env": DummyQuadrupedEnv,
              "num_gpus": 1,
              "num_workers": NUM_CPUS,
              "num_envs_per_worker": 1,  # to test if same as SB
              "lr": 1e-4,
              # "monitor": True,
              "model": model,
              "train_batch_size": train_batch_size,  # 4096, #n_steps,
              "num_sgd_iter": 10,
              "sgd_minibatch_size": 256,  # 256,# WAS 64#nminibatches, ty 256 after
              "rollout_fragment_length": 400,  # 1024,
              "clip_param": 0.2,
              # "use_kl_loss": False,
              "vf_clip_param": 1,  # 0.2,#10, # in SB, this is default as the same as policy clip_param
              "vf_loss_coeff": 0.5,
              "lambda": 0.95,  # lam in SB
              "grad_clip": 0.5,  # max_grad_norm in SB
              "kl_coeff": 0.2,  # no KL in SB, look into this more
              "kl_target": 0.01,
              "entropy_coeff": 0.0,
              "observation_filter": "MeanStdFilter",  # THIS IS NEEDED IF NOT USING VEC_NORMALIZE
              # "observation_filter": PartialFilter2,
              "clip_actions": True,
              # "vf_share_layers": True,  # this is true in SB # TODO: test this, tune the vf_clip_param
              "normalize_actions": True,  # added
              "preprocessor_pref": "rllib",  # what are these even doing? anything?
              # "batch_mode": "complete_episodes", #"truncate_episodes",
              "batch_mode": "truncate_episodes",
              # "custom_preprocessor": "NoPreprocessor"

              # this seemed to have an effect
              "no_done_at_end": True,
              # "soft_horizon": True,
              # "horizon": None, # changing from 501
              "shuffle_buffer_size": NUM_CPUS * num_samples_each_worker,  # 4096,
              # "callbacks": {
              #     "on_train_result": on_train_result,
              # },
              "framework": "tf2",
              "eager_tracing": True,
              # "gamma": 0.95,
              }
config.update(config_PPO)


def stopper(trial_id, result):
    return result["episode_reward_mean"] > 15 and result["episode_len_mean"] > 900


reporter = CLIReporter(max_progress_rows=10)
tune.run("PPO", config=config,
         local_dir=SAVE_RESULTS_PATH,
         checkpoint_freq=50,
         verbose=2,
         checkpoint_at_end=True,
         # stop=stopper,
         stop={"timesteps_total": 18000000},
         progress_reporter=reporter
         )