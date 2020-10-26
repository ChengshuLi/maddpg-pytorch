import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from gibson2.envs.parallel_env import ParallelNavEnvironment
from gibson2.envs.locomotor_env import NavigateRandomEnv
from IPython import embed
from collections import defaultdict

# USE_CUDA = False  # torch.cuda.is_available()
USE_CUDA = True

# def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
#     def get_env_fn(rank):
#         def init_env():
#             env = make_env(env_id, discrete_action=discrete_action)
#             env.seed(seed + rank * 1000)
#             np.random.seed(seed + rank * 1000)
#             return env
#         return init_env
#     if n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn(0)])
#     else:
#         return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def batch_obs(observations, torchify=False):
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(obs[sensor])

    for sensor in batch:
        batch[sensor] = np.array(batch[sensor])
        if torchify:
            batch[sensor] = Variable(torch.Tensor(batch[sensor]), requires_grad=False) 
    return batch

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    # env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
    #                         config.discrete_action)

    config_file = '/cvgl2/u/chengshu/iGibson-MM/examples/configs/master_config.yaml'
    def load_env(env_mode):
        return NavigateRandomEnv(config_file=config_file,
                                 mode=env_mode,
                                 action_timestep=1/10.0,
                                 physics_timestep=1/40.0,
                                 random_height=True,
                                 automatic_reset=True,
                                 device_idx=0)

    env = [lambda: load_env("headless")
                  for env_id in range(config.n_rollout_threads)]
    env = ParallelNavEnvironment(env, blocking=False)
    obs = env.reset()

    # maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
    #                               adversary_alg=config.adversary_alg,
    #                               tau=config.tau,
    #                               lr=config.lr,
    #                               hidden_dim=config.hidden_dim)
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr)
    replay_buffer = ReplayBuffer(config.buffer_length,
                                 maddpg.nagents,
                                 env.observation_space,
                                 [2, 5, 3])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        # env has automatic reset
        # obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            print('step:', et_i)
            # rearrange observations to be per agent, and convert to torch Variable
            # torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
            #                       requires_grad=False)
            #              for i in range(maddpg.nagents)]

            obs_batched_torch = batch_obs(obs, torchify=True)
            obs_batched_torch = [obs_batched_torch for _ in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(obs_batched_torch, explore=True)
            # convert actions to numpy arrays
            # agent_actions: list of [N, A], N is number of parallel envs, A is action space, list length is N_agents
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            camera_actions = np.argmax(agent_actions[2], axis=1)
            camera_actions = np.expand_dims(camera_actions, axis=1)

            base_and_arm_actions = np.concatenate([agent_actions[0], agent_actions[1]], axis=1)
            # rearrange actions to be per environment
            # actions: list of [N_agents, A], N_agents is number of agents, A is action space, list length is N, number of parallel envs
            # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            env.set_camera(camera_actions)

            # next_obs, rewards, dones, infos = env.step(actions)
            outputs = env.step(base_and_arm_actions)
            next_obs, rewards, dones, infos = [list(x) for x in zip(*outputs)]
            obs_batched = batch_obs(obs, torchify=False)
            next_obs_batched = batch_obs(next_obs, torchify=False)
            rewards = np.array(rewards)
            dones = np.array(dones)

            # replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            replay_buffer.push(obs_batched, agent_actions, rewards, next_obs_batched, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA,
                                                      norm_rews=False)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                    print('update')
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
