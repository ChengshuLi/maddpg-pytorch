import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent
from IPython import embed
from utils.cnn_networks import Net
import torch.nn as nn
from collections import OrderedDict
import gym

MSELoss = torch.nn.MSELoss()

class Actor(nn.Module):
    def __init__(self, encoder, action_dim, discrete_action, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.actor_head = MLPNetwork(512, action_dim,
                                     hidden_dim=hidden_dim,
                                     constrain_out=True,
                                     discrete_action=discrete_action)

    def forward(self, obs):
        return self.actor_head(self.encoder(obs))

class Critic(nn.Module):
    def __init__(self, encoder, total_action_dim, hidden_dim=256, has_goal=True):
        super().__init__()
        self.encoder = encoder
        self.actor_feat = nn.Sequential(
            nn.Linear(total_action_dim, 256),
            nn.ReLU()
        )
        self.has_goal = has_goal
        if self.has_goal:
            self.goal_feat = nn.Sequential(
                nn.Linear(4, 256),
                nn.ReLU()
            )
            critic_head_in = 512 + 256 + 256
        else:
            critic_head_in = 512 + 256

        self.critic_head = MLPNetwork(critic_head_in, 1,
                                      hidden_dim=hidden_dim,
                                      constrain_out=False)

    def forward(self, obs, action):
        obs_feat = self.encoder(obs)
        actor_feat = self.actor_feat(action)
        if self.has_goal:
            goal_feat = self.goal_feat(obs['goal'])
            feat = torch.cat([obs_feat, actor_feat, goal_feat], dim=1)
        else:
            feat = torch.cat([obs_feat, actor_feat], dim=1)
        return self.critic_head(feat)

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, observation_space, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types

        # arm_observation_space, head_cam_observation_space = \
        #     self.split_observation_space(observation_space)

        base_encoder = Net(observation_space)
        target_base_encoder = Net(observation_space)
        # arm_encoder =  Net(arm_observation_space)
        # target_arm_encoder =  Net(arm_observation_space)
        # head_cam_encoder = Net(head_cam_observation_space)
        # target_head_cam_encoder =  Net(head_cam_observation_space)

        base_policy = Actor(base_encoder, action_dim=2, discrete_action=False, hidden_dim=256)
        target_base_policy = Actor(target_base_encoder, action_dim=2, discrete_action=False, hidden_dim=256)

        # arm_policy = Actor(arm_encoder, action_dim=5, discrete_action=False, hidden_dim=256)
        # target_arm_policy = Actor(target_arm_encoder, action_dim=5, discrete_action=False, hidden_dim=256)

        # head_cam_policy = Actor(head_cam_encoder, action_dim=3, discrete_action=False, hidden_dim=256)
        # target_head_cam_policy = Actor(target_head_cam_encoder, action_dim=3, discrete_action=False, hidden_dim=256)

        # critic = Critic(base_encoder, total_action_dim=10, hidden_dim=256)
        # target_critic = Critic(target_base_encoder, total_action_dim=10, hidden_dim=256)
        has_goal = 'goal' in observation_space
        critic = Critic(base_encoder, total_action_dim=2, hidden_dim=256, has_goal=has_goal)
        target_critic = Critic(target_base_encoder, total_action_dim=2, hidden_dim=256)

        self.agents = [
            DDPGAgent(policy=base_policy,
                      target_policy=target_base_policy,
                      critic=critic,
                      target_critic=target_critic,
                      action_dim=2,
                      lr=lr,
                      discrete_action=False),
            # DDPGAgent(policy=arm_policy,
            #           target_policy=target_arm_policy,
            #           critic=critic,
            #           target_critic=target_critic,
            #           action_dim=5,
            #           lr=lr,
            #           discrete_action=False),
            # DDPGAgent(policy=head_cam_policy,
            #           target_policy=target_head_cam_policy,
            #           critic=critic,
            #           target_critic=target_critic,
            #           action_dim=3,
            #           lr=lr,
            #           discrete_action=True),
        ]
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def split_observation_space(self, observation_space):
        head_cam_observation_space = OrderedDict()
        arm_observation_space = OrderedDict()

        if "base_proprioceptive" in observation_space.spaces:
            head_cam_observation_space["base_proprioceptive"] = observation_space.spaces["base_proprioceptive"]
            arm_observation_space["base_proprioceptive"] = observation_space.spaces["base_proprioceptive"]

        if "arm_proprioceptive" in observation_space.spaces:
            arm_observation_space["arm_proprioceptive"] = observation_space.spaces["arm_proprioceptive"]

        if "rgb" in observation_space.spaces:
            head_cam_observation_space["rgb"] = observation_space.spaces["rgb"]

        if "depth" in observation_space.spaces:
            head_cam_observation_space["depth"] = observation_space.spaces["depth"]

        if "seg" in observation_space.spaces:
            head_cam_observation_space["seg"] = observation_space.spaces["seg"]

        if "wrist_rgb" in observation_space.spaces:
            arm_observation_space["wrist_rgb"] = observation_space.spaces["wrist_rgb"]

        if "wrist_depth" in observation_space.spaces:
            arm_observation_space["wrist_depth"] = observation_space.spaces["wrist_depth"]

        if "wrist_seg" in observation_space.spaces:
            arm_observation_space["wrist_seg"] = observation_space.spaces["wrist_seg"]

        head_cam_observation_space = gym.spaces.Dict(head_cam_observation_space)
        arm_observation_space = gym.spaces.Dict(arm_observation_space)

        return arm_observation_space, head_cam_observation_space

    def split_observations(self, observations):
        head_cam_observations = {}
        arm_observations = {}

        if "base_proprioceptive" in observations:
            head_cam_observations['base_proprioceptive'] = observations['base_proprioceptive']
            arm_observations['base_proprioceptive'] = observations['base_proprioceptive']

        if "arm_proprioceptive" in observations:
            arm_observations['arm_proprioceptive'] = observations['arm_proprioceptive']

        if "rgb" in observations:
            head_cam_observations["rgb"] = observations["rgb"]

        if "depth" in observations:
            head_cam_observations["depth"] = observations["depth"]

        if "seg" in observations:
            head_cam_observations["seg"] = observations["seg"]

        if "wrist_rgb" in observations:
            arm_observations["wrist_rgb"] = observations["wrist_rgb"]

        if "wrist_depth" in observations:
            arm_observations["wrist_depth"] = observations["wrist_depth"]

        if "wrist_seg" in observations:
            arm_observations["wrist_seg"] = observations["wrist_seg"]

        return arm_observations, head_cam_observations

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        result = []
        base_observations = observations[0]
        arm_observations, _ = self.split_observations(observations[1])
        _, head_cam_observations = self.split_observations(observations[2])

        result.append(self.agents[0].step(base_observations, explore=explore))
        result.append(self.agents[1].step(arm_observations, explore=explore))
        result.append(self.agents[2].step(head_cam_observations, explore=explore))

        return result

        # return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
        #                                                          observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            # if self.discrete_action: # one-hot encode action
            #     all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
            #                     zip(self.target_policies, next_obs)]
            # else:
            #     all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
            #                                                  next_obs)]
            all_trgt_acs = []
            for i, (pi, nobs) in enumerate(zip(self.target_policies, next_obs)):
                if i == 0:
                    nobs = nobs
                elif i == 1:
                    nobs, _ = self.split_observations(nobs)
                else:
                    _, nobs = self.split_observations(nobs)

                if self.agents[i].discrete_action:
                    all_trgt_acs.append(onehot_from_logits(pi(nobs)))
                else:
                    all_trgt_acs.append(pi(nobs))
            # trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
            trgt_vf_obs_in = next_obs[agent_i]
            trgt_vf_act_in = torch.cat(all_trgt_acs, dim=1)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
        # target_value = (rews[agent_i].view(-1, 1) + self.gamma *
        #                 curr_agent.target_critic(trgt_vf_in) *
        #                 (1 - dones[agent_i].view(-1, 1)))

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_obs_in, trgt_vf_act_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG':
            # vf_in = torch.cat((*obs, *acs), dim=1)
            vf_obs_in = obs[agent_i]
            vf_act_in = torch.cat(acs, dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        # actual_value = curr_agent.critic(vf_in)
        actual_value = curr_agent.critic(vf_obs_in, vf_act_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if agent_i == 0:
            curr_obs = obs[agent_i]
        elif agent_i == 1:
            curr_obs, _ = self.split_observations(obs[agent_i])
        else:
            _, curr_obs = self.split_observations(obs[agent_i])
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(curr_obs)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(curr_obs)
            curr_pol_vf_in = curr_pol_out

        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == 0:
                    ob = ob
                elif i == 1:
                    ob, _ = self.split_observations(ob)
                else:
                    _, ob = self.split_observations(ob)

                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            # vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
            vf_obs_in = obs[agent_i]
            vf_act_in = torch.cat(all_pol_acs, dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        # pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss = -curr_agent.critic(vf_obs_in, vf_act_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        # alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
        #              atype in env.agent_types]
        
        # three agents: base, arm, head camera
        # alg_types = [agent_alg for _ in range(3)]
        alg_types = [agent_alg]
        # for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
        #                                alg_types):
        #     num_in_pol = obsp.shape[0]
        #     if isinstance(acsp, Box):
        #         discrete_action = False
        #         get_shape = lambda x: x.shape[0]
        #     else:  # Discrete
        #         discrete_action = True
        #         get_shape = lambda x: x.n
        #     num_out_pol = get_shape(acsp)
        #     if algtype == "MADDPG":
        #         num_in_critic = 0
        #         for oobsp in env.observation_space:
        #             num_in_critic += oobsp.shape[0]
        #         for oacsp in env.action_space:
        #             num_in_critic += get_shape(oacsp)
        #     else:
        #         num_in_critic = obsp.shape[0] + get_shape(acsp)
        #     agent_init_params.append({'num_in_pol': num_in_pol,
        #                               'num_out_pol': num_out_pol,
        #                               'num_in_critic': num_in_critic})
        # init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
        #              'hidden_dim': hidden_dim,
        #              'alg_types': alg_types,
        #              'agent_init_params': agent_init_params,
        #              'discrete_action': discrete_action}
        init_dict = {
            'gamma': gamma,
            'tau': tau,
            'lr': lr,
            'alg_types': alg_types,
            'observation_space': env.observation_space,
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance