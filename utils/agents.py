from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
from .cnn_networks import Net
from IPython import embed
import torch.nn as nn
import torch

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
    def __init__(self, encoder, total_action_dim, hidden_dim=256):
        super().__init__()
        self.encoder = encoder
        self.actor_feat = nn.Sequential(
            nn.Linear(total_action_dim, 256),
            nn.ReLU()
        )
        self.goal_feat = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU()
        )
        self.critic_head = MLPNetwork(512 + 256, 1,
                                      hidden_dim=hidden_dim,
                                      constrain_out=False)
    
    def forward(self, obs, action):
        obs_feat = self.encoder(obs)
        actor_feat = self.actor_feat(action)
        goal_feat = self.goal_feat(obs['goal'])
        feat = torch.cat([obs_feat, actor_feat], dim=1)
        return self.critic_head(feat)


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, observation_space, action_dim, total_action_dim, hidden_dim=256,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.encoder = Net(observation_space)
        self.policy = Actor(self.encoder, action_dim, discrete_action, hidden_dim)
        self.critic = Critic(self.encoder, total_action_dim, hidden_dim)
        
        self.target_encoder = Net(observation_space)
        self.target_policy = Actor(self.target_encoder, action_dim, discrete_action, hidden_dim)
        self.target_critic = Critic(self.target_encoder, total_action_dim, hidden_dim)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(action_dim)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
