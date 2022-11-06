import ptan
import time
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Optional

def batch_generator(exp_source, net, trajectory_size, ppo_epoches,
					batch_size, gamma, gae_lambda, device):
	trj_states = []
	trj_actions = []
	trj_rewards = []
	trj_dones = []
	last_done_index = None
	for (exp,) in exp_source:
		trj_states.append(exp.state)
		trj_actions.append(exp.action)
		trj_rewards.append(exp.reward)
		trj_dones.append(exp.done)
		if exp.done:
			last_done_index = len(trj_states)-1
		if len(trj_states) < trajectory_size:
			continue
		if last_done_index is None or last_done_index == len(trj_states)-1:
			continue

		trj_states_t = T.FloatTensor(trj_states).to(device)
		trj_actions_t = T.tensor(trj_actions).to(device)
		policy_t, trj_values_t = net(trj_states_t)
		trj_values_t = trj_values_t.squeeze()

		adv_t, ref_t = calc_adv_ref(trj_values_t.data.cpu().numpy(),
									trj_dones, trj_rewards, gamma, gae_lambda)
		adv_t = adv_t.to(device)
		ref_t = ref_t.to(device)

		logpolicy_t = F.log_softmax(policy_t, dim=1)
		old_logprob_t = logpolicy_t.gather(1, trj_actions_t.unsqueeze(-1)).squeeze(-1)
		adv_t = (adv_t - T.mean(adv_t)) / T.std(adv_t)
		old_logprob_t = old_logprob_t.detach()

		trj_len = len(trj_states) - 1
		trj_len -= trj_len % batch_size
		trj_len += 1
		indices = np.arange(0, trj_len-1)

		for i in range(ppo_epoches):
			np.random.shuffle(indices)
			for batch_indices in np.split(indices, trj_len // batch_size):
				yield (
					trj_states_t[batch_indices],
					trj_actions_t[batch_indices],
					adv_t[batch_indices],
					ref_t[batch_indices],
					old_logprob_t[batch_indices],
				)

		trj_states.clear()
		trj_actions.clear()
		trj_rewards.clear()
		trj_dones.clear()


def calc_adv_ref(values, dones, rewards, gamma, gae_lambda):
	last_gae = 0.0
	adv, ref = [], []

	for val, next_val, done, reward in zip(reversed(values[:-1]), reversed(values[1:]),
										reversed(dones[:-1]), reversed(rewards[:-1])):
		if done:
			delta = reward - val
			last_gae = delta
		else:
			delta = reward + gamma * next_val - val
			last_gae = delta + gamma * gae_lambda * last_gae
		adv.append(last_gae)
		ref.append(last_gae + val)
	adv = list(reversed(adv))
	ref = list(reversed(ref))
	return T.FloatTensor(adv), T.FloatTensor(ref)


class PPO(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(PPO, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU()
		)

		conv_out_size = self._get_conv_out(input_shape)
		self.actor = nn.Sequential(
			nn.Linear(conv_out_size, 64),
			nn.ReLU(),
			nn.Linear(64, n_actions)
		)
		self.critic = nn.Sequential(
			nn.Linear(conv_out_size, 32),
			nn.ReLU(),
			nn.Linear(32, 1)
		)

	def _get_conv_out(self, shape):
		o = self.conv(T.ones(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x):
		fx = x.float() / 1
		conv_out = self.conv(fx).view(fx.size()[0], -1)
		return self.actor(conv_out), self.critic(conv_out)

	def save_checkpoint(self, env_name):
		print("... saveing checkpoint ...")
		T.save(self.state_dict(), "saves/" + env_name + ".pt")

	def load_checkpoint(self, env_name):
		self.load_state_dict(T.load("saves/" + env_name + ".pt"))
