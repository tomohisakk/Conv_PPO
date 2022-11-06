import os
import ptan
import time
import torch as T
import torch.optim as optim
import torch.nn.functional as F
from ignite.engine import Engine

#from sub_envs.static import MEDAEnv
from sub_envs.dynamic import MEDAEnv
from lib import common, ppo

class Params():
	lr = 1e-3
	entropy_beta = 0.01
	batch_size = 32
	ppo_epoches = 10

	w = 8
	h = 8
	dsize = 1
	s_modules = 0
	d_modules = 3
	importf = "881030.001/20"

	useGPU = False
	env_name = str(w)+str(h)+str(dsize)+str(s_modules)+str(d_modules)+str(lr)
	gamma = 0.99
	gae_lambda = 0.95
	ppo_eps =  0.2
	ppo_trajectory = 1029
	stop_test_reward = 10000
	stop_reward = None

params = Params()

if __name__ == "__main__":
	T.manual_seed(1)
	env = MEDAEnv(w=params.w, h=params.h, dsize=params.dsize, s_modules=params.s_modules, d_modules=params.d_modules)

	if params.useGPU == True:
		device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
	else:
		device = T.device('cpu')
	print("Device is ", device)

	net = ppo.PPO(env.observation_space, env.action_space).to(device)
#	net.load_checkpoint(params.importf)
	print(net)

	agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True,
								   preprocessor=ptan.agent.float32_preprocessor,
								   device=device)

	exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

#	optimizer = optim.Adam(net.parameters(), lr=params.lr, eps=1e-3)
	optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=0.9)

	scheduler = T.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

	if not os.path.exists("saves"):
		os.makedirs("saves")
	if not os.path.exists("saves/" + params.env_name):
		os.makedirs("saves/" + params.env_name)

	def process_batch(engine, batch):
		start_ts = time.time()
		optimizer.zero_grad()
		res = {}

		states_t, actions_t, adv_t, ref_t, old_logprob_t = batch
		policy_t, value_t = net(states_t)
		loss_value_t = F.mse_loss(value_t.squeeze(-1), ref_t)
		res['ref'] = ref_t.mean().item()

		logpolicy_t = F.log_softmax(policy_t, dim=1)

		prob_t = F.softmax(policy_t, dim=1)
		loss_entropy_t = (prob_t * logpolicy_t).sum(dim=1).mean()

		logprob_t = logpolicy_t.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
		ratio_t = T.exp(logprob_t - old_logprob_t)
		surr_obj_t = adv_t * ratio_t
		clipped_surr_t = adv_t * T.clamp(ratio_t, 1.0 - params.ppo_eps, 1.0 + params.ppo_eps)
		loss_policy_t = -T.min(surr_obj_t, clipped_surr_t).mean()

		loss_t = params.entropy_beta * loss_entropy_t + loss_policy_t + loss_value_t
		loss_t.backward()
		optimizer.step()

		res.update({
			"loss": loss_t.item(),
			"loss_value": loss_value_t.item(),
			"loss_policy": loss_policy_t.item(),
			"adv": adv_t.mean().item(),
			"loss_entropy": loss_entropy_t.item(),
			"time_batch": time.time() - start_ts,
		})

		return res

	engine = Engine(process_batch)

	common.setup_ignite(engine, params, exp_source, params.env_name, net, optimizer, scheduler ,extra_metrics=(
		'test_reward', 'avg_test_reward', 'test_steps'))

	engine.run(ppo.batch_generator(exp_source, net, params.ppo_trajectory,
									params.ppo_epoches, params.batch_size,
									params.gamma, params.gae_lambda, device=device))
