import pickle
import torch as T
import numpy as np
import collections

#from sub_envs.static import MEDAEnv
from sub_envs.dynamic import MEDAEnv
from sub_envs.map import MakeMap
from sub_envs.map import Symbols
from lib import common, ppo

def _is_touching(dstate, obj, map, dsize):
		i = 0
		while True:
			j = 0
			while True:
				if map[dstate[1]+j][dstate[0]+i] == obj:
					return True
				j += 1
				if j == dsize:
					break
			i += 1
			if i == dsize:
				break

		return False

def _compute_shortest_route(w, h, dsize, symbols,map, start):
	queue = collections.deque([[start]])
	seen = set([start])
#		print(self.map)
	while queue:
		path = queue.popleft()
#			print(path)
		x, y = path[-1]
		if _is_touching((x,y), symbols.Goal, map, dsize):
			return path
		for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
			if 0 <= x2 < (w-dsize+1) and 0 <= y2 < (h-dsize+1) and \
			(_is_touching((x2,y2), symbols.Dynamic_module, map, dsize) == False) and\
			(_is_touching((x2,y2), symbols.Static_module, map, dsize) == False) and\
			(x2, y2) not in seen:
				queue.append(path + [(x2, y2)])
				seen.add((x2, y2))
#		print("Bad map")
#		print(self.map)
	return False

if __name__ == "__main__":
	T.manual_seed(1)
	###### Set params ##########

	W = 8
	H = 8
	DSIZE = 3
	S_MODULES = 0
	D_MODULES = 0
	N_EPOCH = 1

	############################
	device = T.device('cpu')
	ENV_NAME = str(W)+str(H)+str(DSIZE)+str(S_MODULES)+str(D_MODULES) + "/" + str(N_EPOCH)
#	ENV_NAME = str(W)+str(H)+str(DSIZE)+str(S_MODULES)+str(D_MODULES) + ">0.9" + "/" + str(N_EPOCH)

	test_result = common.test(ENV_NAME, W, H, DSIZE, S_MODULES, D_MODULES)

	print(test_result)