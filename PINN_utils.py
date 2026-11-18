import torch
import sympy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy
import functools
import contextlib
import itertools
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


@contextlib.contextmanager
def freeze(model):
	params = list(model.parameters())
	original_states = [p.requires_grad for p in params]
	try:
		for param in params:
			param.requires_grad = False
		yield
	finally:
		for param, state in zip(params, original_states):
			param.requires_grad_(state)


def sample_points(N_colocacao, N_inicial, N_contorno, xmin=0, xmax=1, tmin=0, tmax=1):
	X_colocacao = torch.cat([
		tmin + (1 - torch.rand(N_colocacao, 1)) * (tmax - tmin),
		xmin + (1 - torch.rand(N_colocacao, 1)) * (xmax - xmin)
	], dim = 1)

	X_inicial = torch.cat([
		tmin + torch.zeros(N_inicial, 1),
		xmin + (1 - torch.rand(N_inicial, 1)) * (xmax - xmin)
	], dim = 1)

	X_contorno = torch.cat([
		tmin + (1 - torch.rand(N_contorno, 1)) * (tmax - tmin), 
		xmin + torch.zeros(N_contorno, 1)
	], dim = 1)

	return X_colocacao, X_inicial, X_contorno

def replace_least_error_points(points, new_points, errors):

	N_todos = points.shape[0]
	N_novos = new_points.shape[0]
	N_resto = N_todos - N_novos
	
	idxs = torch.topk(errors.abs().flatten(), N_resto, dim=0, largest=True, sorted=False).indices

	return torch.cat((points[idxs], new_points), axis=0)

class DomainSampler:
	def __init__(self, X_colocacao, X_inicial, X_contorno, xmin=0, xmax=1, tmin=0, tmax=1):
		self.xmin, self.xmax = xmin, xmax
		self.tmin, self.tmax = tmin, tmax

		self.N_colocacao = X_colocacao.shape[0]
		self.N_inicial = X_inicial.shape[0]
		self.N_contorno = X_contorno.shape[0]

		self.X_colocacao =  X_colocacao.requires_grad_(True)
		self.X_inicial = X_inicial.requires_grad_(True)
		self.X_contorno = X_contorno.requires_grad_(True)

		self.slice_colocacao = slice(None, self.N_colocacao)
		self.slice_inicial = slice(self.N_colocacao, self.N_colocacao + self.N_inicial)
		self.slice_contorno = slice(self.N_colocacao + self.N_inicial, None)


	@classmethod
	def create(cls, N_colocacao, N_inicial, N_contorno, xmin=0, xmax=1, tmin=0, tmax=1):
		return DomainSampler(*sample_points(N_colocacao, N_inicial, N_contorno, xmin, xmax, tmin, tmax), xmin, xmax, tmin, tmax)


	def get_all(self):
		return torch.cat((self.X_colocacao, self.X_inicial, self.X_contorno), dim=0)
	def get_colocacao(self):
		return self.X_colocacao
	def get_inicial(self):
		return self.X_inicial
	def get_contorno(self):
		return self.X_contorno

	def sample_batch(self, batch_colocacao, batch_inicial, batch_contorno):
		X_batch_colocacao = self.get_colocacao()[torch.randint(0, self.N_colocacao, (batch_colocacao,))].detach()
		X_batch_inicial = self.get_inicial()[torch.randint(0, self.N_inicial, (batch_inicial,))].detach()
		X_batch_contorno = self.get_contorno()[torch.randint(0, self.N_contorno, (batch_contorno,))].detach()

		return DomainSampler(X_batch_colocacao, X_batch_inicial, X_batch_contorno, self.xmin, self.xmax, self.tmin, self.tmax)
  

	def resample(self, model, pde, perc=0.15, perc_boundary=0.1):

		y_fake = model(self.get_all())

		L_f = pde.pde_loss(self.get_colocacao(), y_fake[self.slice_colocacao], make_reduction=False, create_graph=False)
		L_b1 = pde.IC_loss(self.get_inicial(), y_fake[self.slice_inicial], make_reduction=False, create_graph=False)
		L_b2 = pde.BC_loss(self.get_contorno(), y_fake[self.slice_contorno], make_reduction=False, create_graph=False)

		X_novos_colocacao, X_novos_inicial, X_novos_contorno = sample_points(
			int(self.N_colocacao * perc),
			int(self.N_inicial * perc_boundary),
			int(self.N_contorno * perc_boundary),
			self.xmin, self.xmax, self.tmin, self.tmax
		)

		self.X_colocacao = replace_least_error_points(self.get_colocacao(), X_novos_colocacao, L_f).detach().requires_grad_(True)
		self.X_inicial = replace_least_error_points(self.get_inicial(), X_novos_inicial, L_b1).detach().requires_grad_(True)
		self.X_contorno = replace_least_error_points(self.get_contorno(), X_novos_contorno, L_b2).detach().requires_grad_(True)





class PDE:
	def __init__(self, S0, mu_w=0.1, mu_g=0.5, S_wc=0.2, S_gr=0.1, S_injecao=0.9):
		self.S0 = S0

		self.mu_w = mu_w
		self.mu_g = mu_g
		self.S_wc = S_wc
		self.S_gr = S_gr
		self.S_injecao = S_injecao

		S = sympy.Symbol("S", real=True)
		self.f_prime = sympy.lambdify(S, sympy.diff(self.f(S), S))
		self.f_second = sympy.lambdify(S, sympy.diff(self.f(S), S, 2))

	def S_n(self, S):
		return (S - self.S_wc) / (1.0 - self.S_wc - self.S_gr);

	def k_rw(self, S):
		return self.S_n(S) ** 4.0

	def k_rg(self, S):
		return (1.0 - self.S_n(S)) ** 2.0

	def f(self, S):
		lambda_w = self.k_rw(S) / self.mu_w
		lambda_g = self.k_rg(S) / self.mu_g
		return lambda_w / (lambda_w + lambda_g)
	
	@functools.cache
	def get_inflection_point(self):
		def obj(S):
			return self.f_second(torch.tensor(S, dtype=torch.float64)).item()
	
		return scipy.optimize.bisect(obj, self.S_wc + 1e-5, 1.0 - self.S_gr - 1e-5)

	def f_prime_inv(self, f_val):
		def obj(S):
			return f_val - self.f_prime(S)

		return scipy.optimize.brentq(obj, self.get_inflection_point(), 1.0 - self.S_gr)

	def get_S_R(self, S_L):
		if S_L >= self.get_inflection_point(): # error?
			return S_L

		return scipy.optimize.brentq(lambda S: self.f_prime(S) - (self.f(S) - self.f(S_L)) / (S - S_L), self.get_inflection_point(), 1.0 - self.S_gr)

	def get_breaking_time(self, xs):
		S0_prime = lambda x, eps=1e-5: (self.S0(x + eps) - self.S0(x - eps)) / (2.0 * eps)
		ts_shock = -1 / (self.f_second(self.S0(xs)) * S0_prime(xs))
		return torch.min(ts_shock[ts_shock >= 0]).item()

	def get_shock_characteristic_start(self, xs):
		S0_prime = lambda x, eps=1e-5: (self.S0(x + eps) - self.S0(x - eps)) / (2.0 * eps)
		ts_shock = -1 / (self.f_second(self.S0(xs)) * S0_prime(xs))
		mask = ts_shock >= 0

		return xs[mask][torch.argmin(ts_shock[mask])]

	def get_breaking_point(self, xs):
		Tb = self.get_breaking_time(xs)
		Xb = self.get_shock_characteristic_start(xs)

		return Xb + self.f_prime(self.S0(Xb)) * Tb



	def pde_loss(self, X_batch, y_fake, make_reduction=True, create_graph=True):
		
		dy_dX = torch.autograd.grad(y_fake, X_batch, grad_outputs=torch.ones_like(y_fake), create_graph=create_graph)[0]
		
		dy_dt = dy_dX[:, 0:1]
		dy_dx = dy_dX[:, 1:]
		df_dx = self.f_prime(y_fake) * dy_dx

		loss = (dy_dt + df_dx) ** 2
		return torch.mean(loss) if make_reduction else loss

	def IC_loss(self, X_batch, y_fake, make_reduction=True, create_graph=True):
		loss = (y_fake - self.S0(X_batch[:, 1:])) ** 2
		return torch.mean(loss) if make_reduction else loss

	def BC_loss(self, X_batch, y_fake, make_reduction=True, create_graph=True):
		loss = (y_fake - self.S_injecao) ** 2
		return torch.mean(loss) if make_reduction else loss

	def IM_loss(self, X_batch, y_fake, make_reduction=True, create_graph=True):
		x_batch = X_batch[:, 1:2]
		t_batch = X_batch[:, 0:1]
		
		loss = (y_fake - self.S0(x_batch - self.f_prime(y_fake) * t_batch)) ** 2
		return torch.mean(loss) if make_reduction else loss

	def Welge_loss(self, X_interface, model_left, model_right, make_reduction=True, create_graph=True):

		u_r = model_right(X_interface.detach())
		with torch.no_grad():
			u_l = model_left(X_interface)
			RH_speed = (self.f(u_l) - self.f(u_r)) / (u_l - u_r)
		

		loss = (self.f_prime(u_r) - RH_speed) ** 2 # rarefaction is always at u_r, arbitrary for now
		return torch.mean(loss) if make_reduction else loss

	def continuity_loss(self, ts, descontinuidade, mask_domain, model_left, model_right, make_reduction=True, create_graph=True):

		pontos_descontinuidade = torch.column_stack((ts[mask_domain], descontinuidade[mask_domain])).detach()
		u_r = model_right(pontos_descontinuidade)
		u_l = model_left(pontos_descontinuidade)

		loss = (u_r - u_l) ** 2
		return torch.mean(loss) if make_reduction else loss

	def RH_loss(self, ts, descontinuidade, mask_domain, model_left, model_right, shock_start=0, shock_characteristic_start=0, create_graph=True):

		with torch.no_grad():
			pontos_descontinuidade = torch.column_stack((ts[mask_domain], descontinuidade[mask_domain]))
			u_l = model_left(pontos_descontinuidade)
			u_r = model_right(pontos_descontinuidade)

			RH_speed = (self.f(u_l) - self.f(u_r)) / (u_l - u_r)
			characteristic_speed = torch.full_like(RH_speed, self.f_prime(self.S0(shock_characteristic_start)))

			desired_speed = torch.where(pontos_descontinuidade[:, 0:1] < shock_start, characteristic_speed, RH_speed)

		speed = torch.autograd.grad(descontinuidade[mask_domain], ts, grad_outputs=torch.ones_like(descontinuidade[mask_domain]), create_graph=create_graph)[0][mask_domain]
		return torch.mean((desired_speed - speed) ** 2)










def make_mlp(input_dim, output_dim, hidden_dim, num_hidden):
	dims = [input_dim] + [hidden_dim] * num_hidden + [output_dim]

	layers = []
	for in_dim, out_dim in zip(dims[:-1], dims[1:]):
		layers.extend([nn.Linear(in_dim, out_dim), nn.Tanh()])

	return nn.Sequential(*layers[:-1])

class PINN(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim=32, num_hidden=4):
		super().__init__()
		self.net = make_mlp(input_dim, output_dim, hidden_dim, num_hidden)

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.num_hidden = num_hidden

	def forward(self, z):
		return self.net(z)

	def save(self, path):
		torch.save({
			"__state_dict": self.state_dict(),
			"input_dim": self.input_dim,
			"output_dim": self.output_dim,
			"hidden_dim": self.hidden_dim,
			"num_hidden": self.num_hidden,
		}, path)

	@classmethod
	def load(cls, path):
		data = torch.load(path, weights_only=True)

		model = cls(
			input_dim=data["input_dim"],
			output_dim=data["output_dim"],
			hidden_dim=data["hidden_dim"],
			num_hidden=data["num_hidden"],
		)

		model.load_state_dict(data["__state_dict"])
		model.eval()

		return model

class RHPiecewise_PINN(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim=32, num_hidden=3, t0=None, x0=None):
		super().__init__()
		self.net_IC = make_mlp(input_dim, output_dim, hidden_dim, num_hidden)
		self.net_BC = make_mlp(input_dim, output_dim, hidden_dim, num_hidden)
		self.net_RH = make_mlp(1, input_dim - 1, hidden_dim, num_hidden)

		self.t0 = t0
		self.x0 = x0

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.num_hidden = num_hidden

	def get_discontinuity(self, t):
		if self.t0 is not None and self.x0 is not None:
			with torch.no_grad():
				x0_fake = self.net_RH(torch.tensor([self.t0]))[0]
			return self.net_RH(t) + self.x0 - x0_fake
		else:
			return self.net_RH(t)

	def forward(self, z):
		descontinuidade = self.get_discontinuity(z[:, 0:1])
		return torch.where(z[:, 1:] > descontinuidade, self.net_IC(z), self.net_BC(z))

	def save(self, path):
		torch.save({
			"__state_dict": self.state_dict(),
			"t0": self.t0,
			"x0": self.x0,
			"input_dim": self.input_dim,
			"output_dim": self.output_dim,
			"hidden_dim": self.hidden_dim,
			"num_hidden": self.num_hidden,
		}, path)

	@classmethod
	def load(cls, path):
		data = torch.load(path, weights_only=True)

		model = cls(
			input_dim=data["input_dim"],
			output_dim=data["output_dim"],
			hidden_dim=data["hidden_dim"],
			num_hidden=data["num_hidden"],
			t0=data["t0"],
			x0=data["x0"],
		)

		model.load_state_dict(data["__state_dict"])
		model.eval()

		return model






def get_reference_sol(path):

	with open(path, "r") as f:
		xs = np.fromstring(f.readline().strip().rstrip(','), sep=',')
		ts = np.fromstring(f.readline().strip().rstrip(','), sep=',')
		res = np.fromstring(f.read().strip().rstrip(','), sep=',')

	res = res.reshape((len(ts), len(xs)))

	return ts, xs, res