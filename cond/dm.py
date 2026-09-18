import torch as th

class DM:
	def __init__(self, beta_1=1e-4, beta_T=0.02, T=1000):
		self.beta_1 = beta_1 
		self.beta_T = beta_T
		self.T = T
		self.beta = th.linspace(beta_1, beta_T, T)
		self.alpha = 1 - self.beta 
		self.alpha_bar = self.alpha.cumprod(-1)

	def q(self, x_0, t):
		'''
		x_0: input data minibatch
		t: timesteps minibatch

		x_t ~ q(x_t|x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)I )
		x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps

		Return:
			x_t, eps
		'''
		dev = t.device
		eps = th.randn_like(x_0)
		alpha_bar_t = self.alpha_bar.to(dev).gather(-1,t).view(-1,1,1,1)
		x_t = alpha_bar_t.sqrt()*x_0 + (1-alpha_bar_t).sqrt() * eps

		return x_t, eps 

	def p(self, x_t, t, y, net):
		'''
		x_t: input data minibatch 
		t: FIXED timestep minibatch
		net: Neural Network eps_theta(x_t,t,y)

		x_{t-1} ~ p(x{t-1}|x_t) = N(x_t; mu_theta, beta_t * I )
		mu_theta = (1/sqrt(alpha_t))*(x_t - beta_t * eps / sqrt(1-alpha_bar_t))
		z = torch.randn_like(x_t) if t[0] >= 1 else 0
		x_{t-1} = mu_theta + sqrt(beta_t) * z

		Return:
			x_t
		'''
		dev = t.device
		z = th.randn_like(x_t) if t[0] >= 1 else 0
		alpha_bar_t = self.alpha_bar.to(dev).gather(-1,t).view(-1,1,1,1)
		alpha_t = self.alpha.to(dev).gather(-1,t).view(-1,1,1,1)
		beta_t = self.beta.to(dev).gather(-1,t).view(-1,1,1,1)
		eps = net(x_t, t, y)

		mu_theta = (1/th.sqrt(alpha_t))*(x_t - beta_t * eps / th.sqrt(1-alpha_bar_t))
		x_t_prev = mu_theta + th.sqrt(beta_t) * z
		return x_t_prev

	def ddim_pred_x0(self, net, x_t, t, y):
		dev=x_t.device
		alpha_bar_t = self.alpha_bar.to(dev).gather(-1,t).view(-1,1,1,1)
		eps_pred = net(x_t, t, y)
		sample = (x_t - (1-alpha_bar_t).sqrt() * eps_pred) / (alpha_bar_t.sqrt())
		return sample, eps_pred

	def ddim_sample(self, eta, net, x_t, t, t_prev, y):
		dev = x_t.device
		alpha_bar_t = self.alpha_bar.to(dev).gather(-1, t).view(-1,1,1,1)
		alpha_bar_t_minus_one = self.alpha_bar.to(dev).gather(-1, t_prev).view(-1,1,1,1)
		sigma_t = eta * (((1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)).sqrt() * (1 - alpha_bar_t/alpha_bar_t_minus_one).sqrt())
		x_0, eps_pred = self.ddim_pred_x0(net, x_t, t, y)
		x_0 = th.clamp(x_0, -1, 1) # Enforce x_0 to be in data domain [-1,1]
		x_prev = alpha_bar_t_minus_one.sqrt() * x_0 + (1 - alpha_bar_t_minus_one - sigma_t**2).sqrt() * eps_pred + sigma_t * th.randn_like(x_t)
		return x_prev, x_0, eps_pred