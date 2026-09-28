import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class Emitter(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_hidden_to_mu = nn.Linear(hidden_dim, x_dim)
        self.lin_hidden_to_sigma = nn.Linear(hidden_dim, x_dim)
        self.log_df = nn.Parameter(torch.ones(x_dim) * 1.5)
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        hidden = torch.tanh(self.lin_z_to_hidden(z_t))
        mu = self.lin_hidden_to_mu(hidden)
        sigma = self.softplus(self.lin_hidden_to_sigma(hidden))
        df = 2.0 + self.softplus(self.log_df)
        return df, mu, sigma


class GatedTransition(nn.Module):
    # maps z_{t-1} -> parameters of p(z_t | z_{t-1})
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.lin_z_gate = nn.Linear(z_dim, hidden_dim)
        self.lin_z_gate_reverse = nn.Linear(hidden_dim, z_dim)

        self.lin_z_transition = nn.Linear(z_dim, hidden_dim)
        self.lin_z_transition_reverse = nn.Linear(hidden_dim, z_dim)

        self.lin_z_mu = nn.Linear(z_dim, z_dim)
        nn.init.eye_(self.lin_z_mu.weight)
        nn.init.zeros_(self.lin_z_mu.bias)
        self.lin_z_sigma = nn.Linear(z_dim, z_dim)

        self.act = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_prev):
        # input: z_prev of shape (batch, z_dim)
        # output: mean, std of shape (batch, z_dim) each
        g_t = self.act(self.lin_z_gate(z_prev))
        g_t = self.sigmoid(self.lin_z_gate_reverse(g_t))

        h_t = self.act(self.lin_z_transition(z_prev))
        h_t = self.lin_z_transition_reverse(h_t)

        mu_t = (1 - g_t) * self.lin_z_mu(z_prev) + g_t * h_t
        sigma_t = self.softplus(self.lin_z_sigma(self.act(h_t)))
        return mu_t, sigma_t


class Combiner(nn.Module):
    # maps (z_{t-1}, h_right_t) -> parameters of q(z_t | z_{t-1}, x_{1:T})
    def __init__(self, z_dim, rnn_hidden_dim):
        super().__init__()
        self.lin_mu = nn.Linear(rnn_hidden_dim, z_dim)
        self.lin_sigma = nn.Linear(rnn_hidden_dim, z_dim)
        self.lin_z = nn.Linear(z_dim, rnn_hidden_dim)
        self.act = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_prev, h_right):
        # inputs: z_prev (batch, z_dim), h_right (batch, rnn_hidden_dim)
        # output: mean, std of shape (batch, z_dim) each
        h_combined = 0.5 * (self.act(self.lin_z(z_prev)) + h_right)
        mu_t = self.lin_mu(h_combined)
        sigma_t = self.softplus(self.lin_sigma(h_combined))
        return mu_t, sigma_t


class DMM(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, rnn_hidden_dim, rnn_num_layers):
        super().__init__()
        self.emitter = Emitter(z_dim, hidden_dim, x_dim)
        self.transition = GatedTransition(z_dim, hidden_dim)
        self.combiner = Combiner(z_dim, rnn_hidden_dim)
        self.rnn = nn.GRU(x_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True)

        # learnable initial latent state
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        # learnable initial RNN hidden state
        self.h_0 = nn.Parameter(torch.zeros(rnn_num_layers, 1, rnn_hidden_dim))

        self.z_dim = z_dim

    def model(self, x):
        # generative model: p(x_{1:T}, z_{1:T})
        # input: x of shape (batch, T, x_dim)
        pyro.module("dmm", self)
        batch_size, T, _ = x.shape

        z_prev = self.z_0.expand(batch_size, self.z_dim)

        with pyro.plate("data", batch_size):
            for t in range(T):
                # get transition distribution parameters from z_prev
                z_mean, z_std = self.transition(z_prev)

                # sample z_t from the prior
                z_t = pyro.sample(f"z_{t}", dist.Normal(z_mean, z_std).to_event(1))

                # get emission distribution parameters from z_t
                df, x_mean, x_std = self.emitter(z_t)

                # sample (or score) the observation
                pyro.sample(f"x_{t}", dist.StudentT(df, x_mean, x_std).to_event(1),
                            obs=x[:, t, :])

                z_prev = z_t

    def guide(self, x):
        # structured variational posterior: q(z_{1:T} | x_{1:T}) = prod_t q(z_t | z_{t-1}, x_{1:T})
        # input: x of shape (batch, T, x_dim)
        pyro.module("dmm", self)
        batch_size, T, _ = x.shape

        # run GRU on reversed sequence so h_right[:, t, :] encodes x_t through x_T
        h_0 = self.h_0.expand(-1, batch_size, -1).contiguous()
        h_right, _ = self.rnn(torch.flip(x, dims=[1]), h_0)
        h_right = torch.flip(h_right, dims=[1])  # (batch, T, rnn_hidden_dim)

        z_prev = self.z_0.expand(batch_size, self.z_dim)

        with pyro.plate("data", batch_size):
            for t in range(T):
                # combine z_{t-1} and h_right_t to get posterior parameters
                z_mean, z_std = self.combiner(z_prev, h_right[:, t, :])

                z_t = pyro.sample(f"z_{t}", dist.Normal(z_mean, z_std).to_event(1))

                z_prev = z_t


    def infer(self, x, K):
        # x: (batch, T, x_dim), K: future window length
        # returns z_means, z_stds each of shape (batch, T-K, z_dim)
        batch_size, T, _ = x.shape
        z_means = []
        z_stds  = []
        z_prev  = self.z_0.expand(batch_size, self.z_dim)
        h_0     = self.h_0.expand(-1, batch_size, -1).contiguous()

        for t in range(T - K):
            window = x[:, t:t + K, :]
            h_out, _ = self.rnn(torch.flip(window, dims=[1]), h_0)
            h_right_t = torch.flip(h_out, dims=[1])[:, 0, :]

            z_mean, z_std = self.combiner(z_prev, h_right_t)
            z_means.append(z_mean)
            z_stds.append(z_std)
            z_prev = z_mean

        return torch.stack(z_means, dim=1), torch.stack(z_stds, dim=1)

    def predict_nll(self, x, K, method='transition'):
        # x: (batch, T, x_dim), K: window size (same as used in infer)
        # method: 'direct'     -> z_t -> emitter -> predict x_{t+K}
        #         'transition' -> z_t -> GatedTransition -> z_{t+1} -> emitter -> predict x_{t+K}
        # returns: mean NLL per step (scalar)
        z_means, _ = self.infer(x, K)  # (batch, T-K, z_dim)

        nll_total = 0.0
        n_steps = z_means.shape[1]

        for t in range(n_steps):
            z_t = z_means[:, t, :]

            if method == 'transition':
                z_next_mu, _ = self.transition(z_t)
                df, x_mu, x_sigma = self.emitter(z_next_mu)
            else:  # direct
                df, x_mu, x_sigma = self.emitter(z_t)

            x_actual = x[:, t + K, :]  # first observation outside z_t's window
            nll_total += -dist.StudentT(df, x_mu, x_sigma).log_prob(x_actual).sum()

        return (nll_total / n_steps).item()


def train(dmm, data, num_epochs, lr=1e-3):
    # data: tensor of shape (num_sequences, T, x_dim)
    pyro.clear_param_store()
    optimizer = Adam({"lr": lr})
    svi = SVI(dmm.model, dmm.guide, optimizer, loss=Trace_ELBO())

    for epoch in range(num_epochs):
        loss = svi.step(data)
        if epoch % 100 == 0:
            print(f"epoch {epoch} loss: {loss:.4f}")
