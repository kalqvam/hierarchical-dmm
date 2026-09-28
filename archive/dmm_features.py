import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


class MixedEmitter(nn.Module):
    # z_t -> emission parameters for three distribution families
    # x layout: first n_studentt cols  = Student-t features (log returns)
    #           next  n_lognormal cols  = log-normal features (rv, vix)
    #           remaining n_normal cols = Gaussian features (yield curve)
    def __init__(self, z_dim, hidden_dim, n_studentt, n_lognormal, n_normal):
        super().__init__()
        self.n_studentt  = n_studentt
        self.n_lognormal = n_lognormal
        self.n_normal    = n_normal
        self.shared_st = nn.Linear(z_dim, hidden_dim)   # for Student-t
        self.shared_ln = nn.Linear(z_dim, hidden_dim)   # for LogNormal  
        self.shared_n  = nn.Linear(z_dim, hidden_dim)   # for Gaussian
        # Student-t heads
        self.st_mu    = nn.Linear(hidden_dim, n_studentt)
        self.st_sigma = nn.Linear(hidden_dim, n_studentt)
        self.log_df   = nn.Parameter(torch.ones(n_studentt) * 1.5)
        # log-normal heads
        self.ln_mu    = nn.Linear(hidden_dim, n_lognormal)
        self.ln_sigma = nn.Linear(hidden_dim, n_lognormal)
        # Gaussian heads
        self.n_mu     = nn.Linear(hidden_dim, n_normal)
        self.n_sigma  = nn.Linear(hidden_dim, n_normal)
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        h_st = torch.tanh(self.shared_st(z_t))
        h_ln = torch.tanh(self.shared_ln(z_t))
        h_n  = torch.tanh(self.shared_n(z_t))
        
        df       = 2.0 + self.softplus(self.log_df)
        st_mu    = self.st_mu(h_st)
        st_sigma = self.softplus(self.st_sigma(h_st))
        ln_mu    = self.ln_mu(h_ln)
        ln_sigma = self.softplus(self.ln_sigma(h_ln))
        n_mu     = self.n_mu(h_n)
        n_sigma  = self.softplus(self.n_sigma(h_n))
        return df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma


class GatedTransition(nn.Module):
    # maps z_{t-1} -> parameters of p(z_t | z_{t-1})
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.lin_z_gate               = nn.Linear(z_dim, hidden_dim)
        self.lin_z_gate_reverse       = nn.Linear(hidden_dim, z_dim)
        self.lin_z_transition         = nn.Linear(z_dim, hidden_dim)
        self.lin_z_transition_reverse = nn.Linear(hidden_dim, z_dim)
        self.lin_z_mu                 = nn.Linear(z_dim, z_dim)
        nn.init.eye_(self.lin_z_mu.weight)
        nn.init.zeros_(self.lin_z_mu.bias)
        self.lin_z_sigma = nn.Linear(z_dim, z_dim)
        self.act      = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid  = nn.Sigmoid()

    def forward(self, z_prev):
        # input: z_prev (batch, z_dim)
        # output: mean, std (batch, z_dim) each
        g_t = self.act(self.lin_z_gate(z_prev))
        g_t = self.sigmoid(self.lin_z_gate_reverse(g_t))
        h_t = self.act(self.lin_z_transition(z_prev))
        h_t = self.lin_z_transition_reverse(h_t)
        mu_t    = (1 - g_t) * self.lin_z_mu(z_prev) + g_t * h_t
        sigma_t = self.softplus(self.lin_z_sigma(self.act(h_t)))
        return mu_t, sigma_t


class Combiner(nn.Module):
    # maps (z_{t-1}, h_right_t) -> parameters of q(z_t | z_{t-1}, x_{1:T})
    def __init__(self, z_dim, rnn_hidden_dim):
        super().__init__()
        self.lin_mu    = nn.Linear(rnn_hidden_dim, z_dim)
        self.lin_sigma = nn.Linear(rnn_hidden_dim, z_dim)
        self.lin_z     = nn.Linear(z_dim, rnn_hidden_dim)
        self.act      = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_prev, h_right):
        # inputs: z_prev (batch, z_dim), h_right (batch, rnn_hidden_dim)
        # output: mean, std (batch, z_dim) each
        h_combined = 0.5 * (self.act(self.lin_z(z_prev)) + h_right)
        mu_t    = self.lin_mu(h_combined)
        sigma_t = self.softplus(self.lin_sigma(h_combined))
        return mu_t, sigma_t


class DMMFeatures(nn.Module):
    def __init__(self, n_studentt, n_lognormal, n_normal, z_dim, hidden_dim, rnn_hidden_dim, rnn_num_layers):
        super().__init__()
        x_dim = n_studentt + n_lognormal + n_normal
        self.n_studentt  = n_studentt
        self.n_lognormal = n_lognormal
        self.n_normal    = n_normal
        self.z_dim       = z_dim

        self.emitter    = MixedEmitter(z_dim, hidden_dim, n_studentt, n_lognormal, n_normal)
        self.transition = GatedTransition(z_dim, hidden_dim)
        self.combiner   = Combiner(z_dim, rnn_hidden_dim)
        self.rnn        = nn.GRU(x_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True)

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(rnn_num_layers, 1, rnn_hidden_dim))

    def _split(self, x):
        a = self.n_studentt
        b = self.n_studentt + self.n_lognormal
        return x[..., :a], x[..., a:b], x[..., b:]

    def model(self, x):
        # x: (batch, T, n_studentt + n_lognormal + n_normal)
        pyro.module("dmm_features", self)
        batch_size, T, _ = x.shape
        x_st, x_ln, x_n = self._split(x)

        z_prev = self.z_0.expand(batch_size, self.z_dim)
        with pyro.plate("data", batch_size):
            for t in range(T):
                z_mean, z_std = self.transition(z_prev)
                z_t = pyro.sample(f"z_{t}", dist.Normal(z_mean, z_std).to_event(1))

                df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma = self.emitter(z_t)
                if self.n_studentt > 0:
                    pyro.sample(f"x_st_{t}", dist.StudentT(df, st_mu, st_sigma).to_event(1),
                                obs=x_st[:, t, :])
                if self.n_lognormal > 0:
                    pyro.sample(f"x_ln_{t}", dist.LogNormal(ln_mu, ln_sigma).to_event(1),
                                obs=x_ln[:, t, :])
                if self.n_normal > 0:
                    pyro.sample(f"x_n_{t}", dist.Normal(n_mu, n_sigma).to_event(1),
                                obs=x_n[:, t, :])
                z_prev = z_t

    def guide(self, x):
        # structured variational posterior q(z_{1:T} | x_{1:T}) via DKS
        pyro.module("dmm_features", self)
        batch_size, T, _ = x.shape
        h_0 = self.h_0.expand(-1, batch_size, -1).contiguous()
        h_right, _ = self.rnn(torch.flip(x, dims=[1]), h_0)
        h_right = torch.flip(h_right, dims=[1])  # (batch, T, rnn_hidden_dim)

        z_prev = self.z_0.expand(batch_size, self.z_dim)
        with pyro.plate("data", batch_size):
            for t in range(T):
                z_mean, z_std = self.combiner(z_prev, h_right[:, t, :])
                z_t = pyro.sample(f"z_{t}", dist.Normal(z_mean, z_std).to_event(1))
                z_prev = z_t

    def infer(self, x, K):
        # x: (batch, T, x_dim), K: window length
        # returns z_means, z_stds each (batch, T-K, z_dim)
        batch_size, T, _ = x.shape
        z_means, z_stds = [], []
        z_prev = self.z_0.expand(batch_size, self.z_dim)
        h_0    = self.h_0.expand(-1, batch_size, -1).contiguous()

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
        # returns: mean NLL per step (scalar), summed across all features
        z_means, _ = self.infer(x, K)
        x_st, x_ln, x_n = self._split(x)

        nll_total = 0.0
        n_steps   = z_means.shape[1]

        for t in range(n_steps):
            z_t = z_means[:, t, :]
            if method == 'transition':
                z_next_mu, _ = self.transition(z_t)
                df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma = self.emitter(z_next_mu)
            else:
                df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma = self.emitter(z_t)

            if self.n_studentt > 0:
                nll_total += -dist.StudentT(df, st_mu, st_sigma).log_prob(x_st[:, t + K, :]).sum()
            if self.n_lognormal > 0:
                nll_total += -dist.LogNormal(ln_mu, ln_sigma).log_prob(x_ln[:, t + K, :]).sum()
            if self.n_normal > 0:
                nll_total += -dist.Normal(n_mu, n_sigma).log_prob(x_n[:, t + K, :]).sum()

        return (nll_total / n_steps).item()

    def predict_nll_breakdown(self, x, K, method='transition'):
        # same as predict_nll but returns per-feature NLLs as a dict
        z_means, _ = self.infer(x, K)
        x_st, x_ln, x_n = self._split(x)

        nll_st = nll_ln = nll_n = 0.0
        n_steps = z_means.shape[1]

        for t in range(n_steps):
            z_t = z_means[:, t, :]
            if method == 'transition':
                z_next_mu, _ = self.transition(z_t)
                df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma = self.emitter(z_next_mu)
            else:
                df, st_mu, st_sigma, ln_mu, ln_sigma, n_mu, n_sigma = self.emitter(z_t)

            if self.n_studentt > 0:
                nll_st += -dist.StudentT(df, st_mu, st_sigma).log_prob(x_st[:, t + K, :]).sum().item()
            if self.n_lognormal > 0:
                nll_ln += -dist.LogNormal(ln_mu, ln_sigma).log_prob(x_ln[:, t + K, :]).sum().item()
            if self.n_normal > 0:
                nll_n  += -dist.Normal(n_mu, n_sigma).log_prob(x_n[:, t + K, :]).sum().item()

        return {
            'studentt':  nll_st / n_steps,
            'lognormal': nll_ln / n_steps,
            'normal':    nll_n  / n_steps,
            'total':    (nll_st + nll_ln + nll_n) / n_steps,
        }
