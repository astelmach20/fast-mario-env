# SPDX-License-Identifier: GPL-3.0-or-later
"""Nature-CNN actor-critic, identical in architecture and init to SB3 `CnnPolicy` (shared NatureCNN
features, net_arch=[] -> separate linear pi / vf heads; orthogonal init gains sqrt(2) / 0.01 / 1),
plus a graph-capturable PPO loss identical to SB3 PPO.train() (per-minibatch advantage
normalisation, clipped surrogate, unclipped MSE value loss, -mean(entropy))."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureAC(nn.Module):
    def __init__(self, n_actions=7, in_ch=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Linear(3136, 512)
        self.pi = nn.Linear(512, n_actions)
        self.vf = nn.Linear(512, 1)
        for m in list(self.cnn) + [self.fc]:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, math.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.vf.weight, 1.0); nn.init.zeros_(self.vf.bias)

    def forward(self, obs_u8, channels_last=False):
        """obs uint8 (B,4,84,84) -> logits f32 (B,A), value f32 (B,).
        Convs + fc follow the ambient autocast (bf16 if enabled); the two heads always run in fp32 so
        rollout and update log-probs agree to fp32 precision regardless of batch size."""
        x = obs_u8.float().mul_(1.0 / 255.0)  # SB3: obs / 255 (same rounding)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        h = F.relu(self.fc(self.cnn(x)))
        with torch.autocast("cuda", enabled=False):
            h = h.float()
            return self.pi(h), self.vf(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_u8, channels_last=False, greedy=False):
        """Sample (Gumbel-max == Categorical.sample in distribution; graph-capturable).
        returns action int64, logprob f32, value f32."""
        logits, v = self(obs_u8, channels_last)
        logp_all = F.log_softmax(logits, -1)
        if greedy:
            a = logits.argmax(-1)
        else:
            u = torch.rand_like(logits).clamp_(1e-20, 1.0)
            a = (logp_all - torch.log(-torch.log(u))).argmax(-1)
        return a, logp_all.gather(-1, a[:, None]).squeeze(-1), v


def ppo_loss(net, obs, act, old_logp, adv, ret, old_v, clip, ent_coef, vf_coef, channels_last=False,
             norm_adv=True, clip_vf=None):
    logits, v = net(obs, channels_last)
    logp_all = F.log_softmax(logits, -1)
    logp = logp_all.gather(-1, act[:, None]).squeeze(-1)
    ent = -(logp_all.exp() * logp_all).sum(-1)
    if norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    lr_ = (logp - old_logp).clamp(-20.0, 20.0)  # guard exp overflow (stale/bf16 behaviour log-probs)
    ratio = torch.exp(lr_)
    pg = -torch.min(adv * ratio, adv * ratio.clamp(1 - clip, 1 + clip)).mean()
    if clip_vf is not None:
        v = old_v + (v - old_v).clamp(-clip_vf, clip_vf)
    vl = F.mse_loss(ret, v)
    el = -ent.mean()
    loss = pg + ent_coef * el + vf_coef * vl
    with torch.no_grad():
        stats = torch.stack([pg, vl, -el, ((ratio - 1).abs() > clip).float().mean(),
                             ((torch.exp(lr_) - 1) - lr_).mean()])  # approx_kl as SB3
    return loss, stats


STAT_NAMES = ("policy_gradient_loss", "value_loss", "entropy", "clip_fraction", "approx_kl", "nonfinite_skips")
