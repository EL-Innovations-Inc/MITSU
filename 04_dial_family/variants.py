"""MITSU dial variants — every way of filling the mother function's three seats.

    mother:  f(x, y, z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x + y + z)

Seat x always carries the signal as s = softplus(u). What changes is y and z:

    fixed       y = a, z = b            constants
    learn01     y, z = squash(theta)    one learned pair per layer, range (0.05, 1)
    learn02     y, z = squash(theta, 2) one learned pair per layer, range (0.05, 1.95),
                                        starts at exactly a = b = 1 (current MITSU)
    inscalar    y, z = squash(w.x + c)  one pair per example/token, from the
                                        layer's own input (tiny extra layer)
    inunit      y, z = squash(W x + c)  one pair per unit per example/token
                                        (3 pre-activations per unit)
    tri         x, y, z = three different signals (the repo's tri block)

Every dial form is centred by subtracting its resting level f(0+, a, b) =
(a^b + b^a)/(a + b), so silence maps to 0. With a = b = 1 this is exactly
MITSU(u) = s(s-1)/(s+2).

log=True takes log f instead of f (centred by subtracting log of the resting
level), computed with logsumexp so nothing can overflow.

All of it is plain autograd; no custom backward.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

S_FLOOR = 1e-6          # floor on softplus so log(s) and s^p stay finite
ECAP, ETA = 6.0, 0.15   # tri-block constants, frozen in the repo's prereg1
DIAL_FLOOR = 0.05       # learned / input-driven dials live in [floor, cap - floor*(cap-1)]


def squash(raw, cap=1.0):
    """Map an unbounded number to a dial. cap=1: (0.05, 1). cap=2: (0.05, 1.95),
    with raw = 0 giving exactly 1. The floor keeps log(dial) finite when a
    sigmoid underflows to 0 in float32 (caught by the sanity gate)."""
    lo = DIAL_FLOOR
    hi = cap if cap <= 1.0 else cap - DIAL_FLOOR
    return lo + (hi - lo) * torch.sigmoid(raw)


def sp(u):
    return F.softplus(u).clamp_min(S_FLOOR)


def mitsu_ref(u):
    """Reference closed form, a = b = 1."""
    s = F.softplus(u)
    return s * (s - 1) / (s + 2)


def _as_t(v, like):
    return v if torch.is_tensor(v) else torch.as_tensor(v, dtype=like.dtype, device=like.device)


def dial(u, a, b, log=False):
    """Centred mother function with the signal in seat x and dials a, b in y, z.

    a, b: positive floats or tensors broadcastable to u.
    """
    s = sp(u)
    a = _as_t(a, s)
    b = _as_t(b, s)
    ls, la, lb = torch.log(s), torch.log(a), torch.log(b)
    t1 = (a + b) * ls            # log of s^(a+b)
    t2 = (s + b) * la            # log of a^(s+b)
    t3 = (s + a) * lb            # log of b^(s+a)
    rest = (torch.exp(b * la) + torch.exp(a * lb)) / (a + b)
    if log:
        lf = torch.logsumexp(torch.stack(torch.broadcast_tensors(t1, t2, t3), 0), 0) - torch.log(s + a + b)
        return lf - torch.log(rest)
    f = (torch.exp(t1) + torch.exp(t2) + torch.exp(t3)) / (s + a + b)
    return f - rest


def tri(U, log=False):
    """Three-seat competition. U: (..., N, 3) -> (..., N).

    log=False: the repo's tri block, verbatim:
        out = [sum_c MITSU(u_c) + sum_c s_c^e_c - 3] / (S + ETA)
        s_c = 2*sigmoid(u_c),  S = sum s_c,  e_c = min(S - s_c, ECAP)
    log=True: log of the pure competition term (the numerator above can be
        negative, so the MITSU sum and the -3 are dropped):
        out = log[ sum_c s_c^e_c / (S + ETA) ] - log[3 / (3 + ETA)]
    """
    s = (2 * torch.sigmoid(U)).clamp_min(1e-9)
    S = s.sum(-1, keepdim=True)
    e = torch.clamp(S - s, max=ECAP)
    if log:
        lf = torch.logsumexp(e * torch.log(s), -1) - torch.log(S[..., 0] + ETA)
        return lf - math.log(3.0 / (3.0 + ETA))
    pw = s ** e
    m = mitsu_ref(U).sum(-1)
    return (m + pw.sum(-1) - 3.0) / (S[..., 0] + ETA)


# --------------------------------------------------------------------------
# Variant registry. Each spec says how a layer turns its input into `width`
# activations. kind: 'std' (one pre-activation per unit, parameter-free act),
# 'learn' (per-layer learned dials), 'inscalar', 'inunit', 'tri'.
# --------------------------------------------------------------------------
FIXED = [(0.9, 0.9), (0.75, 1.0), (0.5, 1.0), (0.75, 0.75), (0.5, 0.5), (1.0, 1.25), (1.5, 1.5)]


def _fx(a, b):
    return f"fixed({a:g},{b:g})"


def specs():
    out = {}
    out["relu"] = dict(kind="std", fn=F.relu)
    out["gelu"] = dict(kind="std", fn=lambda u: F.gelu(u, approximate="tanh"))
    out["silu"] = dict(kind="std", fn=F.silu)
    for lg in (False, True):
        p = "log-" if lg else ""
        out[p + "mitsu"] = dict(kind="std", fn=(lambda u, lg=lg: dial(u, 1.0, 1.0, lg)))
        for a, b in FIXED:
            out[p + _fx(a, b)] = dict(kind="std", fn=(lambda u, a=a, b=b, lg=lg: dial(u, a, b, lg)))
        out[p + "learn01"] = dict(kind="learn", cap=1.0, init=3.0, log=lg)
        out[p + "learn02"] = dict(kind="learn", cap=2.0, init=0.0, log=lg)
        out[p + "inscalar"] = dict(kind="inscalar", log=lg)
        out[p + "inunit"] = dict(kind="inunit", log=lg, heavy=True)
        out[p + "tri"] = dict(kind="tri", log=lg, heavy=True)
    return out


class Layer(nn.Module):
    """input (.., d_in) -> activations (.., width), for any variant spec.

    Weight init is supplied by the caller through `init_w(out_features,
    in_features) -> tensor` so each benchmark keeps its own convention. The
    main (signal) weight comes from `init_w`; the small dial projection of
    `inscalar` comes from `init_aux`, a separate random stream, so every
    one-pre-activation variant starts from identical signal weights for a
    given seed.
    Dial projections start small (0.1x) with bias +3, i.e. dials near 0.95,
    so every input-driven variant starts close to current MITSU.
    """

    def __init__(self, spec, d_in, width, init_w, bias=True, init_aux=None):
        super().__init__()
        self.spec, self.kind, self.width = spec, spec["kind"], width
        self.record, self.last_ab = False, None
        init_aux = init_aux or init_w
        k = self.kind
        npre = 3 if k in ("inunit", "tri") else 1
        self.fc = nn.Linear(d_in, npre * width, bias=bias)
        with torch.no_grad():
            if npre == 1:
                self.fc.weight.copy_(init_w(width, d_in))
            else:
                w = init_w(3 * width, d_in).view(width, 3, d_in)
                if k == "inunit":
                    w[:, 1:, :] *= 0.1
                self.fc.weight.copy_(w.reshape(3 * width, d_in))
            if bias:
                self.fc.bias.zero_()
        if k == "inunit":
            self.dial_bias = nn.Parameter(torch.full((2,), 3.0))
        if k == "learn":
            self.theta = nn.Parameter(torch.full((2,), float(spec["init"])))
        if k == "inscalar":
            self.dl = nn.Linear(d_in, 2)
            with torch.no_grad():
                self.dl.weight.copy_(init_aux(2, d_in) * 0.1)
                self.dl.bias.fill_(3.0)

    def dials(self):
        """Current learned dials (learn variants only)."""
        return squash(self.theta, self.spec["cap"]).detach().tolist()

    def _rec(self, a, b):
        if self.record:
            self.last_ab = [float(a.mean()), float(a.std()), float(b.mean()), float(b.std())]

    def forward(self, x):
        k = self.kind
        if k == "std":
            return self.spec["fn"](self.fc(x))
        if k == "learn":
            ab = squash(self.theta, self.spec["cap"])
            return dial(self.fc(x), ab[0], ab[1], self.spec["log"])
        if k == "inscalar":
            ab = squash(self.dl(x))
            self._rec(ab[..., 0], ab[..., 1])
            return dial(self.fc(x), ab[..., 0:1], ab[..., 1:2], self.spec["log"])
        U = self.fc(x)
        U = U.view(*U.shape[:-1], self.width, 3)
        if k == "inunit":
            a = squash(U[..., 1] + self.dial_bias[0])
            b = squash(U[..., 2] + self.dial_bias[1])
            self._rec(a, b)
            return dial(U[..., 0], a, b, self.spec["log"])
        return tri(U, self.spec["log"])
