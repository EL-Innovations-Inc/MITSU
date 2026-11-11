# Stability: what the scalar slope does and does not establish

Version 1.0.1 clarifies the scope of the original S2 hypothesis. There is no
established large-model stability advantage in this repository.

## The unscaled activation

For s = softplus(u),

    m(u) = s(s - 1)/(s + 2)
    m'(u) = [1 - 6/(s + 2)^2] sigmoid(u).

Since s is positive and sigmoid(u) is between 0 and 1, the exact real-valued
scalar derivative lies between -0.5 and 1. This loose analytic bound says
nothing by itself about the norm of a full network's parameter gradients.
Floating-point implementations need separate numerical checks.

## The gain changes that bound

For a fixed positive gain g, the gate is a(u) = g m(u) and

    a'(u) = g m'(u),       -g/2 < a'(u) < g.

The pilot's g = 3.0374 therefore has an upper slope limit of 3.0374, rather
than 1. Its lower loose bound is -1.5187. The value 3.0374 estimates the ratio
std(SiLU(Z))/std(MITSU(Z)) for standard-normal Z. It matches that one standard
deviation, not the mean, second moment, derivative distribution, or the
actual pre-activation distribution at every layer and training step.
A gain is a controlled experimental choice, not a proof of stability.

## A gated feed-forward block has two gradient paths

For column-vector x and h(x) = W2 [a(W1 x) elementwise-multiplied by W3 x],

    J_h(x) = W2 [diag(W3 x) diag(a'(W1 x)) W1
                 + diag(a(W1 x)) W3].

Both weight norms and activation magnitudes matter. The linear up-projection
W3 x can grow with x; the scalar slope bound does not bound this Jacobian
uniformly over all x and weights. Residual paths, normalization, attention,
optimizer state, learning rate, precision and data also affect training.

## Numerical implementation in this release

The canonical NumPy and PyTorch activation avoid squaring a large softplus
value. Low-precision PyTorch inputs are evaluated internally in float32 and
cast back to their input dtype. This prevents the demonstrated FP16
intermediate overflow at u = 257. It does not promise that an arbitrary
network, scaled gate, or output outside the dtype's representable range
will remain finite. See the numerical tests and benchmark results.

## How to test S2

Compare matched models, fixed gain, identical initial weights and data order,
multiple paired seeds, and equal token budgets. Freeze loss-spike and
non-finite-gradient definitions before collecting training results. Report
both the event count and exposure (steps/tokens), convergence/loss, and
throughput. Zero observed incidents over a small exposure is insufficient
evidence that incident rates differ. The new experiment 05 includes these
diagnostics at a short CPU horizon; the larger GPU study remains future work.

The historical PLAN and RING2_PLAYBOOK describe proposed studies. Their
prices, timing estimates and go/no-go language are not measured outcomes
or instructions to spend money.
