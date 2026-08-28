"""
Task 4: break tanh into exp / division / pow, show it produces the same
gradients as the fused tanh() op. Then triple-check the neuron's
gradients three independent ways:
  1) our Value.backward() (analytical, via chain rule)
  2) numerical derivative (central difference), same method as week 1
  3) PyTorch autograd
All three should agree.
"""

import math
from value import Value

X1, X2, W1, W2, B = 2.0, 0.0, -3.0, 1.0, 6.8813735870195432


def make_neuron(x1, x2, w1, w2, b, use_fused_tanh):
    """Builds o = tanh(x1*w1 + x2*w2 + b) as Values, either with the
    fused .tanh() op or by hand from exp/div/pow, and returns the leaves
    plus the output so callers can call .backward() and read .grad."""
    x1, x2 = Value(x1, label="x1"), Value(x2, label="x2")
    w1, w2 = Value(w1, label="w1"), Value(w2, label="w2")
    b = Value(b, label="b")
    n = x1 * w1 + x2 * w2 + b
    n.label = "n"

    if use_fused_tanh:
        o = n.tanh()
    else:
        e = (n * 2).exp()  # e^(2n)
        o = (e - 1) / (e + 1)  # (e^2n - 1)/(e^2n + 1) == tanh(n)
    o.label = "o"
    return x1, x2, w1, w2, b, o


# ---------------------------------------------------------------------
# 1) fused tanh() vs decomposed exp/div/pow - same gradients
# ---------------------------------------------------------------------
leaves_fused = make_neuron(X1, X2, W1, W2, B, use_fused_tanh=True)
*params_fused, o_fused = leaves_fused
o_fused.backward()
grads_fused = {p.label: p.grad for p in params_fused}

leaves_decomp = make_neuron(X1, X2, W1, W2, B, use_fused_tanh=False)
*params_decomp, o_decomp = leaves_decomp
o_decomp.backward()
grads_decomp = {p.label: p.grad for p in params_decomp}

print("=== fused tanh() vs decomposed exp/div/pow ===")
print("o (fused)     =", o_fused.data)
print("o (decomposed)=", o_decomp.data)
print("grads (fused)     =", grads_fused)
print("grads (decomposed)=", grads_decomp)
for k in grads_fused:
    assert math.isclose(grads_fused[k], grads_decomp[k], rel_tol=1e-9, abs_tol=1e-12), k
print("MATCH: decomposing tanh gives the same gradients\n")


# ---------------------------------------------------------------------
# 2) analytical (backward()) vs numerical derivative (central difference)
#    same technique as week 1's numerical gradient checking
# ---------------------------------------------------------------------
def neuron_out(x1, x2, w1, w2, b):
    n = x1 * w1 + x2 * w2 + b
    return math.tanh(n)


def numerical_grad(f, args, i, h=1e-6):
    args_plus = list(args)
    args_minus = list(args)
    args_plus[i] += h
    args_minus[i] -= h
    return (f(*args_plus) - f(*args_minus)) / (2 * h)


args = [X1, X2, W1, W2, B]
numerical_grads = [numerical_grad(neuron_out, args, i) for i in range(len(args))]
analytical_grads = [grads_fused["x1"], grads_fused["x2"], grads_fused["w1"], grads_fused["w2"], grads_fused["b"]]

print("=== analytical (backward()) vs numerical (central difference) ===")
names = ["x1", "x2", "w1", "w2", "b"]
for name, a_grad, n_grad in zip(names, analytical_grads, numerical_grads):
    print(f"{name}: analytical={a_grad:.8f}  numerical={n_grad:.8f}")
    assert math.isclose(a_grad, n_grad, rel_tol=1e-4, abs_tol=1e-4), name
print("MATCH: analytical gradients agree with numerical derivative\n")


# ---------------------------------------------------------------------
# 3) analytical (backward()) vs PyTorch autograd
# ---------------------------------------------------------------------
try:
    import torch

    x1t = torch.tensor(X1, requires_grad=True, dtype=torch.double)
    x2t = torch.tensor(X2, requires_grad=True, dtype=torch.double)
    w1t = torch.tensor(W1, requires_grad=True, dtype=torch.double)
    w2t = torch.tensor(W2, requires_grad=True, dtype=torch.double)
    bt = torch.tensor(B, requires_grad=True, dtype=torch.double)

    nt = x1t * w1t + x2t * w2t + bt
    ot = torch.tanh(nt)
    ot.backward()

    print("=== analytical (backward()) vs PyTorch autograd ===")
    print("o (ours) =", o_fused.data, " o (torch) =", ot.item())
    torch_grads = {"x1": x1t.grad.item(), "x2": x2t.grad.item(), "w1": w1t.grad.item(),
                   "w2": w2t.grad.item(), "b": bt.grad.item()}
    for name in names:
        print(f"{name}: ours={grads_fused[name]:.8f}  torch={torch_grads[name]:.8f}")
        assert math.isclose(grads_fused[name], torch_grads[name], rel_tol=1e-9, abs_tol=1e-9), name
    print("MATCH: our backward() agrees with PyTorch autograd")
except ImportError:
    print("PyTorch not installed - skipping the torch comparison "
          "(`pip install torch` to run it). See task4_verify_tanh.py.")
