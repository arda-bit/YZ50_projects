"""
Task 2: fill in gradients BY HAND (no backward() yet) for the two
examples from the video, then check the hand-derived numbers against
backward() to make sure the chain-rule-by-hand was done right.
"""

from value import Value

# ---------------------------------------------------------------------
# Example 1: L = (a*b + c) * f      (32:10 - 51:10)
# ---------------------------------------------------------------------
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b
e.label = "e"
d = e + c
d.label = "d"
f = Value(-2.0, label="f")
L = d * f
L.label = "L"

print("=== Example 1: L = (a*b + c) * f ===")
print("L.data =", L.data)

# By hand, working backwards from L:
#   dL/dL = 1
#   L = d * f  ->  dL/dd = f.data,  dL/df = d.data
#   d = e + c  ->  local derivative of + is 1, so gradient just passes
#                  through: dL/de = dL/dd * 1,  dL/dc = dL/dd * 1
#   e = a * b  ->  dL/da = dL/de * b.data,  dL/db = dL/de * a.data
L.grad = 1.0
d.grad = f.data
f.grad = d.data
e.grad = d.grad * 1.0
c.grad = d.grad * 1.0
a.grad = e.grad * b.data
b.grad = e.grad * a.data

manual_1 = {
    "L": L.grad, "d": d.grad, "f": f.grad,
    "e": e.grad, "c": c.grad, "a": a.grad, "b": b.grad,
}
print("manual grads:", manual_1)

# reset and let backward() compute the same thing, to check our hand math
for v in (a, b, c, e, d, f, L):
    v.grad = 0.0
L.backward()
auto_1 = {
    "L": L.grad, "d": d.grad, "f": f.grad,
    "e": e.grad, "c": c.grad, "a": a.grad, "b": b.grad,
}
print("backward():", auto_1)
assert manual_1 == auto_1, "hand-derived gradients don't match backward()!"
print("MATCH\n")


# ---------------------------------------------------------------------
# Example 2: a single neuron  o = tanh(x1*w1 + x2*w2 + b)   (52:52 - 1:09:02)
# ---------------------------------------------------------------------
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")

x1w1 = x1 * w1
x1w1.label = "x1*w1"
x2w2 = x2 * w2
x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b
n.label = "n"
o = n.tanh()
o.label = "o"

print("=== Example 2: single neuron o = tanh(x1*w1 + x2*w2 + b) ===")
print("o.data =", o.data)

# By hand:
#   do/do = 1
#   o = tanh(n)  ->  d(tanh)/dn = 1 - tanh(n)^2 = 1 - o.data**2
#   n = x1w1x2w2 + b  -> local deriv of + is 1, passes through unchanged
#   x1w1x2w2 = x1w1 + x2w2 -> same, passes through unchanged
#   x1w1 = x1 * w1  ->  dL/dx1 = dL/dx1w1 * w1.data, dL/dw1 = dL/dx1w1 * x1.data
#   x2w2 = x2 * w2  ->  dL/dx2 = dL/dx2w2 * w2.data, dL/dw2 = dL/dx2w2 * x2.data
o.grad = 1.0
n.grad = (1 - o.data**2) * o.grad
x1w1x2w2.grad = n.grad * 1.0
b.grad = n.grad * 1.0
x1w1.grad = x1w1x2w2.grad * 1.0
x2w2.grad = x1w1x2w2.grad * 1.0
x1.grad = x1w1.grad * w1.data
w1.grad = x1w1.grad * x1.data
x2.grad = x2w2.grad * w2.data
w2.grad = x2w2.grad * x2.data

manual_2 = {
    "x1": x1.grad, "w1": w1.grad, "x2": x2.grad, "w2": w2.grad, "b": b.grad,
}
print("manual grads:", manual_2)

for v in (x1, x2, w1, w2, b, x1w1, x2w2, x1w1x2w2, n, o):
    v.grad = 0.0
o.backward()
auto_2 = {
    "x1": x1.grad, "w1": w1.grad, "x2": x2.grad, "w2": w2.grad, "b": b.grad,
}
print("backward():", auto_2)
assert manual_2 == auto_2, "hand-derived neuron gradients don't match backward()!"
print("MATCH")
