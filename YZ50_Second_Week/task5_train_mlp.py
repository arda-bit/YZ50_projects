"""
Task 5: Neuron/Layer/MLP (see nn.py) trained on the small binary
classification dataset from the video. Loss should decrease step by
step. Gradients are zeroed at the start of every step - forgetting this
is "the famous bug" from the video: gradients would otherwise keep
accumulating across steps on top of the previous step's values.
"""

import random
from value import Value
from nn import MLP

random.seed(1337)  # reproducible weight init, like the video

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

n = MLP(3, [4, 4, 1])
print(n)
print("number of parameters:", len(n.parameters()))

learning_rate = 0.05
steps = 100

for k in range(steps):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # backward pass - MUST zero grads first, or they accumulate across
    # steps on top of stale values from the previous step (the bug)
    n.zero_grad()
    loss.backward()

    # gradient descent update
    for p in n.parameters():
        p.data += -learning_rate * p.grad

    if k % 10 == 0 or k == steps - 1:
        print(f"step {k:3d}  loss {loss.data:.6f}")

print("\nfinal predictions (target -> prediction):")
for x, ygt in zip(xs, ys):
    pred = n(x)
    print(f"  {ygt:+.1f} -> {pred.data:+.4f}")
