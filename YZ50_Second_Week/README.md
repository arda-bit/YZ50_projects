# YZ50 - Week 2: Backpropagation / micrograd

A small, from-scratch autograd engine (`value.py`) plus a neural net
built on top of it (`nn.py`), following Karpathy's
["The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0).

## Files

- `value.py` - the `Value` class: `+`, `*`, `**`, `exp`, `/`, `tanh`, the
  computation graph (`_prev`, `_op`), `backward()`, and an optional
  graphviz visualizer.
- `nn.py` - `Neuron`, `Layer`, `MLP` built on top of `Value`.
- `task1_value_and_graph.py` - builds a small expression, inspects the
  graph it produced, optionally renders it with graphviz.
- `task2_manual_gradients.py` - fills in gradients by hand (no
  `backward()`) for the simple expression and the single-neuron
  examples from the video, then checks the hand math against
  `backward()`.
- `task3_backward.py` - exercises `backward()`, including the classic
  "variable used twice" bug (gradients must accumulate with `+=`, not
  overwrite with `=`).
- `task4_verify_tanh.py` - decomposes `tanh` into `exp`/`/`/`**` and
  checks it gives the same gradients as the fused `tanh()` op, then
  verifies the neuron's gradients three ways: our `backward()`,
  numerical (central-difference) derivative, and PyTorch autograd.
- `task5_train_mlp.py` - builds an `MLP(3, [4, 4, 1])` and trains it on
  the video's 4-example toy dataset, zeroing gradients every step.

## Running

```bash
pip install -r requirements.txt   # torch + graphviz (graphviz's `dot`
                                   # binary is optional - only needed to
                                   # render the graph image in task 1)
python task1_value_and_graph.py
python task2_manual_gradients.py
python task3_backward.py
python task4_verify_tanh.py
python task5_train_mlp.py
```

All five scripts print their own results and assert internally that
the numbers match what they should (manual grads vs `backward()`,
decomposed tanh vs fused tanh, analytical vs numerical vs PyTorch,
loss decreasing during training).

## Results

- Task 2 & 3: hand-derived gradients match `backward()` exactly on both
  the `(a*b + c) * f` expression and the single neuron.
- Task 4: `backward()`, the central-difference numerical derivative,
  and PyTorch's autograd all agree on the neuron's gradients to ~1e-8.
- Task 5: loss drops from ~3.09 at step 0 to ~0.000000 by step 90 on
  the 4-example dataset; final predictions match targets almost
  exactly (e.g. target `+1.0` -> prediction `+1.0000`).
