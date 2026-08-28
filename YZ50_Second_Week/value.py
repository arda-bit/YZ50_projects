"""
Task 1 & 3 & 4: a tiny autograd engine (our own micrograd).

A Value wraps a scalar and remembers which other Values produced it
(`_prev`) and through which operation (`_op`). That's the computation
graph. `backward()` walks the graph in reverse topological order and
applies the chain rule at every node.
"""

import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._op = _op
        self._prev = set(_children)
        # filled in by each operation: how to push out.grad back to the
        # children that produced `out`
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # ---- Task 1: + and * ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # ---- Task 4: pow, exp, true division (to build tanh from parts) ----
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad  # d/dx e^x = e^x

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ---- Task 2: tanh as a single fused op (built-in, non-linearity) ----
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    # ---- Task 3: backward() ----
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ---- Task 1 (optional): render the computation graph with graphviz ----
    def draw_graph(self, filename="graph"):
        try:
            from graphviz import Digraph
        except ImportError:
            print("graphviz not installed (`pip install graphviz`), skipping draw")
            return None

        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(self)

        for v in nodes:
            uid = str(id(v))
            dot.node(
                uid,
                label=f"{{ {v.label} | data {v.data:.4f} | grad {v.grad:.4f} }}",
                shape="record",
            )
            if v._op:
                dot.node(uid + v._op, label=v._op)
                dot.edge(uid + v._op, uid)

        for v1, v2 in edges:
            dot.edge(str(id(v1)), str(id(v2)) + v2._op)

        try:
            dot.render(filename, cleanup=True)
        except Exception as exc:
            print(f"graphviz 'dot' executable not available, skipping render ({exc})")
        return dot
