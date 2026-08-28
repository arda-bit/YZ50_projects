"""
Task 1: Value class with + and *, each Value remembering the Values (and
op) that produced it. Build a small expression and inspect / draw the
resulting computation graph.
"""

from value import Value

# a*b + c, same expression Karpathy builds in the video
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

print("d.data  =", d.data)                     # 2*-3 + 10 = 4
print("d._op   =", d._op)                       # '+'
print("d._prev =", {v.label for v in d._prev})  # {'e', 'c'}
print("e._prev =", {v.label for v in e._prev})  # {'a', 'b'}

# render the graph if graphviz is available (pip install graphviz + the
# `dot` binary); harmless no-op otherwise
d.draw_graph(filename="task1_graph")
