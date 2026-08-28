"""
Task 3: backward() - set output grad to 1, walk nodes in reverse
topological order, apply the chain rule at each node. See value.py's
Value.backward() for the implementation; this script exercises it,
including the "used more than once" case where gradients must be
ACCUMULATED (+=), not overwritten (=) - the famous bug from the video.
"""

from value import Value

# ---------------------------------------------------------------------
# The bug, minimal repro: if a variable is used twice, `=` in _backward
# silently drops one of the two gradient contributions.
# ---------------------------------------------------------------------
a = Value(3.0, label="a")
bsum = a + a   # a used twice
bsum.label = "a+a"
bsum.backward()
print("b = a + a  ->  db/da should be 2.0, got:", a.grad)
assert a.grad == 2.0, "gradient accumulation is broken for repeated use of a variable"

# ---------------------------------------------------------------------
# The example from the video: f = (a*b) * (a+b), with a and b each
# feeding into two different paths of the graph.
# ---------------------------------------------------------------------
a = Value(-2.0, label="a")
b = Value(3.0, label="b")
d = a * b
d.label = "d"
e = a + b
e.label = "e"
f = d * e
f.label = "f"

f.backward()

print("\nf = (a*b) * (a+b), a=-2, b=3")
print("f.data =", f.data)
print("a.grad =", a.grad)  # expected -3.0
print("b.grad =", b.grad)  # expected -8.0

assert f.data == -6.0
assert a.grad == -3.0
assert b.grad == -8.0
print("MATCH: gradients agree with the hand-worked chain rule from the video")
