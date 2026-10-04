<p align="center">
  <img src="docs/_static/metile-logo.png" width="160" alt="meTile tiled M logo">
</p>

<h1 align="center">meTile</h1>

<p align="center">GPU kernels in Python. Compiled for Apple silicon.</p>

Write GPU kernels in Python; meTile compiles them to Metal. Declare tensors,
then choose explicit layouts and schedules where needed. MLX integration is
optional.

**[Read the docs](https://andreslavescu.github.io/meTile/docs/)** ·
[Get started](docs/getting-started/install.rst) ·
[First kernel](docs/getting-started/first-kernel.rst) ·
[Benchmarks](docs/guide/benchmarks.rst)

From a checkout, on an Apple silicon Mac with Python 3.10+:

```sh
python -m pip install -e .
```

For ready-made kernels, install the separate [kernels](kernels/) project with
`python -m pip install -e ./kernels` and import `metile_kernels`. The compiler
does not require it.

meTile is experimental. The docs include measured results, limitations and
commands to reproduce them; performance depends on your workload.

## Citations

Ideas and inspiration: [CuTe](https://arxiv.org/abs/2603.02298),
[Triton](https://doi.org/10.1145/3315508.3329973),
[Horace He's min-cut discussion](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467),
and [Lap Chi Lau's CS 341 notes](https://cs.uwaterloo.ca/~lapchi/cs341-2025/notes.html).
[Full references, acknowledgments, and BibTeX](docs/guide/references.rst).

[Contributing](.github/CONTRIBUTING.md) · [MIT license](LICENSE)
