<p align="center">
  <img src="docs/_static/metile-logo.png" width="160" alt="meTile tiled M logo">
</p>

<h1 align="center">meTile</h1>

<p align="center">GPU kernels in Python. Compiled for Apple silicon.</p>

Define your tensors and write the computation. meTile generates Metal code,
with explicit layouts and schedules when you need more control. MLX integration
is optional.

**[Read the docs](https://andreslavescu.github.io/meTile/docs/)** ·
[Get started](docs/getting-started/install.rst) ·
[Examples](docs/getting-started/first-kernel.rst) ·
[Benchmarks](docs/guide/benchmarks.rst)

From a checkout, on an Apple silicon Mac with Python 3.10+:

```sh
python -m pip install -e .
```

Ready-made kernels live in the separately installable [kernels](kernels/)
project. Add them with `python -m pip install -e ./kernels` and import
`metile_kernels`. The compiler itself does not depend on that package.

Experimental compiler. Performance depends on the workload; the docs include
results, limitations, and reproduction commands.

## Citations

Ideas and inspiration: [CuTe](https://arxiv.org/abs/2603.02298),
[Triton](https://doi.org/10.1145/3315508.3329973),
[Horace He's min-cut discussion](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467),
and [Lap Chi Lau's CS 341 notes](https://cs.uwaterloo.ca/~lapchi/cs341-2025/notes.html).
[Full references, acknowledgments, and BibTeX](docs/guide/references.rst).

[Contributing](.github/CONTRIBUTING.md) · [MIT license](LICENSE)
