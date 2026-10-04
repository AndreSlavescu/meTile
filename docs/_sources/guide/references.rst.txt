Research references
===================

meTile borrows ideas from several compiler and GPU-programming projects. These
references explain the ideas, but they aren't evidence that meTile has the same
feature coverage or performance.

Tiled programming and layouts
-----------------------------

`Triton <https://doi.org/10.1145/3315508.3329973>`_ introduced a compiler and
intermediate language for tiled neural-network kernels. Its programming model
informs meTile's Python kernel language. `CuTe's layout algebra
<https://arxiv.org/abs/2603.02298>`_ informs the separation of logical coordinates
from memory and execution layouts.

For distributed ownership, see Triton's `Gluon layout tutorial
<https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html>`_
and the `Linear Layouts paper <https://arxiv.org/abs/2505.23819>`_. meTile's
``ThreadLayout`` is a narrower register/thread-bit permutation model, not an
implementation of every Gluon or CuTe layout.

Selecting compiler rewrites
---------------------------

`Lap Chi Lau's CS 341 notes
<https://cs.uwaterloo.ca/~lapchi/cs341-2025/notes.html>`_ cover maximum flow,
minimum cut and project selection. The project-selection reduction helped
shape meTile's rewrite selector. `Horace He's AOTAutograd discussion
<https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467>`_
shows min-cut applied to a different compiler problem: choosing activations to
save or recompute.

The project author's CS 341 course with Lap Chi Lau in Spring 2025 helped
inspire this direction, particularly the treatment of project selection.

In meTile, overlapping rewrite candidates form a conflict graph. An s-t min-cut
selects a maximum-weight independent set when a conflict component is
bipartite. Non-bipartite components use a deterministic greedy fallback.
The implementation does **not** solve arbitrary maximum-weight independent
set problems with min-cut. See :doc:`graph-fusion` and :doc:`architecture`.

Metal and compiler boundaries
-----------------------------

The `Metal Shading Language specification
<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_
defines shader types, address spaces and synchronization rules. Apple's
`Metal Performance Primitives guide
<https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf>`_
describes the native tensor operations used by eligible backends.

The :doc:`compiler-bypasses` guide separates supported Metal compilation from
AIR and native-ISA research. Emitting a vector expression or changing AIR does
not guarantee a particular final GPU instruction sequence.

Bibliography
------------

.. code-block:: bibtex

   @misc{he2022mincut,
       title={Min-cut optimal(*) recomputation (i.e. activation checkpointing) with AOTAutograd},
       author={Horace He},
       year={2022},
       howpublished={PyTorch Dev Discussions},
       url={https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467}
   }

   @misc{lau2025cs341,
       title={CS 341: Algorithms, Lectures 15 and 16: Maximum Flow, Minimum Cut, and Applications},
       author={Lap Chi Lau},
       year={2025},
       howpublished={University of Waterloo course notes},
       url={https://cs.uwaterloo.ca/~lapchi/cs341-2025/notes.html}
   }

   @misc{cecka2026cute,
       title={CuTe Layout Representation and Algebra},
       author={Cris Cecka},
       year={2026},
       eprint={2603.02298},
       archivePrefix={arXiv},
       primaryClass={cs.MS},
       url={https://arxiv.org/abs/2603.02298}
   }

   @inproceedings{tillet2019triton,
       title={Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations},
       author={Philippe Tillet and H. T. Kung and David Cox},
       booktitle={Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
       year={2019},
       doi={10.1145/3315508.3329973}
   }
