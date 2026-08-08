

# meTile

Escribe kernels de GPU en Python, obtén Metal. Se ejecuta en chipsets Apple Silicon y se integra con MLX.

```python
@metile.kernel
def softmax(X, Out, N, BLOCK: metile.constexpr):
    row = metile.program_id(0)

    m = -1e38
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        x = metile.load(X + row * N + cols, mask=cols < N)
        m = metile.maximum(m, x)
    m = metile.max(m)

    s = 0.0
    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        x = metile.load(X + row * N + cols, mask=cols < N)
        s = s + metile.exp(x - m)
    s = metile.sum(s)

    for i in metile.tile_range(0, N, BLOCK):
        cols = i + metile.arange(0, BLOCK)
        x = metile.load(X + row * N + cols, mask=cols < N)
        metile.store(Out + row * N + cols, metile.exp(x - m) / s, mask=cols < N)
```

Escribes los tres pasos obvios. El compilador detecta que los dos primeros pueden fusionarse en uno y los reescribe. Esto lee la entrada dos veces en lugar de tres y se ejecuta 1.28x más rápido, pero solo después de verificar que la fusión es algebraicamente válida.

## Aceleración de un modelo MLX-LM

Una sola llamada. Parchea lo que reconoce, verifica el resultado contra el modelo sin parchear y te informa de lo que hizo.

```python
import metile
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
print(metile.compile(model))
```

```
meTile on qwen2
  accelerating: attention, rms_norm, graph_fusion, quantized_mlp
  surfaces replaced: mlp, input_layernorm, post_attention_layernorm, block
  verification: logits match MLX exactly
```

Las arquitecturas se emparejan por estructura y no por una lista de nombres, por lo que un modelo con el MLP gated habitual es candidato independientemente de si alguien lo ha visto antes. Sin embargo, la estructura es solo una prueba de candidatura: una clase puede tener `gate_proj`, `up_proj` y `down_proj` y aún así combinarlos de manera diferente. Por eso, `compile` ejecuta el modelo antes y después, compara los logits y conserva solo lo que reproduce MLX. Si el conjunto completo no coincide, bisecta y conserva las partes que pasan:

```
meTile on llama
  accelerating: attention, rms_norm, graph_fusion
  verification: logits match MLX exactly
  declined quantized_mlp: changed the logits by 0.0352, 2.3e-03 relative
                          -- reduction-order scale, raise tolerance to keep it
```

En ese caso, la diferencia es del orden de la suma, donde meTile es el lado *más* preciso, por lo que `metile.compile(model, tolerance=5e-3)` lo conserva. El valor predeterminado es exacto porque el error que vale la pena detectar (un emparejamiento estructural cuya aritmética difiere) se sitúa muy por fuera del redondeo.

El informe se evalúa como falso cuando no se reemplaza nada, por lo que `if not metile.compile(model)` es una verificación real. Esto importa más de lo que suena: el resultado peligroso no es un fallo (crash), sino una no-operación silenciosa, y este proyecto emitió una para tres familias de modelos antes de que alguien leyera la lista de omitidos.

Llama a `.restore()` en el informe para restaurar las implementaciones propias de MLX-LM.

## Velocidad

Todo lo siguiente se ejecuta en un **Apple M5 (32 GB, MLX 0.32.0)** y compara meTile contra MLX **utilizando los mismos pesos en el mismo formato en ambos lados**, por lo que una aceleración implica un kernel más rápido, no un cambio en la precisión numérica. `1.00x` significa la misma velocidad que MLX.

### Los lotes pequeños es donde meTile gana

MLX se ralentiza bruscamente una vez que le alimentas más de una fila a la vez. meTile no. Este es el rango en el que se ejecuta una pasada de verificación de decodificación especulativa.

| filas por envío | 1 | 2 | 4 | 8 | 16 | 32 | 128 |
|---|---|---|---|---|---|---|---|
| **BF16** | 1.02x | **1.69x** | **1.82x** | **1.65x** | **1.52x** | 1.06x | 1.11x |
| **INT4** | 1.02x | 1.02x | 1.02x | **1.29x** | **1.31x** | **1.23x** | 1.00x |
| INT8 | 0.98x | 0.99x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

Los resultados de BF16 son **idénticos a nivel de bits** a ejecutar esas filas a través de MLX una por una, por lo que el agrupamiento por lotes cambia la velocidad y nada más.

INT8 se mantiene a la par porque meTile no tiene un kernel propio allí y llama al de MLX, por lo que esa fila describe ambos backends.

![Speedup by batch size](docs/_static/mlx-matched-speedup.png)

Por qué ocurre: cada fila de un lote lee los mismos pesos, por lo que alimentar más filas no debería costar más tráfico de pesos. MLX los vuelve a leer por tile de fila y meTile no, esa es la brecha entre cada par de líneas a continuación. Ambos lados aún muestran una pendiente descendente después de ocho filas, pero esa parte no es desperdicio: los mismos pesos están sirviendo de ocho a treinta y dos veces más aritmética para entonces.

![Weight bandwidth by batch size](docs/_static/mlx-batch-efficiency.png)

La decodificación de una sola fila es otra historia y no tiene margen de mejora alguno. MLX la ejecuta al 93-97% de lo que una lectura en secuencia (streaming read) puede mover, y un kernel escrito a mano la iguala sin superarla, por lo que el 1.02x anterior es todo lo que queda por obtener.

### Modelos completos

| Modelo | Decodificación | Prefill |
|---|---|---|
| Llama 3.2 1B 4-bit | 1.00x | **1.34x** |
| Qwen 2.5 0.5B 4-bit | 0.99x | **1.27x** |
| Qwen 2.5 1.5B 4-bit | 1.00x | **1.33x** |
| Qwen 2.5 1.5B BF16 | 1.00x | 1.06x |
| Qwen 3.5 4B 4-bit | 1.00x | 1.00x |
| Qwen 3.5 9B 4-bit | 1.00x | 1.00x |

Esos últimos dos valen la pena leerlos con cuidado. Un 1.00x plano parece indicar "nada aquí", y no es así: significa que no había nada disponible para la única forma (shape) que este entorno de pruebas ejerce. Mide los mismos modelos en sus propias formas de capa y ambos ganan rendimiento una vez que alimentas más de una fila. Nueve modelos a continuación, incluidos dos modelos de visión y lenguaje, donde solo se mide la torre de lenguaje porque el codificador de visión se ejecuta una vez por imagen en lugar de por token:

![Speedup by model shape](docs/_static/mlx-model-shape-speedup.png)

Cada modelo está cerca de la paridad con una fila, porque está limitado por el ancho de banda y no hay nada por ganar. Cada modelo gana rendimiento a dieciséis filas, de 1.24x a 1.81x, porque los pesos se reutilizan. Esto se mantiene hasta Qwen3.6 27B y ambos VLM. Solo el prefill depende del modelo, y depende exactamente de una cosa:

![Speedup by projection width](docs/_static/mlx-width-cliff.png)

MLX cambia de kernel en algún lugar entre anchos de salida 2048 y 2560, y el que utiliza por debajo de ese umbral es deficiente. Un modelo gana si sus capas son lo suficientemente estrechas como para caer en esa banda. Llama 3.2 1B tiene una proyección descendente de 2048 de ancho y obtiene 3.16x; Llama 3.2 3B tiene una de 3072 y obtiene 1.06x. La profundidad es irrelevante, ya que multiplica ambos lados por igual.

En las formas más anchas ya estamos al **97% del matmul más rápido que esta máquina puede ejecutar**, por lo que no queda margen de mejora allí, sino que simplemente es un límite aún por alcanzar.

![Decode and prefill speedup by model](docs/_static/mlx-model-speedup.png)

![Latency speedup by model](docs/_static/mlx-model-latency-speedup.png)

### Kernels individuales

| | Aceleración |
|---|---|
| Atención, 1 consulta sobre 1024 claves | **1.29x** |
| Atención, 512 consultas, causal | 1.00x |
| Suma residual + RMSNorm, 512 x 4096 | **1.21x** |
| Suma residual + RMSNorm, tamaño de decodificación | 1.00x |

### Dónde meTile *no* es más rápido

- **Decodificación de un solo token: aproximadamente lo mismo que MLX.** Generar un token a la vez está limitado por la velocidad de memoria, no por el kernel. Un kernel de lectura en secuencia sin procesar alcanza un máximo de 121 GB/s en esta máquina y MLX ya se ejecuta al 93-97% de eso, por lo que casi no queda nada por ganar. El agrupamiento por lotes es lo que mueve este número, razón por la cual la tabla anterior comienza en 1 y asciende.
- **INT8: aproximadamente lo mismo que MLX.** meTile no tiene un kernel propio allí y se aparta en lugar de forzar uno. INT4 solía decir lo mismo y ya no lo hace por encima de cuatro filas.
- **Softmax: 0.74x a 0.99x.** El de MLX ya es un kernel fusionado único.

### Intercambiando precisión por velocidad

meTile también puede almacenar partes de un modelo BF16 como INT8 y decodificar **1.37x a 1.75x** más rápido. Esto *no* es la comparación anterior. Es más rápido porque lee menos bytes, no porque el kernel sea mejor.

```python
import mlx.core as mx
from mlx_lm import load
from metile.integrations.mlx_lm import (
    apply_metile_to_mlx_lm,
    autotune_metile_for_mlx_lm,
    prepare_mlx_lm_compressed_attention,
    prepare_mlx_lm_compressed_down,
    prepare_mlx_lm_compressed_gate_up,
    prepare_mlx_lm_compressed_vocab,
)

model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-bf16")

# Store these projections as INT8. The BF16 weights are kept, and any layer that
# fails the accuracy check keeps using them.
compressed = {
    "compressed_down": prepare_mlx_lm_compressed_down(model, format="affine8"),
    "compressed_gate_up": prepare_mlx_lm_compressed_gate_up(model),
    "compressed_attention": prepare_mlx_lm_compressed_attention(model),
    "compressed_vocab": prepare_mlx_lm_compressed_vocab(model),
}

# Time each combination on the real model and keep whatever actually wins.
sample = mx.array([tokenizer.encode("Explain tiled matrix multiplication.")])
plan = autotune_metile_for_mlx_lm(model, sample, quantized_mlp=False, **compressed)

with apply_metile_to_mlx_lm(model=model, plan=plan, **compressed):
    ...  # generate as usual

# Leaving the block restores every patched function.
```

Solo se ve afectada la decodificación de un solo token. El prefill permanece en BF16. Una capa se comprime solo si el siguiente token no cambia y el error de logit se mantiene dentro de un límite fijo, por lo que las capas sensibles a la cuantización conservan sus pesos originales. Los tamaños de grupo se eligen mediante medición. Detalles en la [guía del backend de MLX](docs/guide/mlx-backend.rst).

## Instalación

```bash
pip install -e ".[dev]"

pip install -e ".[mlx-lm]"       # Integración con MLX
pip install -e ".[benchmarks]"   # Renderizador de gráficos
```

## Pruebas y benchmarks

```bash
make test                                      # todo
python -m pytest tests/test_gemm.py -v         # un archivo

make bench                                     # todo
python benchmarks/matched_representation_matrix.py   # la tabla de tamaño de lote anterior
python benchmarks/model_shape_matrix.py              # cada modelo en sus propias formas de capa
python benchmarks/shape_sensitivity.py               # los dos gráficos de forma anteriores
python benchmarks/graph_fusion_speedup.py            # la tabla de kernels anterior
python benchmarks/compile_comparison.py              # meTile vs mx.compile
```

## Documentación

| | |
|---|---|
| [Lenguaje](docs/guide/language.rst) | Escritura de kernels |
| [Operaciones de tile](docs/guide/tile-ops.rst) | El conjunto de operaciones |
| [Memoria](docs/guide/memory.rst) | Disposiciones y espacios de direcciones |
| [Autosintonización](docs/guide/autotuning.rst) | Cómo se seleccionan los horarios |
| [Fusión de grafos](docs/guide/graph-fusion.rst) | Fusión entre operaciones |
| [Backend de MLX](docs/guide/mlx-backend.rst) | Uso de meTile desde MLX y las tablas de benchmark completas |
| [Arquitectura](docs/guide/architecture.rst) | Cómo se ensambla el compilador |

## Enlaces

- [Contribuir](.github/CONTRIBUTING.md)
- [Panel de rendimiento](https://andreslavescu.github.io/meTile/dev/bench/)

## Citas

El álgebra de disposiciones sigue a CuTe, y el lenguaje de kernels sigue a Triton.

Elegir qué reescrituras aplicar es un problema de flujo máximo aquí. Los candidatos superpuestos no pueden aplicarse ambos, por lo que elegir el mejor conjunto es un conjunto independiente de peso máximo, lo que se reduce a un corte mínimo s-t exacto. Dos fuentes que condujeron a este encuadre de la idea. PyTorch resuelve un problema de compilador diferente de la misma manera, utilizando el corte mínimo para decidir qué activaciones guardar frente a recomputar. Al estudiar CS 341, las notas de Lap Chi Lau presentan la reducción en sí misma, incluido el problema de selección de proyectos, que es la forma que el selector utiliza realmente. Tras seguir su clase (Primavera 2025), esto inspiró esta línea de pensamiento.

```bibtex
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
```

## Licencia

MIT
