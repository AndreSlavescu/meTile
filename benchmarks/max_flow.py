import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metile.compiler.max_flow import FlowNetwork, clear_flow_solver_cache


def build_network(vertex_count, degree, seed):
    random_source = random.Random(seed)
    network = FlowNetwork()
    for source in range(vertex_count - 1):
        network.add_edge(source, source + 1, random_source.randint(1, 100))
        for _ in range(degree - 1):
            target = random_source.randrange(vertex_count)
            if target != source:
                network.add_edge(source, target, random_source.randint(1, 100))
    return network


def measure(network, solver, trials):
    samples = []
    result = None
    for _ in range(trials):
        start = time.perf_counter_ns()
        result = network.minimum_cut(0, network.vertex_count - 1, solver=solver)
        samples.append(time.perf_counter_ns() - start)
    return statistics.median(samples) / 1e6, result


def main():
    parser = argparse.ArgumentParser(description="Benchmark exact meTile min-cut solvers")
    parser.add_argument("--vertices", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--degrees", nargs="+", type=int, default=[2, 6, 12])
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print(
        f"{'vertices':>8} {'edges':>8} {'degree':>7} {'dinic ms':>10} "
        f"{'push ms':>10} {'tune ms':>10} {'warm ms':>10} {'warm/best':>10} {'policy':>12}"
    )
    for vertex_count in args.vertices:
        for degree in args.degrees:
            network = build_network(vertex_count, degree, args.seed + vertex_count + degree)
            dinic_ms, dinic_result = measure(network, "dinic", args.trials)
            push_ms, push_result = measure(network, "push_relabel", args.trials)
            clear_flow_solver_cache()
            start = time.perf_counter_ns()
            auto_result = network.minimum_cut(0, vertex_count - 1)
            tune_ms = (time.perf_counter_ns() - start) / 1e6
            auto_ms, auto_result = measure(network, "auto", args.trials)
            if abs(dinic_result[0] - push_result[0]) > 1e-9:
                raise RuntimeError("exact solvers produced different maximum-flow values")
            if abs(dinic_result[0] - auto_result[0]) > 1e-9:
                raise RuntimeError("automatic solver produced a different maximum-flow value")
            print(
                f"{vertex_count:8d} {network.edge_count:8d} {degree:7d} "
                f"{dinic_ms:10.3f} {push_ms:10.3f} {tune_ms:10.3f} {auto_ms:10.3f} "
                f"{auto_ms / min(dinic_ms, push_ms):10.2f} "
                f"{network.select_solver(source=0, sink=vertex_count - 1):>12}"
            )


if __name__ == "__main__":
    main()
