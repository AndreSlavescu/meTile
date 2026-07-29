"""Ordering, confirming and selecting between candidate kernels.

A candidate is any sequence whose first element identifies it and whose second element is a
thunk that runs it. Anything after that is the caller's own metadata and is ignored here.

The two measurement shapes exist for different jobs. `round_robin` times every candidate
together and is for cheap triage of a large field. `confirm_pairwise` times each candidate
against a baseline on its own and is for deciding, because a candidate's measured time
depends on how many others share the rotation: a kernel that wins head to head can read
slower in a crowded tournament.
"""

import statistics


def token_bucket(tokens):
    """Round a row count up to a power of two, so nearby shapes share a tuning result."""
    return 1 << max(tokens - 1, 0).bit_length()


def round_robin(candidates, rounds, measure):
    """Time every candidate in each round, rotating and reversing the order between rounds.

    Rotating spreads any position effect evenly, and reversing alternate rounds keeps a
    candidate from always following the same neighbour. This ranks a large field cheaply.
    It does not rank it precisely: see `confirm_pairwise` before acting on the order.
    """
    candidates = list(candidates)
    if not candidates:
        return {}
    samples = {candidate[0]: [] for candidate in candidates}
    for index in range(rounds):
        shift = index % len(candidates)
        ordered = candidates[shift:] + candidates[:shift]
        if index & 1:
            ordered.reverse()
        for candidate in ordered:
            samples[candidate[0]].append(measure(candidate[1]))
    return samples


def confirm_pairwise(candidates, baseline, rounds, measure):
    """Time each candidate against `baseline` alone, and return {key: seconds}.

    Every candidate is measured in an identically sized context, which is what makes the
    results comparable to each other. Ranking uses the ratio to the baseline rather than
    the raw time, so drift between one pairing and the next cancels; the returned seconds
    are those ratios rebuilt against a single baseline reading, so absolute-time policy
    like a switch margin still applies unchanged.
    """
    candidates = list(candidates)
    baseline_candidate = next(candidate for candidate in candidates if candidate[0] == baseline)
    others = [candidate for candidate in candidates if candidate[0] != baseline]
    if not others:
        samples = round_robin([baseline_candidate], rounds, measure)
        return {baseline: statistics.median(samples[baseline])}

    ratios = {}
    baseline_times = []
    for candidate in others:
        samples = round_robin([baseline_candidate, candidate], rounds, measure)
        baseline_time = statistics.median(samples[baseline])
        baseline_times.append(baseline_time)
        ratios[candidate[0]] = statistics.median(samples[candidate[0]]) / baseline_time

    reference = statistics.median(baseline_times)
    timings = {baseline: reference}
    for key, ratio in ratios.items():
        timings[key] = ratio * reference
    return timings


def select_fastest(results, baseline, margin_for, *, tie_cutoff=1.0025, tie_break=None):
    """Pick a candidate, or the baseline when nothing beats it by enough to be worth it.

    `results` are (seconds, weight, key) triples. `margin_for(key)` gives the fraction a
    candidate must win by; anything smaller is inside the noise this measurement can
    resolve, and switching on it means committing to a kernel that is not actually faster.

    Among candidates within `tie_cutoff` of the fastest, `tie_break` chooses. Passing the
    description length there prefers the simpler kernel when speed cannot separate them.
    """
    alternatives = [result for result in results if result[2] != baseline]
    baseline_result = next(result for result in results if result[2] == baseline)
    if not alternatives:
        return baseline_result[2]

    fastest = min(alternatives, key=lambda result: result[0])
    if fastest[0] >= baseline_result[0] * (1.0 - margin_for(fastest[2])):
        return baseline_result[2]

    contenders = [result for result in alternatives if result[0] <= fastest[0] * tie_cutoff]
    if tie_break is None:
        return min(contenders, key=lambda result: result[0])[2]
    return tie_break(contenders)
