"""FP32 recurrent delta-rule kernels, without normalization or model adapters.

State is [batch, head, key, value]; sequence inputs are BTHD, not BHSD.
KDA uses key-channel log decay; Gated DeltaNet uses scalar log decay per head.
The decay-before-correction recurrence follows equation (1) of
https://arxiv.org/html/2510.26692v1 . Official model sources are
https://github.com/MoonshotAI/Kimi-Linear and
https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_next
These independently written kernels implement the core, not either full model.
"""

import metile


@metile.kernel
def gated_delta_forward_kernel(
    Query,
    Key,
    Value,
    LogDecay,
    Beta,
    InitialState,
    Output,
    FinalState,
    States,
    Sequence,
    scale,
    BATCH: metile.constexpr,
    HEADS: metile.constexpr,
    KEY_DIM: metile.constexpr,
    VALUE_DIM: metile.constexpr,
    CHANNEL_DECAY: metile.constexpr,
    SAVE_STATES: metile.constexpr,
    BLOCK: metile.constexpr,
):
    queries = metile.tensor(Query, shape=(BATCH, Sequence, HEADS, KEY_DIM), access="read")
    key_values = metile.tensor(Key, shape=(BATCH, Sequence, HEADS, KEY_DIM), access="read")
    values = metile.tensor(Value, shape=(BATCH, Sequence, HEADS, VALUE_DIM), access="read")
    betas = metile.tensor(Beta, shape=(BATCH, Sequence, HEADS), access="read")
    if CHANNEL_DECAY:
        decays = metile.tensor(LogDecay, shape=(BATCH, Sequence, HEADS, KEY_DIM), access="read")
    else:
        decays = metile.tensor(LogDecay, shape=(BATCH, Sequence, HEADS), access="read")
    initial_states = metile.tensor(
        InitialState, shape=(BATCH, HEADS, KEY_DIM, VALUE_DIM), access="read"
    )
    final_states = metile.tensor(
        FinalState, shape=(BATCH, HEADS, KEY_DIM, VALUE_DIM), access="write"
    )
    outputs = metile.tensor(Output, shape=(BATCH, Sequence, HEADS, VALUE_DIM, 1), access="write")
    if SAVE_STATES:
        saved_states = metile.tensor(
            States, shape=(BATCH, Sequence + 1, HEADS, KEY_DIM, VALUE_DIM), access="write"
        )
    column = metile.program_id(0)
    value_index = column % VALUE_DIM
    head = (column // VALUE_DIM) % HEADS
    batch = column // (HEADS * VALUE_DIM)
    keys = metile.arange(0, BLOCK)
    state = metile.loop_state(initial_states.load((batch, head, keys, value_index)))
    if SAVE_STATES:
        saved_states.store((batch, 0, head, keys, value_index), state.value)
    for token in metile.tile_range(0, Sequence, 1):
        query = queries.load((batch, token, head, keys))
        key = key_values.load((batch, token, head, keys))
        value = values.load((batch, token, head, value_index))
        beta = betas.load((batch, token, head))
        if CHANNEL_DECAY:
            log_decay = decays.load((batch, token, head, keys))
        else:
            log_decay = decays.load((batch, token, head))
        decayed = state.value * metile.exp(log_decay)
        prediction = metile.sum(decayed * key)
        correction = beta * (value - prediction)
        next_state = decayed + key * correction
        state.update(next_state)
        output = metile.sum(next_state * query) * scale
        outputs.store((batch, token, head, value_index, keys), output)
        if SAVE_STATES:
            saved_states.store((batch, token + 1, head, keys, value_index), next_state)
    final_states.store((batch, head, keys, value_index), state.value)


@metile.kernel
def gated_delta_backward_kernel(
    Query,
    Key,
    Value,
    LogDecay,
    Beta,
    States,
    OutputGradient,
    FinalStateGradient,
    QueryPartial,
    KeyPartial,
    DecayPartial,
    BetaPartial,
    ValueGradient,
    InitialStateGradient,
    Sequence,
    scale,
    BATCH: metile.constexpr,
    HEADS: metile.constexpr,
    KEY_DIM: metile.constexpr,
    VALUE_DIM: metile.constexpr,
    CHANNEL_DECAY: metile.constexpr,
    BLOCK: metile.constexpr,
):
    queries = metile.tensor(Query, shape=(BATCH, Sequence, HEADS, KEY_DIM), access="read")
    key_values = metile.tensor(Key, shape=(BATCH, Sequence, HEADS, KEY_DIM), access="read")
    values = metile.tensor(Value, shape=(BATCH, Sequence, HEADS, VALUE_DIM), access="read")
    betas = metile.tensor(Beta, shape=(BATCH, Sequence, HEADS), access="read")
    if CHANNEL_DECAY:
        decays = metile.tensor(LogDecay, shape=(BATCH, Sequence, HEADS, KEY_DIM), access="read")
    else:
        decays = metile.tensor(LogDecay, shape=(BATCH, Sequence, HEADS), access="read")
    saved_states = metile.tensor(
        States, shape=(BATCH, Sequence + 1, HEADS, KEY_DIM, VALUE_DIM), access="read"
    )
    output_gradients = metile.tensor(
        OutputGradient, shape=(BATCH, Sequence, HEADS, VALUE_DIM), access="read"
    )
    final_state_gradients = metile.tensor(
        FinalStateGradient, shape=(BATCH, HEADS, KEY_DIM, VALUE_DIM), access="read"
    )
    initial_state_gradients = metile.tensor(
        InitialStateGradient, shape=(BATCH, HEADS, KEY_DIM, VALUE_DIM), access="write"
    )
    value_gradients = metile.tensor(
        ValueGradient, shape=(BATCH, Sequence, HEADS, VALUE_DIM, 1), access="write"
    )
    query_partials = metile.tensor(
        QueryPartial, shape=(BATCH, Sequence, HEADS, VALUE_DIM, KEY_DIM), access="write"
    )
    key_partials = metile.tensor(
        KeyPartial, shape=(BATCH, Sequence, HEADS, VALUE_DIM, KEY_DIM), access="write"
    )
    decay_partials = metile.tensor(
        DecayPartial, shape=(BATCH, Sequence, HEADS, VALUE_DIM, KEY_DIM), access="write"
    )
    beta_partials = metile.tensor(
        BetaPartial, shape=(BATCH, Sequence, HEADS, VALUE_DIM, 1), access="write"
    )
    column = metile.program_id(0)
    value_index = column % VALUE_DIM
    head = (column // VALUE_DIM) % HEADS
    batch = column // (HEADS * VALUE_DIM)
    keys = metile.arange(0, BLOCK)
    state_gradient = metile.loop_state(final_state_gradients.load((batch, head, keys, value_index)))
    for reverse_token in metile.tile_range(0, Sequence, 1):
        token = Sequence - 1 - reverse_token
        query = queries.load((batch, token, head, keys))
        key = key_values.load((batch, token, head, keys))
        value = values.load((batch, token, head, value_index))
        beta = betas.load((batch, token, head))
        output_gradient = output_gradients.load((batch, token, head, value_index))
        previous_state = saved_states.load((batch, token, head, keys, value_index))
        current_state = saved_states.load((batch, token + 1, head, keys, value_index))
        if CHANNEL_DECAY:
            log_decay = decays.load((batch, token, head, keys))
        else:
            log_decay = decays.load((batch, token, head))
        decay = metile.exp(log_decay)
        decayed = previous_state * decay
        error = value - metile.sum(decayed * key)
        total_gradient = state_gradient.value + query * (output_gradient * scale)
        projected_gradient = metile.sum(total_gradient * key)
        error_gradient = beta * projected_gradient
        decayed_gradient = total_gradient - key * error_gradient
        query_gradient = current_state * (output_gradient * scale)
        key_gradient = beta * total_gradient * error - decayed * error_gradient
        query_partials.store((batch, token, head, value_index, keys), query_gradient)
        key_partials.store((batch, token, head, value_index, keys), key_gradient)
        decay_partials.store((batch, token, head, value_index, keys), decayed_gradient * decayed)
        beta_partials.store((batch, token, head, value_index, keys), projected_gradient * error)
        value_gradients.store((batch, token, head, value_index, keys), error_gradient)
        state_gradient.update(decayed_gradient * decay)
    initial_state_gradients.store((batch, head, keys, value_index), state_gradient.value)


@metile.kernel
def gated_delta_reduce_gradients_kernel(
    QueryPartial,
    KeyPartial,
    DecayPartial,
    BetaPartial,
    QueryGradient,
    KeyGradient,
    DecayGradient,
    BetaGradient,
    ROWS: metile.constexpr,
    KEY_DIM: metile.constexpr,
    VALUE_DIM: metile.constexpr,
    CHANNEL_DECAY: metile.constexpr,
    BLOCK: metile.constexpr,
):
    query_partials = metile.tensor(QueryPartial, shape=(ROWS, VALUE_DIM, KEY_DIM), access="read")
    key_partials = metile.tensor(KeyPartial, shape=(ROWS, VALUE_DIM, KEY_DIM), access="read")
    decay_partials = metile.tensor(DecayPartial, shape=(ROWS, VALUE_DIM, KEY_DIM), access="read")
    beta_partials = metile.tensor(BetaPartial, shape=(ROWS, VALUE_DIM), access="read")
    query_gradients = metile.tensor(QueryGradient, shape=(ROWS, KEY_DIM), access="write")
    key_gradients = metile.tensor(KeyGradient, shape=(ROWS, KEY_DIM), access="write")
    if CHANNEL_DECAY:
        decay_gradients = metile.tensor(DecayGradient, shape=(ROWS, KEY_DIM), access="write")
    else:
        decay_gradients = metile.tensor(DecayGradient, shape=(ROWS, 1), access="write")
    beta_gradients = metile.tensor(BetaGradient, shape=(ROWS, 1), access="write")
    row = metile.program_id(0)
    keys = metile.arange(0, BLOCK)
    query_gradient = metile.loop_state(metile.zeros((BLOCK,), dtype="f32"))
    key_gradient = metile.loop_state(metile.zeros((BLOCK,), dtype="f32"))
    decay_gradient = metile.loop_state(metile.zeros((BLOCK,), dtype="f32"))
    for value_index in metile.tile_range(0, VALUE_DIM, 1):
        query_gradient.update(query_gradient.value + query_partials.load((row, value_index, keys)))
        key_gradient.update(key_gradient.value + key_partials.load((row, value_index, keys)))
        decay_gradient.update(decay_gradient.value + decay_partials.load((row, value_index, keys)))
    query_gradients.store((row, keys), query_gradient.value)
    key_gradients.store((row, keys), key_gradient.value)
    if CHANNEL_DECAY:
        decay_gradients.store((row, keys), decay_gradient.value)
    else:
        total_decay_gradient = metile.sum(decay_gradient.value)
        decay_gradients.store((row, keys), total_decay_gradient)
    beta_gradient = metile.sum(beta_partials.load((row, keys)))
    beta_gradients.store((row, keys), beta_gradient)
