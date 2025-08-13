import collections

# --- Core Algorithm Ported from SOURCE Repository ---
# Based on `Algorithms/DB-RPNI/`


def get_successors(state, string, mealy_machine):
    """Follows a string of inputs from a state to find the resulting output and final state."""
    output_string = []
    current_state = state
    for i in string:
        output_string.append(mealy_machine["outputs"].get(current_state, {}).get(i))
        current_state = mealy_machine["transitions"].get(current_state, {}).get(i)
        if current_state is None:
            return None, None
    return current_state, tuple(output_string)


def get_distinguishing_string(s1, s2, mealy_machine, alphabet):
    """Finds a string that distinguishes two states."""
    queue = collections.deque([("",)])
    visited = {("",)}

    while queue:
        current_string = queue.popleft()[0]
        _, s1_output = get_successors(s1, current_string, mealy_machine)
        _, s2_output = get_successors(s2, current_string, mealy_machine)

        if s1_output != s2_output:
            return current_string

        for symbol in alphabet:
            next_string = current_string + (symbol,)
            if next_string not in visited:
                visited.add(next_string)
                queue.append((next_string,))
    return None


def merge_states(mealy_machine, s1, s2):
    """Merges state s2 into state s1."""
    if s2 in mealy_machine["states"]:
        mealy_machine["states"].remove(s2)

    for state in mealy_machine["states"]:
        for symbol, next_state in list(
            mealy_machine["transitions"].get(state, {}).items()
        ):
            if next_state == s2:
                mealy_machine["transitions"][state][symbol] = s1

    if s2 in mealy_machine["transitions"]:
        for symbol, next_state in mealy_machine["transitions"][s2].items():
            if symbol not in mealy_machine["transitions"].get(s1, {}):
                mealy_machine["transitions"].setdefault(s1, {})[symbol] = next_state

    if s2 in mealy_machine["outputs"]:
        for symbol, output in mealy_machine["outputs"][s2].items():
            if symbol not in mealy_machine["outputs"].get(s1, {}):
                mealy_machine["outputs"].setdefault(s1, {})[symbol] = output

    if s2 in mealy_machine["transitions"]:
        del mealy_machine["transitions"][s2]
    if s2 in mealy_machine["outputs"]:
        del mealy_machine["outputs"][s2]

    return mealy_machine


def find_merge(mealy_machine, alphabet):
    """Finds a valid pair of states to merge."""
    states_to_consider = sorted(list(mealy_machine["states"]))
    for i in range(len(states_to_consider)):
        for j in range(i + 1, len(states_to_consider)):
            s1 = states_to_consider[i]
            s2 = states_to_consider[j]

            potential_machine = merge_states(mealy_machine.copy(), s1, s2)
            if get_distinguishing_string(s1, s2, potential_machine, alphabet) is None:
                return s1, s2
    return None, None


def construct_APTA(trajectories, alphabet):
    """Constructs an Augmented Prefix Tree Automaton from trajectories."""
    states = {0}
    transitions = {0: {}}
    outputs = {0: {}}
    state_counter = 1

    for inputs, out_seq in trajectories:
        current_state = 0
        for i in range(len(inputs)):
            symbol = inputs[i]
            output = out_seq[i]

            if symbol in transitions.get(current_state, {}):
                # Check for output consistency
                if outputs[current_state][symbol] != output:
                    raise ValueError(
                        f"Inconsistent output for state {current_state} and symbol {symbol}"
                    )
                current_state = transitions[current_state][symbol]
            else:
                transitions.setdefault(current_state, {})[symbol] = state_counter
                outputs.setdefault(current_state, {})[symbol] = output
                states.add(state_counter)
                current_state = state_counter
                state_counter += 1

    return {
        "states": states,
        "initial_state": 0,
        "transitions": transitions,
        "outputs": outputs,
    }


def find_machine(trajectories, alphabet):
    """Main RPNI-style algorithm to infer a Mealy machine."""
    mealy_machine = construct_APTA(trajectories, alphabet)

    while True:
        s1, s2 = find_merge(mealy_machine, alphabet)
        if s1 is not None and s2 is not None:
            mealy_machine = merge_states(mealy_machine, s1, s2)
        else:
            break
    return mealy_machine


def format_machine(machine):
    """Create a canonical, comparable representation of the machine."""
    # Deep copy to avoid modifying the original
    m = {
        "states": sorted(list(machine["states"])),
        "initial_state": machine["initial_state"],
        "transitions": {
            k: machine["transitions"][k] for k in sorted(machine["transitions"].keys())
        },
        "outputs": {
            k: machine["outputs"][k] for k in sorted(machine["outputs"].keys())
        },
    }
    return m


# --- Experiment Definition ---

if __name__ == "__main__":
    print("--- DB-RPNI State Machine Inference Experiment ---")

    # 1. Define Ground Truth: A simple 2-state lock automaton
    # S0: Locked, S1: Unlocked
    # Inputs: 'k' (key), 'p' (push)
    # Outputs: '0' (no change/locked), '1' (action success)
    ground_truth_machine = {
        "states": [0, 1],
        "initial_state": 0,
        "transitions": {0: {"p": 0, "k": 1}, 1: {"p": 1, "k": 1}},
        "outputs": {0: {"p": "0", "k": "1"}, 1: {"p": "1", "k": "0"}},
    }
    print("\nGround Truth Machine:")
    print(ground_truth_machine)

    # 2. Define sample trajectories consistent with the ground truth machine
    # Each trajectory is a tuple of (input_sequence, output_sequence)
    alphabet = ("k", "p")
    sample_trajectories = [
        (("p",), ("0",)),
        (("k",), ("1",)),
        (("k", "p"), ("1", "1")),
        (("k", "k"), ("1", "0")),
        (("p", "k"), ("0", "1")),
        (("p", "p"), ("0", "0")),
        (("k", "p", "k"), ("1", "1", "0")),
    ]
    print("\nUsing Sample Trajectories:")
    for t in sample_trajectories:
        print(t)

    # 3. Run the inference algorithm
    inferred_machine_raw = find_machine(sample_trajectories, alphabet)

    # 4. Format for comparison
    inferred_machine = format_machine(inferred_machine_raw)
    print("\nInferred Machine:")
    print(inferred_machine)

    # 5. Verify correctness
    print("\n--- Verification ---")
    try:
        assert inferred_machine == ground_truth_machine
        print("[SUCCESS] Inferred machine matches ground truth.")
    except AssertionError:
        print("[FAILURE] Inferred machine does NOT match ground truth.")
