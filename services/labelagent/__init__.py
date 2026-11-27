"""The label agent: checks a machine label's class against several independent specialists before the gate.

Built measurement-first. `truth` turns human rulings into an answer key, `benchmark` replays any classifier
stack against it, and nothing in the agent is switched on until the benchmark says it fixes more labels than
it breaks.
"""
