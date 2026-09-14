"""Synthetic frames that nothing measuring a model can ever see.

Everything here writes frames with `Frame.origin != 'real'` and labels with `state = source = 'synthetic'`
(see `core/origin.py`). The quarantine is structural: allow-list readers (gold, precision draws, control
samples, settlement) never select the synthetic state, and every deny-list reader says `Frame.origin == REAL`.
The trainer is the one consumer that opts in, through `BuildSpec.include_synthetic`.
"""
