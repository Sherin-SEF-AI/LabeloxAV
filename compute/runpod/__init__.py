"""Programmatic RunPod control for the warm cloud-GPU session: an orchestrator (provision, status,
terminate, pause, list) that returns values and raises, the cost model + safety guards, and the
warm-session manager that holds at most one pod and guarantees it is torn down."""
