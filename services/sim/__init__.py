"""Replaying exported scenarios, so a scenario document is checked for describing something possible.

An OpenSCENARIO file that parses is not a scenario that runs. An actor placed off the road network, a
speed no vehicle reaches, a trigger whose condition never fires: each produces a valid document that
describes nothing, and nothing in this engine could tell the difference between that and a good one.
"""
