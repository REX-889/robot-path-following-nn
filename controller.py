from math import atan2, cos, sin

from trajectory import Reference


def wrap_to_pi(angle: float) -> float:
    # Sprowadzenie kata do zakresu [-pi, pi].
    return atan2(sin(angle), cos(angle))


def tracking_errors(state, reference: Reference):
    # Bledy pozycji liczone w lokalnym ukladzie robota.
    # e1 - blad do przodu/tylu robota,
    # e2 - blad boczny,
    # e3 - blad orientacji.
    theta, x, y, _omega, _v = state

    dx = reference.x - x
    dy = reference.y - y
    e1 = cos(theta) * dx + sin(theta) * dy
    e2 = -sin(theta) * dx + cos(theta) * dy
    e3 = wrap_to_pi(reference.theta - theta)

    return e1, e2, e3
