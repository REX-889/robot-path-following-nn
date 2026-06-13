from dataclasses import dataclass
from math import atan2, cos, pi, sin

from trajectory import Reference


@dataclass(frozen=True)
class TrackingGains:
    # Wzmocnienia regulatora sledzenia trajektorii.
    # k1 poprawia blad wzdluz osi robota, k2 blad boczny, k3 blad kata.
    k1: float = 1.8
    k2: float = 4.0
    k3: float = 1.5


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


def tracking_controller(state, reference: Reference, gains: TrackingGains):
    # Regulator ASP z modelu Simulinka.
    # Wejscie: aktualny stan robota i punkt trajektorii zadanej.
    # Wyjscie: zadane predkosci platformy [omega_ref, v_ref].
    e1, e2, e3 = tracking_errors(state, reference)

    v_ref = reference.v * cos(e3) + gains.k1 * e1
    omega_ref = reference.omega + gains.k2 * reference.v * e2 + gains.k3 * reference.v * sin(e3)

    return omega_ref, v_ref
