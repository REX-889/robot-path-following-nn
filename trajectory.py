from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt


@dataclass(frozen=True)
class Reference:
    # Punkt trajektorii zadanej w danej chwili czasu.
    x: float
    y: float
    theta: float
    v: float
    omega: float
    s: float = 0.0


def figure_eight_reference(t: float) -> Reference:
    # Trajektoria osemkowa podobna do tej z Simulinka.
    # x(t), y(t) definiuja tor, a predkosci wynikaja z pochodnych.
    a = 0.8
    b = 0.4
    w = 0.25

    x = a * sin(w * t)
    y = b * sin(2.0 * w * t)

    dx = a * w * cos(w * t)
    dy = 2.0 * b * w * cos(2.0 * w * t)
    ddx = -a * w**2 * sin(w * t)
    ddy = -4.0 * b * w**2 * sin(2.0 * w * t)

    theta = atan2(dy, dx)
    v = sqrt(dx**2 + dy**2)

    denominator = dx**2 + dy**2
    omega = 0.0 if denominator < 1e-9 else (ddy * dx - ddx * dy) / denominator

    return Reference(x=x, y=y, theta=theta, v=v, omega=omega)


def circle_reference(t: float) -> Reference:
    # Trajektoria po okregu. Dobra do testowania stalego skretu.
    radius = 1.0
    w = 0.2

    x = radius * cos(w * t)
    y = radius * sin(w * t)
    dx = -radius * w * sin(w * t)
    dy = radius * w * cos(w * t)

    theta = atan2(dy, dx)
    v = radius * w
    omega = w

    return Reference(x=x, y=y, theta=theta, v=v, omega=omega)


def sinus_reference(t: float) -> Reference:
    # Trajektoria sinusoidalna: robot jedzie w osi X, a Y faluje.
    speed_x = 0.18
    amplitude = 0.35
    length = 3.0

    x = speed_x * t
    phase = 2.0 * pi * x / length
    y = amplitude * sin(phase)

    dx = speed_x
    dy = amplitude * cos(phase) * 2.0 * pi * speed_x / length
    ddy = -amplitude * sin(phase) * (2.0 * pi * speed_x / length) ** 2

    theta = atan2(dy, dx)
    v = sqrt(dx**2 + dy**2)

    denominator = dx**2 + dy**2
    omega = 0.0 if denominator < 1e-9 else (ddy * dx) / denominator

    return Reference(x=x, y=y, theta=theta, v=v, omega=omega)
