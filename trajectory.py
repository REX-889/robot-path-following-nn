from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    # Punkt referencyjny sciezki.
    x: float
    y: float
    theta: float
    v: float
    omega: float
    s: float = 0.0
