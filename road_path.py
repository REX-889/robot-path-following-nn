from bisect import bisect_right
from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt
from random import Random

from controller import wrap_to_pi
from trajectory import Reference


@dataclass(frozen=True)
class PathPose:
    # Punkt na torze opisany wspolrzednymi, orientacja stycznej i dlugoscia luku.
    x: float
    y: float
    theta: float
    s: float


class RoadPath:
    # Tor/droga reprezentowana jako gesta lista punktow.
    # Parametryzacja odbywa sie po dlugosci luku s, a nie po czasie.
    def __init__(self, points, closed=True):
        if len(points) < 2:
            raise ValueError("Tor musi miec co najmniej dwa punkty.")

        self.closed = closed
        self.points = list(points)
        if closed and self.points[0] != self.points[-1]:
            self.points.append(self.points[0])

        self.lengths = [0.0]
        for i in range(1, len(self.points)):
            x0, y0 = self.points[i - 1]
            x1, y1 = self.points[i]
            self.lengths.append(self.lengths[-1] + sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

        self.total_length = self.lengths[-1]

    def sample(self, s: float) -> PathPose:
        # Pobranie pozycji na torze dla zadanej dlugosci luku.
        if self.closed:
            s = s % self.total_length
        else:
            s = min(max(s, 0.0), self.total_length)

        i = min(max(1, bisect_right(self.lengths, s)), len(self.points) - 1)
        s0 = self.lengths[i - 1]
        s1 = self.lengths[i]
        x0, y0 = self.points[i - 1]
        x1, y1 = self.points[i]

        segment_length = max(s1 - s0, 1e-9)
        alpha = (s - s0) / segment_length
        x = x0 + alpha * (x1 - x0)
        y = y0 + alpha * (y1 - y0)
        theta = atan2(y1 - y0, x1 - x0)

        return PathPose(x=x, y=y, theta=theta, s=s)

    def nearest_s(self, x: float, y: float, previous_s: float = 0.0, search_window: float = 1.5):
        # Najblizszy punkt na sciezce dla aktualnej pozycji robota.
        # Dla otwartej sciezki ograniczamy wyszukiwanie do okna przed/za
        # poprzednim postepem, zeby postep nie skakal daleko wstecz.
        best_s = previous_s
        best_distance2 = float("inf")

        if self.closed:
            start_index = 1
            end_index = len(self.points)
        else:
            s_min = max(0.0, previous_s - search_window)
            s_max = min(self.total_length, previous_s + search_window)
            start_index = max(1, bisect_right(self.lengths, s_min) - 1)
            end_index = min(len(self.points), bisect_right(self.lengths, s_max) + 1)

        for i in range(start_index, end_index):
            x0, y0 = self.points[i - 1]
            x1, y1 = self.points[i]
            dx = x1 - x0
            dy = y1 - y0
            segment2 = dx * dx + dy * dy
            if segment2 <= 1e-12:
                continue

            alpha = ((x - x0) * dx + (y - y0) * dy) / segment2
            alpha = min(max(alpha, 0.0), 1.0)
            px = x0 + alpha * dx
            py = y0 + alpha * dy
            distance2 = (x - px) ** 2 + (y - py) ** 2

            if distance2 < best_distance2:
                best_distance2 = distance2
                best_s = self.lengths[i - 1] + alpha * sqrt(segment2)

        if not self.closed:
            best_s = max(previous_s, min(best_s, self.total_length))

        return best_s, sqrt(best_distance2)

    def reference_at_s(self, s: float, speed: float = 0.16) -> Reference:
        # Referencja geometryczna dla path following, bez narzucania czasu.
        pose = self.sample(s)
        remaining = max(0.0, self.total_length - pose.s) if not self.closed else self.total_length
        speed_scale = min(1.0, remaining / 0.7) if not self.closed else 1.0
        v_ref = speed * max(0.0, speed_scale)

        ds = 0.02
        pose_before = self.sample(pose.s - ds)
        pose_after = self.sample(pose.s + ds)
        dtheta = wrap_to_pi(pose_after.theta - pose_before.theta)
        curvature = dtheta / (2.0 * ds)

        return Reference(
            x=pose.x,
            y=pose.y,
            theta=pose.theta,
            v=v_ref,
            omega=v_ref * curvature,
            s=pose.s,
        )

    def reference(self, t: float, speed: float = 0.16) -> Reference:
        # Zamiana czasu na punkt referencyjny jadacy po torze ze stala predkoscia.
        s = speed * t
        pose = self.sample(s)

        # Predkosc katowa wynika z krzywizny toru: omega = v * kappa.
        ds = 0.02
        pose_before = self.sample(s - ds)
        pose_after = self.sample(s + ds)
        dtheta = wrap_to_pi(pose_after.theta - pose_before.theta)
        curvature = dtheta / (2.0 * ds)
        omega = speed * curvature

        return Reference(
            x=pose.x,
            y=pose.y,
            theta=pose.theta,
            v=speed,
            omega=omega,
            s=pose.s,
        )

    def local_preview(self, state, s: float, lookaheads):
        # Punkty toru przed robotem przeliczone do lokalnego ukladu robota.
        theta, x, y, _omega, _v = state
        preview = []
        for distance in lookaheads:
            pose = self.sample(s + distance)
            dx = pose.x - x
            dy = pose.y - y
            local_x = cos(theta) * dx + sin(theta) * dy
            local_y = -sin(theta) * dx + cos(theta) * dy
            local_theta = wrap_to_pi(pose.theta - theta)
            preview.extend([local_x, local_y, local_theta])
        return preview


def _catmull_rom_point(p0, p1, p2, p3, u: float):
    # Zamkniety spline Catmulla-Roma: przechodzi przez punkty kontrolne
    # i daje gladki tor bez potrzeby SciPy.
    u2 = u * u
    u3 = u2 * u
    x = 0.5 * (
        2.0 * p1[0]
        + (-p0[0] + p2[0]) * u
        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * u2
        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * u3
    )
    y = 0.5 * (
        2.0 * p1[1]
        + (-p0[1] + p2[1]) * u
        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * u2
        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * u3
    )
    return x, y


def make_closed_spline_path(seed: int = 0, control_count: int = 10, samples_per_segment: int = 60) -> RoadPath:
    # Docelowy tor: zamkniety, gladki spline z losowo zaburzonych punktow
    # kontrolnych rozmieszczonych wokol elipsy.
    rng = Random(seed)
    controls = []

    for i in range(control_count):
        a = 2.0 * pi * i / control_count
        radius_x = 2.0 + rng.uniform(-0.35, 0.35)
        radius_y = 1.25 + rng.uniform(-0.25, 0.25)
        x = radius_x * cos(a) + rng.uniform(-0.15, 0.15)
        y = radius_y * sin(a) + rng.uniform(-0.15, 0.15)
        controls.append((x, y))

    points = []
    n = len(controls)
    for i in range(n):
        p0 = controls[(i - 1) % n]
        p1 = controls[i % n]
        p2 = controls[(i + 1) % n]
        p3 = controls[(i + 2) % n]
        for j in range(samples_per_segment):
            u = j / samples_per_segment
            points.append(_catmull_rom_point(p0, p1, p2, p3, u))

    return RoadPath(points, closed=True)


DEFAULT_SPLINE_PATH = make_closed_spline_path(seed=21)
