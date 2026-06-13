from dataclasses import dataclass

from controller import tracking_errors
from direct_path_nn import DirectNNVoltageController
from robot_model import RobotParams, rk4_step
from trajectory import Reference


@dataclass(frozen=True)
class DirectPathSample:
    t: float
    state: tuple[float, float, float, float, float]
    reference: Reference
    voltage: tuple[float, float]
    errors: tuple[float, float, float]
    global_errors: tuple[float, float, float]
    progress_s: float
    nearest_distance: float
    remaining: float


def initial_state_for_path(path):
    start = path.sample(0.0)
    return start.theta, start.x, start.y, 0.0, 0.0


def simulate_direct_path_following(
    path,
    controller=None,
    duration=45.0,
    dt=0.01,
    initial_state=None,
    stop_at_end=True,
):
    # Symulacja path following bez PI/PIC: NN zwraca bezposrednio napiecia.
    params = RobotParams()
    controller = controller or DirectNNVoltageController()
    if hasattr(controller, "reset"):
        controller.reset()
    state = initial_state if initial_state is not None else initial_state_for_path(path)
    progress_s = 0.0
    t = 0.0
    log = []

    for _ in range(int(duration / dt) + 1):
        progress_s, nearest_distance = path.nearest_s(state[1], state[2], progress_s)
        remaining = path.total_length if path.closed else max(0.0, path.total_length - progress_s)
        reference = path.reference_at_s(progress_s)
        voltage = controller.compute_voltage(state, path, progress_s, remaining)
        errors = tracking_errors(state, reference)
        global_errors = (
            reference.x - state[1],
            reference.y - state[2],
            errors[2],
        )

        log.append(
            DirectPathSample(
                t=t,
                state=state,
                reference=reference,
                voltage=voltage,
                errors=errors,
                global_errors=global_errors,
                progress_s=progress_s,
                nearest_distance=nearest_distance,
                remaining=remaining,
            )
        )

        if stop_at_end and not path.closed and remaining < 0.03:
            break

        state = rk4_step(t, state, dt, voltage, params)
        t += dt

    return log
