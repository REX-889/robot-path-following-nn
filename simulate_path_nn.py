from dataclasses import dataclass

from direct_path_nn import DirectNNVoltageController
from robot_model import RobotParams, rk4_step


@dataclass(frozen=True)
class DirectPathSample:
    t: float
    progress_s: float
    nearest_distance: float


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
        voltage = controller.compute_voltage(state, path, progress_s, remaining)

        log.append(
            DirectPathSample(
                t=t,
                progress_s=progress_s,
                nearest_distance=nearest_distance,
            )
        )

        if stop_at_end and not path.closed and remaining < 0.03:
            break

        state = rk4_step(t, state, dt, voltage, params)
        t += dt

    return log
