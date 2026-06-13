from math import atan2, cos

from direct_path_nn import DEFAULT_CRUISE_SPEED, target_speed_for_path
from robot_model import (
    RobotParams,
    back_emf,
    resistance_forces,
    h_tilde_times_u,
    motor_shaft_speeds,
    wheel_speeds,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    # Ograniczenie wartosci do bezpiecznego zakresu.
    return max(lower, min(upper, value))


def _steady_voltage_for_platform(target_platform, params: RobotParams):
    # Przyblizone odwrocenie modelu dla jazdy ustalonej.
    # Liczymy napiecie potrzebne do utrzymania zadanych omega, v
    # z kompensacja SEM, oporow i tlumien.
    q_h = h_tilde_times_u(target_platform, params)
    q_f = resistance_forces(target_platform, params)
    q_rot = q_h[0] + q_f[0]
    q_lin = q_h[1] + q_f[1]

    a = params.b / (params.r * params.ng)
    c = 1.0 / (params.r * params.ng)
    torque_right = 0.5 * (q_lin / c + q_rot / a)
    torque_left = 0.5 * (q_lin / c - q_rot / a)

    target_wheels = wheel_speeds(target_platform, params)
    target_motor_speeds = motor_shaft_speeds(target_wheels, params)
    e_m = back_emf(target_motor_speeds, params)

    voltage_right = params.R * torque_right / params.km + e_m[0]
    voltage_left = params.R * torque_left / params.km + e_m[1]
    return voltage_right, voltage_left


def expert_voltage_for_path(
    state,
    path,
    progress_s: float,
    voltage_limit: float = 12.0,
    cruise_speed: float = DEFAULT_CRUISE_SPEED,
):
    # Ekspert do uczenia NN. W runtime go nie uzywamy.
    # Ten ekspert jest nastawiony na jak najmniejszy blad od sciezki:
    # - blad boczny dociaga robota do najblizszego punktu toru,
    # - blad kata ustawia robota zgodnie ze styczna toru,
    # - omega_ref dodaje wyprzedzenie wynikajace z krzywizny sciezki.
    # Potem zamieniamy zadane omega/v na napiecia modelem odwrotnym.
    params = RobotParams()
    remaining = path.total_length if path.closed else max(0.0, path.total_length - progress_s)
    target_speed = target_speed_for_path(state, path, progress_s, remaining, cruise_speed)

    _local_x, lateral_error, heading_error = path.local_preview(state, progress_s, (0.0,))[0:3]
    reference = path.reference_at_s(progress_s, speed=target_speed)

    # Przy duzym bledzie lekko zwalniamy, zeby robot nie "przestrzelil" zakretu.
    v_cmd = target_speed * _clamp(cos(heading_error), 0.35, 1.0)
    if abs(lateral_error) > 0.12 or abs(heading_error) > 0.45:
        v_cmd *= 0.75

    # atan2 daje gladne, ograniczone wzmocnienie bledu bocznego.
    # Znaki sa zgodne z lokalnym ukladem robota: punkt sciezki po lewej
    # stronie daje dodatnie omega, czyli skret w lewo.
    omega_cmd = reference.omega
    omega_cmd += 2.2 * heading_error
    omega_cmd += 3.0 * atan2(lateral_error, 0.35)

    max_speed = min(0.45, max(0.24, 1.10 * cruise_speed))
    v_cmd = _clamp(v_cmd, 0.0, max_speed)
    omega_cmd = _clamp(omega_cmd, -2.4, 2.4)
    target_platform = (omega_cmd, v_cmd)

    target_wheels = wheel_speeds(target_platform, params)
    measured_wheels = wheel_speeds((state[3], state[4]), params)
    steady_voltage = _steady_voltage_for_platform(target_platform, params)

    speed_kp = 2.2
    voltage_right = steady_voltage[0] + speed_kp * (target_wheels[0] - measured_wheels[0])
    voltage_left = steady_voltage[1] + speed_kp * (target_wheels[1] - measured_wheels[1])

    voltage_right = _clamp(voltage_right, -voltage_limit, voltage_limit)
    voltage_left = _clamp(voltage_left, -voltage_limit, voltage_limit)
    return voltage_right, voltage_left
