from dataclasses import dataclass
from math import cos, sin, tanh
from typing import Callable, List, Optional, Tuple, Union


State = Tuple[float, float, float, float, float]
Vector2 = Tuple[float, float]
VoltageInput = Union[Vector2, Callable[[float, State], Vector2]]


@dataclass(frozen=True)
class RobotParams:
    # Parametry mechaniczne platformy mobilnej.
    m: float = 24.0
    b: float = 0.15
    r: float = 0.0845
    xi: float = 0.1
    Ik: float = 1.014e-5

    # Parametry silnikow DC i przekladni.
    Im: float = 4.22e-3 * 1e-4
    xi_m: float = 1.85e-8
    ng: float = 1.0
    R: float = 3.78
    km: float = 0.855

    # Parametry oporow ruchu.
    g: float = 9.81
    Cr: float = 5e-4
    Daw: float = 5e-4
    Dav: float = 5e-3
    nu: float = 10.0

    @property
    def Ic(self) -> float:
        # Moment bezwladnosci platformy wzgledem osi pionowej.
        B = 2.0 * self.b
        return self.m * (B**2 + B**2) / 12.0


def wheel_speeds(u: Vector2, p: RobotParams) -> Vector2:
    # Zamiana predkosci platformy na predkosci katowe kol.
    # u = [omega, v], wynik = [omega_prawego_kola, omega_lewego_kola].
    omega, v = u
    return (p.b / p.r * omega + v / p.r, -p.b / p.r * omega + v / p.r)


def motor_shaft_speeds(omega_wheels: Vector2, p: RobotParams) -> Vector2:
    # Predkosci walow silnikow po uwzglednieniu przekladni.
    omega_p, omega_l = omega_wheels
    return omega_p / p.ng, omega_l / p.ng


def motor_currents(u_me: Vector2, p: RobotParams) -> Vector2:
    # Prawo Ohma dla uzwojen silnikow: i = U / R.
    # u_me to napiecie skuteczne po odjeciu sily elektromotorycznej.
    return u_me[0] / p.R, u_me[1] / p.R


def motor_torques(i_m: Vector2, p: RobotParams) -> Vector2:
    # Zamiana pradu silnika na moment obrotowy: Tm = km * i.
    return p.km * i_m[0], p.km * i_m[1]


def back_emf(omega_m: Vector2, p: RobotParams) -> Vector2:
    # Sila elektromotoryczna silnika, przeciwna do napiecia zasilania.
    return p.km * omega_m[0], p.km * omega_m[1]


def b_tilde_times_torque(t_m: Vector2, p: RobotParams) -> Vector2:
    # Przeliczenie momentow kol na wymuszenie ruchu platformy.
    # Wynik ma skladowe: [moment obrotowy platformy, sila postepowa].
    t_p, t_l = t_m
    wsp = 1.0 / (p.r * p.ng)
    return p.b * wsp * t_p - p.b * wsp * t_l, wsp * t_p + wsp * t_l


def h_tilde_times_u(u: Vector2, p: RobotParams) -> Vector2:
    # Czlon tlumienia lepko-suchego z lozysk kol i silnikow.
    # Dziala przeciwnie do aktualnych predkosci platformy.
    omega, v = u
    h11 = (2.0 * p.b**2 / p.r**2) * p.xi + (2.0 * p.b**2 / (p.r**2 * p.ng**2)) * p.xi_m
    h22 = (2.0 / p.r**2) * p.xi + (2.0 / (p.r**2 * p.ng**2)) * p.xi_m
    return h11 * omega, h22 * v


def resistance_forces(u: Vector2, p: RobotParams) -> Vector2:
    # Opory ruchu: opor aerodynamiczny i opor toczenia.
    # Wynik = [opor obrotu, opor ruchu postepowego].
    omega, v = u

    # Przyblizone predkosci liniowe prawego i lewego kola.
    omega_p = v + p.b * omega
    omega_l = v - p.b * omega

    # Opory aerodynamiczne zalezne od predkosci.
    n_a = p.Daw * abs(omega) * omega
    f_a = p.Dav * abs(v) * v

    # Opory toczenia wygladzone funkcja tanh, zeby uniknac skoku w zerze.
    rw_p = p.r * omega_p
    rw_l = p.r * omega_l
    n_r = (p.b / 2.0) * p.m * p.g * p.Cr * (tanh(p.nu * rw_p) - tanh(p.nu * rw_l))
    f_r = (1.0 / 2.0) * p.m * p.g * p.Cr * (tanh(p.nu * rw_p) + tanh(p.nu * rw_l))

    return n_a + n_r, f_a + f_r


def m_tilde_inverse_diagonal(p: RobotParams) -> Vector2:
    # Odwrotnosc diagonalnej macierzy bezwladnosci M_tilde.
    # Zwracamy tylko przekatna, bo poza przekatna sa zera.
    m11 = p.Ic + (2.0 * p.b**2 / p.r**2) * p.Ik + (2.0 * p.b**2 / (p.r**2 * p.ng**2)) * p.Im
    m22 = p.m + (2.0 / p.r**2) * p.Ik + (2.0 / (p.r**2 * p.ng**2)) * p.Im
    return 1.0 / m11, 1.0 / m22


def derivatives(
    _t: float,
    state: State,
    voltage: Vector2,
    p: RobotParams,
) -> State:
    # Glowna funkcja modelu ciaglego.
    # Przyjmuje aktualny stan i napiecia silnikow, a zwraca pochodne stanu.
    # Stan: [theta, x, y, omega, v].
    # Napiecie: [U_p, U_l], czyli prawe i lewe kolo.
    theta, _x, _y, omega, v = state
    u = (omega, v)

    # Blok elektryczno-mechaniczny silnikow.
    omega_wheels = wheel_speeds(u, p)
    omega_m = motor_shaft_speeds(omega_wheels, p)
    e_m = back_emf(omega_m, p)
    u_me = voltage[0] - e_m[0], voltage[1] - e_m[1]
    i_m = motor_currents(u_me, p)
    t_m = motor_torques(i_m, p)

    # Blok dynamiczny platformy.
    btm = b_tilde_times_torque(t_m, p)
    hu = h_tilde_times_u(u, p)
    forces = resistance_forces(u, p)
    inv_m11, inv_m22 = m_tilde_inverse_diagonal(p)

    # Kinematyka robota roznicowego.
    dtheta = omega
    dx = v * cos(theta)
    dy = v * sin(theta)

    # Dynamika predkosci obrotowej i liniowej.
    domega = inv_m11 * (btm[0] - hu[0] - forces[0])
    dv = inv_m22 * (btm[1] - hu[1] - forces[1])

    return dtheta, dx, dy, domega, dv


def rk4_step(
    t: float,
    state: State,
    dt: float,
    voltage: Vector2,
    p: RobotParams,
) -> State:
    # Jeden krok calkowania metoda Rungego-Kutty 4. rzedu.
    # Jest dokladniejsza niz zwykly Euler przy tym samym kroku dt.
    def add_scaled(base, delta, scale):
        return tuple(base[i] + scale * delta[i] for i in range(len(base)))

    k1 = derivatives(t, state, voltage, p)
    k2 = derivatives(t + dt / 2.0, add_scaled(state, k1, dt / 2.0), voltage, p)
    k3 = derivatives(t + dt / 2.0, add_scaled(state, k2, dt / 2.0), voltage, p)
    k4 = derivatives(t + dt, add_scaled(state, k3, dt), voltage, p)

    return tuple(state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0 for i in range(len(state)))


def simulate(
    duration: float,
    dt: float,
    voltage: VoltageInput,
    initial_state: State = (0.0, 0.0, 0.0, 0.0, 0.0),
    params: Optional[RobotParams] = None,
) -> List[Tuple[float, State]]:
    # Petla symulacji.
    # voltage moze byc:
    # - stala krotka (U_p, U_l),
    # - funkcja sterowania voltage(t, state), ktora zwraca (U_p, U_l).
    p = params or RobotParams()
    t = 0.0
    state = initial_state
    history = [(t, state)]

    steps = int(duration / dt)
    for _ in range(steps):
        u_m = voltage(t, state) if callable(voltage) else voltage
        state = rk4_step(t, state, dt, u_m, p)
        t += dt
        history.append((t, state))

    return history
