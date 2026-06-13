import argparse
from math import cos, sin

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Button

from direct_path_nn import DirectNNVoltageController
from road_path import RoadPath, make_closed_spline_path
from robot_model import RobotParams, rk4_step
from simulate_path_nn import initial_state_for_path


DEFAULT_ANIMATION_SPEEDUP = 2.0
DEFAULT_CRUISE_SPEED = 0.55
DEFAULT_INTERVAL_MS = 20


def transform_path_to_pose(path: RoadPath, x_target: float, y_target: float, theta_target: float) -> RoadPath:
    # Przesuniecie i obrot nowej sciezki tak, zeby jej poczatek byl w pozycji robota.
    # Dzieki temu po zmianie toru robot nie dostaje naglego skoku startu trasy.
    start = path.sample(0.0)
    angle = theta_target - start.theta
    ca = cos(angle)
    sa = sin(angle)

    unique_points = path.points[:-1] if path.closed else path.points
    transformed = []
    for x, y in unique_points:
        dx = x - start.x
        dy = y - start.y
        transformed.append((x_target + ca * dx - sa * dy, y_target + sa * dx + ca * dy))

    return RoadPath(transformed, closed=path.closed)


def unwrapped_forward_delta(current_s: float, previous_s: float, path_length: float) -> float:
    # Postep po zamknietej sciezce z obsluga przeskoku z konca petli na poczatek.
    delta = current_s - previous_s
    if delta < -0.5 * path_length:
        delta += path_length
    elif delta > 0.5 * path_length:
        delta -= path_length
    return max(0.0, delta)


class RealtimePathAnimation:
    # Animacja robota sledzacego zamkniety spline.
    # Po wykonaniu jednego okrazenia generowany jest nowy spline.
    def __init__(
        self,
        model_path: str,
        seed: int,
        dt: float,
        steps_per_frame: int,
        trail_limit: int,
        lap_fraction: float,
        cruise_speed: float,
    ):
        self.params = RobotParams()
        self.controller = DirectNNVoltageController(model_path=model_path, cruise_speed=cruise_speed)
        self.seed = seed
        self.dt = dt
        self.steps_per_frame = steps_per_frame
        self.trail_limit = trail_limit
        self.lap_fraction = lap_fraction
        self.cruise_speed = cruise_speed

        self.t = 0.0
        self.lap_index = 1
        self.path = make_closed_spline_path(seed=self.seed)
        self.state = initial_state_for_path(self.path)
        self.progress_s = 0.0
        self.previous_s = 0.0
        self.lap_progress = 0.0
        self.nearest_distance = 0.0
        self.voltage = (0.0, 0.0)
        self.running = False

        self.trail_x = [self.state[1]]
        self.trail_y = [self.state[2]]
        self.error_times = []
        self.error_values = []
        self.voltage_times = []
        self.voltage_right_values = []
        self.voltage_left_values = []

        self.fig, (self.ax_path, self.ax_error, self.ax_voltage) = plt.subplots(
            1,
            3,
            figsize=(16, 5),
            gridspec_kw={"width_ratios": [2.0, 1.0, 1.0]},
        )
        self._build_static_plot()
        self._refresh_path_plot()
        self._refresh_robot_plot()

    def _build_static_plot(self) -> None:
        # Glowne okno: tor, slad robota, pozycja robota i kierunek jazdy.
        self.path_line, = self.ax_path.plot([], [], "--", linewidth=2, label="zadana sciezka")
        self.trail_line, = self.ax_path.plot([], [], linewidth=2, label="slad robota")
        self.start_point, = self.ax_path.plot([], [], "o", markersize=6, label="start okrazenia")
        self.robot_body = Circle((self.state[1], self.state[2]), 0.12, fill=False, linewidth=2)
        self.ax_path.add_patch(self.robot_body)
        self.heading_line, = self.ax_path.plot([], [], linewidth=3, label="zwrot robota")
        self.info_text = self.ax_path.text(
            0.02,
            0.98,
            "",
            transform=self.ax_path.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        self.ax_path.set_title("Animacja path following NN")
        self.ax_path.set_xlabel("x [m]")
        self.ax_path.set_ylabel("y [m]")
        self.ax_path.set_aspect("equal", adjustable="box")
        self.ax_path.grid(True)
        self.ax_path.legend(loc="lower right")

        # Wykres bledu od sciezki w czasie rzeczywistym.
        self.error_line, = self.ax_error.plot([], [], linewidth=2)
        self.ax_error.set_title("Blad od sciezki")
        self.ax_error.set_xlabel("t [s]")
        self.ax_error.set_ylabel("m")
        self.ax_error.grid(True)

        # Wykres napiec sterujacych prawym i lewym kolem.
        self.voltage_right_line, = self.ax_voltage.plot([], [], linewidth=2, label="U_p")
        self.voltage_left_line, = self.ax_voltage.plot([], [], linewidth=2, label="U_l")
        self.ax_voltage.set_title("Napiecia kol")
        self.ax_voltage.set_xlabel("t [s]")
        self.ax_voltage.set_ylabel("V")
        self.ax_voltage.set_ylim(-12.5, 12.5)
        self.ax_voltage.grid(True)
        self.ax_voltage.legend()

        self.fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
        self.start_button_axis = self.fig.add_axes([0.43, 0.03, 0.14, 0.06])
        self.start_button = Button(self.start_button_axis, "Start")
        self.start_button.on_clicked(self.start_simulation)

    def start_simulation(self, _event) -> None:
        # Przycisk sluzy tylko do startu. Po uruchomieniu znika z okna.
        self.running = True
        self.start_button_axis.set_visible(False)
        self._refresh_robot_plot()
        self.fig.canvas.draw_idle()

    def _refresh_path_plot(self) -> None:
        # Aktualizacja linii toru po wygenerowaniu nowej trasy.
        path_x = [p[0] for p in self.path.points]
        path_y = [p[1] for p in self.path.points]
        self.path_line.set_data(path_x, path_y)
        self.start_point.set_data([path_x[0]], [path_y[0]])

        margin = 0.6
        self.ax_path.set_xlim(min(path_x) - margin, max(path_x) + margin)
        self.ax_path.set_ylim(min(path_y) - margin, max(path_y) + margin)

    def _change_path_after_lap(self) -> None:
        # Nowa trasa po jednym okrazeniu.
        # Resetujemy pamiec GRU, bo zmienil sie kontekst sciezki.
        self.seed += 1
        self.lap_index += 1
        theta, x, y, _omega, _v = self.state
        raw_path = make_closed_spline_path(seed=self.seed)
        self.path = transform_path_to_pose(raw_path, x, y, theta)
        self.controller.reset()

        self.progress_s = 0.0
        self.previous_s = 0.0
        self.lap_progress = 0.0
        self.trail_x = [x]
        self.trail_y = [y]
        self._refresh_path_plot()

    def _simulate_one_step(self) -> None:
        # Jeden krok symulacji: pomiar bledu, sterowanie NN, calkowanie RK4.
        self.progress_s, self.nearest_distance = self.path.nearest_s(
            self.state[1],
            self.state[2],
            self.progress_s,
        )
        self.lap_progress += unwrapped_forward_delta(self.progress_s, self.previous_s, self.path.total_length)
        self.previous_s = self.progress_s

        remaining = self.path.total_length
        self.voltage = self.controller.compute_voltage(self.state, self.path, self.progress_s, remaining)
        self.state = rk4_step(self.t, self.state, self.dt, self.voltage, self.params)
        self.t += self.dt

        if self.lap_progress >= self.lap_fraction * self.path.total_length:
            self._change_path_after_lap()

    def _append_live_history(self) -> None:
        # Zapis danych do rysowania, ograniczony do ostatnich probek.
        self.trail_x.append(self.state[1])
        self.trail_y.append(self.state[2])
        self.trail_x = self.trail_x[-self.trail_limit :]
        self.trail_y = self.trail_y[-self.trail_limit :]

        self.error_times.append(self.t)
        self.error_values.append(self.nearest_distance)
        self.voltage_times.append(self.t)
        self.voltage_right_values.append(self.voltage[0])
        self.voltage_left_values.append(self.voltage[1])

        self.error_times = self.error_times[-self.trail_limit :]
        self.error_values = self.error_values[-self.trail_limit :]
        self.voltage_times = self.voltage_times[-self.trail_limit :]
        self.voltage_right_values = self.voltage_right_values[-self.trail_limit :]
        self.voltage_left_values = self.voltage_left_values[-self.trail_limit :]

    def _refresh_robot_plot(self) -> None:
        # Aktualizacja pozycji robota, zwrotu, sladu i opisow.
        theta, x, y, _omega, v = self.state
        self.robot_body.center = (x, y)
        self.heading_line.set_data([x, x + 0.28 * cos(theta)], [y, y + 0.28 * sin(theta)])
        self.trail_line.set_data(self.trail_x, self.trail_y)

        lap_percent = 100.0 * self.lap_progress / max(1e-9, self.path.total_length)
        self.info_text.set_text(
            f"status: {'START' if self.running else 'PAUZA'}\n"
            f"czas: {self.t:6.2f} s\n"
            f"okraz.: {self.lap_index}\n"
            f"seed trasy: {self.seed}\n"
            f"v doc.: {self.cruise_speed:5.3f} m/s\n"
            f"postep: {lap_percent:5.1f} %\n"
            f"blad: {self.nearest_distance * 100.0:5.2f} cm\n"
            f"v: {v:5.3f} m/s\n"
            f"U_p: {self.voltage[0]:5.2f} V\n"
            f"U_l: {self.voltage[1]:5.2f} V"
        )

    def _refresh_history_plots(self) -> None:
        # Aktualizacja wykresow bledu i napiec.
        self.error_line.set_data(self.error_times, self.error_values)
        self.voltage_right_line.set_data(self.voltage_times, self.voltage_right_values)
        self.voltage_left_line.set_data(self.voltage_times, self.voltage_left_values)

        if self.error_times:
            t_min = max(0.0, self.error_times[-1] - 30.0)
            t_max = self.error_times[-1] + 0.5
            self.ax_error.set_xlim(t_min, t_max)
            self.ax_voltage.set_xlim(t_min, t_max)

        if self.error_values:
            max_error = max(0.02, max(self.error_values) * 1.15)
            self.ax_error.set_ylim(0.0, max_error)

    def update(self, _frame):
        # Funkcja wywolywana przez Matplotlib co klatke animacji.
        if self.running:
            for _ in range(self.steps_per_frame):
                self._simulate_one_step()
            self._append_live_history()

        self._refresh_robot_plot()
        self._refresh_history_plots()

        return (
            self.path_line,
            self.trail_line,
            self.start_point,
            self.robot_body,
            self.heading_line,
            self.info_text,
            self.error_line,
            self.voltage_right_line,
            self.voltage_left_line,
        )

    def run(self, interval_ms: int) -> None:
        # Uruchomienie okna animacji.
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="direct_path_nn.pt")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps-per-frame", type=int, default=0, help="0 = wylicz automatycznie z --speedup.")
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument(
        "--speedup",
        type=float,
        default=DEFAULT_ANIMATION_SPEEDUP,
        help="Przyspieszenie animacji wzgledem czasu rzeczywistego; np. 5 = okolo 5x szybciej.",
    )
    parser.add_argument("--cruise-speed", type=float, default=DEFAULT_CRUISE_SPEED, help="Docelowa predkosc jazdy robota [m/s].")
    parser.add_argument("--trail-limit", type=int, default=2500)
    parser.add_argument(
        "--lap-fraction",
        type=float,
        default=1.0,
        help="1.0 oznacza zmiane trasy po pelnym okrazeniu; np. 0.25 tylko do szybkiego testu.",
    )
    args = parser.parse_args()

    if args.steps_per_frame > 0:
        steps_per_frame = args.steps_per_frame
    else:
        simulated_time_per_frame = args.interval_ms / 1000.0 * args.speedup
        steps_per_frame = max(1, round(simulated_time_per_frame / args.dt))
    print(
        f"animacja: speedup={args.speedup:.2f}x, "
        f"dt={args.dt:.4f} s, "
        f"steps_per_frame={steps_per_frame}, "
        f"interval={args.interval_ms} ms, "
        f"cruise_speed={args.cruise_speed:.3f} m/s"
    )

    animation = RealtimePathAnimation(
        model_path=args.model,
        seed=args.seed,
        dt=args.dt,
        steps_per_frame=steps_per_frame,
        trail_limit=args.trail_limit,
        lap_fraction=args.lap_fraction,
        cruise_speed=args.cruise_speed,
    )
    animation.run(interval_ms=args.interval_ms)


if __name__ == "__main__":
    main()
