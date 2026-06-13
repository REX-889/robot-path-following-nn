from math import cos, sin, sqrt

import matplotlib.pyplot as plt


def plot_direct_path(log, path):
    times = [sample.t for sample in log]
    theta = [sample.state[0] for sample in log]
    x = [sample.state[1] for sample in log]
    y = [sample.state[2] for sample in log]
    v = [sample.state[4] for sample in log]
    voltage_right = [sample.voltage[0] for sample in log]
    voltage_left = [sample.voltage[1] for sample in log]
    nearest_distance = [sample.nearest_distance for sample in log]
    remaining = [sample.remaining for sample in log]
    error_x = [sample.global_errors[0] for sample in log]
    error_y = [sample.global_errors[1] for sample in log]
    error_theta = [sample.global_errors[2] for sample in log]

    path_x = [p[0] for p in path.points]
    path_y = [p[1] for p in path.points]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_path, ax_voltage, ax_distance = axes[0]
    ax_errors, ax_speed, ax_progress = axes[1]

    ax_path.plot(path_x, path_y, "--", linewidth=2, label="zadana sciezka")
    ax_path.plot(x, y, linewidth=2, label="robot NN")
    ax_path.scatter(path_x[0], path_y[0], color="green", label="start", zorder=3)
    ax_path.scatter(path_x[-1], path_y[-1], color="red", label="meta", zorder=3)

    step = max(1, len(log) // 14)
    for i in range(0, len(log), step):
        ax_path.arrow(
            x[i],
            y[i],
            0.08 * cos(theta[i]),
            0.08 * sin(theta[i]),
            head_width=0.025,
            head_length=0.035,
            length_includes_head=True,
            color="black",
            alpha=0.65,
        )

    ax_path.set_title("Path following NN")
    ax_path.set_xlabel("x [m]")
    ax_path.set_ylabel("y [m]")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True)
    ax_path.legend()

    ax_voltage.plot(times, voltage_right, label="U_p prawe")
    ax_voltage.plot(times, voltage_left, label="U_l lewe")
    ax_voltage.set_title("Napiecia NN")
    ax_voltage.set_xlabel("t [s]")
    ax_voltage.set_ylabel("U [V]")
    ax_voltage.grid(True)
    ax_voltage.legend()

    ax_distance.plot(times, nearest_distance, label="odleglosc od sciezki")
    ax_distance.set_title("Blad geometryczny")
    ax_distance.set_xlabel("t [s]")
    ax_distance.set_ylabel("m")
    ax_distance.grid(True)
    ax_distance.legend()

    ax_errors.plot(times, error_x, label="x_ref - x")
    ax_errors.plot(times, error_y, label="y_ref - y")
    ax_errors.plot(times, error_theta, label="theta_ref - theta")
    ax_errors.set_title("Bledy x/y/theta")
    ax_errors.set_xlabel("t [s]")
    ax_errors.grid(True)
    ax_errors.legend()

    ax_speed.plot(times, v, label="v robota")
    ax_speed.set_title("Predkosc liniowa")
    ax_speed.set_xlabel("t [s]")
    ax_speed.set_ylabel("m/s")
    ax_speed.grid(True)
    ax_speed.legend()

    if path.closed:
        ax_progress.plot(times, [sample.progress_s for sample in log], label="s na splajnie")
        ax_progress.set_title("Pozycja na zamknietym splajnie")
    else:
        ax_progress.plot(times, [path.total_length - r for r in remaining], label="przejechane s")
        ax_progress.plot(times, remaining, label="pozostalo")
        ax_progress.set_title("Postep po sciezce")
    ax_progress.set_xlabel("t [s]")
    ax_progress.set_ylabel("m")
    ax_progress.grid(True)
    ax_progress.legend()

    fig.tight_layout()
    plt.show()
