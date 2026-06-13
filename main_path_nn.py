from math import sqrt

from direct_path_nn import DirectNNVoltageController, desired_speed_from_remaining
from plots_direct_path import plot_direct_path
from road_path import DEFAULT_SPLINE_PATH
from simulate_path_nn import simulate_direct_path_following


def travelled_progress(log, path):
    # Szacowany postep do przodu po torze, z obsluga zawiniecia zamknietej petli.
    if len(log) < 2:
        return 0.0

    total = 0.0
    previous_s = log[0].progress_s
    path_length = path.total_length

    for sample in log[1:]:
        current_s = sample.progress_s
        delta = current_s - previous_s
        if path.closed:
            if delta < -0.5 * path_length:
                delta += path_length
            elif delta > 0.5 * path_length:
                delta -= path_length
        total += max(0.0, delta)
        previous_s = current_s

    return total


def main():
    # Docelowa galaz: path following po zamknietym splajnie bez PI/PIC.
    # GRU dostaje sciezke i stan robota, a zwraca bezposrednio napiecia kol.
    controller = DirectNNVoltageController(model_path="direct_path_nn.pt")
    log = simulate_direct_path_following(
        path=DEFAULT_SPLINE_PATH,
        controller=controller,
        duration=100.0,
        stop_at_end=False,
    )

    last = log[-1]
    mean_distance = sum(sample.nearest_distance for sample in log) / len(log)
    max_distance = max(sample.nearest_distance for sample in log)
    expected_progress = max(1e-9, desired_speed_from_remaining(DEFAULT_SPLINE_PATH.total_length) * last.t)
    progress_ratio = travelled_progress(log, DEFAULT_SPLINE_PATH) / expected_progress

    print("PATH FOLLOWING: GRU NN -> U_p, U_l")
    print(f"czas = {last.t:.2f} s")
    print(f"postep po splajnie = {last.progress_s:.3f} / {DEFAULT_SPLINE_PATH.total_length:.3f} m")
    print(f"postep wzgledem oczekiwanego = {progress_ratio:.2f}")
    print(f"sredni blad od sciezki = {mean_distance:.4f} m")
    print(f"max blad od sciezki = {max_distance:.4f} m")
    print(f"aktualny blad pozycji = {sqrt(last.global_errors[0] ** 2 + last.global_errors[1] ** 2):.4f} m")
    print(f"blad theta = {last.global_errors[2]:.4f} rad")

    plot_direct_path(log, DEFAULT_SPLINE_PATH)


if __name__ == "__main__":
    main()
