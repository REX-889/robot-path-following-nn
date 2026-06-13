import argparse

from direct_path_nn import DirectNNVoltageController, feasible_progress_for_duration
from road_path import make_closed_spline_path
from simulate_path_nn import simulate_direct_path_following


def travelled_progress(log, path):
    # Szacowany postep do przodu po torze, z obsluga zawiniecia petli.
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


def summarize(log, path, cruise_speed: float):
    mean_distance = sum(sample.nearest_distance for sample in log) / len(log)
    max_distance = max(sample.nearest_distance for sample in log)
    final = log[-1]
    u_max = max(max(abs(sample.voltage[0]), abs(sample.voltage[1])) for sample in log)
    expected_progress = max(1e-9, feasible_progress_for_duration(path, final.t, cruise_speed))
    travelled = travelled_progress(log, path)
    progress_ratio = travelled / expected_progress
    mean_speed = travelled / max(1e-9, final.t)
    return mean_distance, max_distance, progress_ratio, mean_speed, u_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="direct_path_nn.pt")
    parser.add_argument("--cruise-speed", type=float, default=0.16)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--first-seed", type=int, default=21)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    controller = DirectNNVoltageController(model_path=args.model, cruise_speed=args.cruise_speed)
    for seed in range(args.first_seed, args.first_seed + args.count):
        path = make_closed_spline_path(seed=seed)
        log = simulate_direct_path_following(
            path=path,
            controller=controller,
            duration=args.duration,
            stop_at_end=False,
        )
        mean_distance, max_distance, progress, mean_speed, u_max = summarize(log, path, args.cruise_speed)
        print(
            f"seed {seed:02d} | "
            f"sredni blad: {mean_distance:.4f} m | "
            f"max blad: {max_distance:.4f} m | "
            f"postep: {progress:.2f} | "
            f"speed: {mean_speed:.3f} m/s | "
            f"Umax: {u_max:.1f} V"
        )


if __name__ == "__main__":
    main()
