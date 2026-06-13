import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from math import cos, sin
from pathlib import Path
from random import Random
import time

import matplotlib.pyplot as plt
import torch
from torch import nn

from controller import wrap_to_pi
from direct_path_nn import DirectVoltagePathGRU, build_direct_path_features, feasible_progress_for_duration
from expert_voltage import expert_voltage_for_path
from road_path import make_closed_spline_path
from robot_model import RobotParams, rk4_step
from simulate_path_nn import simulate_direct_path_following


def available_cpu_threads() -> int:
    # Liczba logicznych watkow CPU widoczna dla Pythona.
    return max(1, os.cpu_count() or 1)


def resolve_thread_count(requested_threads: int) -> int:
    # 0 lub liczba ujemna oznacza: uzyj wszystkich dostepnych watkow.
    cpu_threads = available_cpu_threads()
    if requested_threads <= 0:
        return cpu_threads
    return max(1, min(requested_threads, cpu_threads))


def parse_speed_list(text: str | None, fallback_speed: float) -> list[float]:
    # Lista predkosci walidacyjnych, np. "0.16,0.35,0.55".
    if not text:
        return [fallback_speed]

    speeds = []
    for item in text.split(","):
        item = item.strip()
        if item:
            speeds.append(float(item))

    return speeds or [fallback_speed]


def choose_cruise_speed(
    rng: Random,
    min_cruise_speed: float,
    max_cruise_speed: float,
    slow_replay_fraction: float,
    slow_cruise_speed_min: float,
    slow_cruise_speed_max: float,
) -> float:
    # Czasem celowo losujemy wolna jazde, zeby model szybki nie zapomnial
    # zachowania przy malych predkosciach.
    if slow_replay_fraction > 0.0 and rng.random() < slow_replay_fraction:
        low = min(slow_cruise_speed_min, slow_cruise_speed_max)
        high = max(slow_cruise_speed_min, slow_cruise_speed_max)
        return rng.uniform(low, high)

    return rng.uniform(min_cruise_speed, max_cruise_speed)


def random_state_near_path(path, progress_s: float, rng: Random, max_speed: float):
    # Stan robota w okolicy splajna: pozycja, orientacja i predkosci sa zaburzone.
    # Wiekszosc probek jest blisko toru, bo tam chcemy maksymalnej precyzji.
    # Czesc probek ma wieksze odchylenia, zeby siec umiala wrocic na trase.
    pose = path.sample(progress_s)
    if rng.random() < 0.70:
        lateral_error = rng.uniform(-0.08, 0.08)
        longitudinal_error = rng.uniform(-0.05, 0.05)
        heading_error = rng.uniform(-0.22, 0.22)
        omega = rng.uniform(-0.18, 0.18)
        v = rng.uniform(0.02, min(0.45, 1.15 * max_speed))
    else:
        lateral_error = rng.uniform(-0.28, 0.28)
        longitudinal_error = rng.uniform(-0.14, 0.14)
        heading_error = rng.uniform(-0.70, 0.70)
        omega = rng.uniform(-0.50, 0.50)
        v = rng.uniform(0.0, min(0.50, 1.25 * max_speed))

    x = pose.x + longitudinal_error * cos(pose.theta)
    x += lateral_error * cos(pose.theta + 1.57079632679)
    y = pose.y + longitudinal_error * sin(pose.theta)
    y += lateral_error * sin(pose.theta + 1.57079632679)

    theta = wrap_to_pi(pose.theta + heading_error)
    return theta, x, y, omega, v


def _make_sequence_chunk(
    start_index: int,
    sequence_count: int,
    sequence_length: int,
    seed: int,
    voltage_limit: float,
    min_cruise_speed: float,
    max_cruise_speed: float,
    slow_replay_fraction: float,
    slow_cruise_speed_min: float,
    slow_cruise_speed_max: float,
):
    # Fragment datasetu generowany przez jeden proces roboczy.
    rng = Random(seed + 1000003 * start_index)
    params = RobotParams()
    dt = 0.02

    all_features = []
    all_targets = []

    for local_i in range(sequence_count):
        i = start_index + local_i
        path = make_closed_spline_path(seed=3000 + i % 1000)
        cruise_speed = choose_cruise_speed(
            rng,
            min_cruise_speed,
            max_cruise_speed,
            slow_replay_fraction,
            slow_cruise_speed_min,
            slow_cruise_speed_max,
        )
        progress_s = rng.uniform(0.0, path.total_length)
        state = random_state_near_path(path, progress_s, rng, max_cruise_speed)

        features = []
        targets = []
        t = 0.0

        for _ in range(sequence_length):
            progress_s, _nearest_distance = path.nearest_s(state[1], state[2], progress_s)
            remaining = path.total_length
            voltage = expert_voltage_for_path(
                state,
                path,
                progress_s,
                voltage_limit=voltage_limit,
                cruise_speed=cruise_speed,
            )

            features.append(build_direct_path_features(state, path, progress_s, remaining, cruise_speed))
            targets.append([voltage[0] / voltage_limit, voltage[1] / voltage_limit])

            state = rk4_step(t, state, dt, voltage, params)
            t += dt

        all_features.append(features)
        all_targets.append(targets)

    return all_features, all_targets


def make_sequence_dataset(
    sample_count: int,
    sequence_length: int,
    seed: int,
    voltage_limit: float,
    workers: int,
    min_cruise_speed: float,
    max_cruise_speed: float,
    slow_replay_fraction: float,
    slow_cruise_speed_min: float,
    slow_cruise_speed_max: float,
):
    # Dataset sekwencyjny dla GRU.
    # Kazda sekwencja to krotki fragment jazdy eksperta po zamknietym splajnie.
    sequence_count = max(1, sample_count // sequence_length)
    workers = max(1, min(workers, sequence_count))

    if workers == 1:
        all_features, all_targets = _make_sequence_chunk(
            0,
            sequence_count,
            sequence_length,
            seed,
            voltage_limit,
            min_cruise_speed,
            max_cruise_speed,
            slow_replay_fraction,
            slow_cruise_speed_min,
            slow_cruise_speed_max,
        )
        return torch.tensor(all_features, dtype=torch.float32), torch.tensor(all_targets, dtype=torch.float32)

    chunk_sizes = []
    base = sequence_count // workers
    rest = sequence_count % workers
    start = 0
    for worker_id in range(workers):
        count = base + (1 if worker_id < rest else 0)
        chunk_sizes.append((start, count))
        start += count

    all_features = []
    all_targets = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _make_sequence_chunk,
                start_index,
                count,
                sequence_length,
                seed,
                voltage_limit,
                min_cruise_speed,
                max_cruise_speed,
                slow_replay_fraction,
                slow_cruise_speed_min,
                slow_cruise_speed_max,
            )
            for start_index, count in chunk_sizes
            if count > 0
        ]

        for future in as_completed(futures):
            features, targets = future.result()
            all_features.extend(features)
            all_targets.extend(targets)

    return torch.tensor(all_features, dtype=torch.float32), torch.tensor(all_targets, dtype=torch.float32)


def save_training_history(history, csv_path: str, plot_path: str) -> None:
    # Zapis historii uczenia do CSV i wykresu PNG.
    csv_target = Path(csv_path)
    with csv_target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "epoch_seconds"])
        writer.writerows(history)

    epochs = [row[0] for row in history]
    losses = [row[1] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoka")
    plt.ylabel("Loss MSE")
    plt.title("Proces uczenia GRU")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


class LiveTrainingPlot:
    # Wykres loss i bledu sledzenia aktualizowany w trakcie treningu.
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.fig = None
        self.ax_loss = None
        self.ax_error = None
        self.loss_line = None
        self.mean_error_line = None
        self.max_error_line = None
        self.progress_line = None

        if not enabled:
            return

        plt.ion()
        self.fig, (self.ax_loss, self.ax_error, self.ax_progress) = plt.subplots(1, 3, figsize=(16, 5))

        self.loss_line, = self.ax_loss.plot([], [], linewidth=2, label="loss")
        self.ax_loss.set_yscale("log")
        self.ax_loss.set_xlabel("Epoka")
        self.ax_loss.set_ylabel("Loss MSE")
        self.ax_loss.set_title("Uczenie GRU")
        self.ax_loss.grid(True)
        self.ax_loss.legend()

        self.mean_error_line, = self.ax_error.plot([], [], linewidth=2, label="sredni blad")
        self.max_error_line, = self.ax_error.plot([], [], linewidth=2, label="max blad")
        self.ax_error.set_xlabel("Epoka")
        self.ax_error.set_ylabel("Blad od splajna [m]")
        self.ax_error.set_title("Walidacja sledzenia")
        self.ax_error.grid(True)
        self.ax_error.legend()

        self.progress_line, = self.ax_progress.plot([], [], linewidth=2, label="postep")
        self.ax_progress.axhline(1.0, linestyle="--", linewidth=1, color="gray", label="cel")
        self.ax_progress.set_xlabel("Epoka")
        self.ax_progress.set_ylabel("Postep / oczekiwany")
        self.ax_progress.set_title("Postep po torze")
        self.ax_progress.set_ylim(0.0, 1.25)
        self.ax_progress.grid(True)
        self.ax_progress.legend()

        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, history, error_history) -> None:
        if not self.enabled or not history:
            return

        epochs = [row[0] for row in history]
        losses = [row[1] for row in history]
        self.loss_line.set_data(epochs, losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        if error_history:
            error_epochs = [row[0] for row in error_history]
            mean_errors = [row[1] for row in error_history]
            max_errors = [row[2] for row in error_history]
            progress_ratios = [row[3] for row in error_history]
            self.mean_error_line.set_data(error_epochs, mean_errors)
            self.max_error_line.set_data(error_epochs, max_errors)
            self.ax_error.relim()
            self.ax_error.autoscale_view()

            self.progress_line.set_data(error_epochs, progress_ratios)
            self.ax_progress.relim()
            self.ax_progress.autoscale_view()
            self.ax_progress.set_ylim(0.0, max(1.25, max(progress_ratios) * 1.10))

        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def finish(self, keep_open: bool) -> None:
        if not self.enabled:
            return

        plt.ioff()
        if keep_open:
            plt.show()


class InMemoryDirectController:
    # Kontroler walidacyjny uzywajacy aktualnych wag modelu bez zapisu na dysk.
    def __init__(self, model, voltage_limit: float, cruise_speed: float):
        self.model = model
        self.voltage_limit = voltage_limit
        self.cruise_speed = cruise_speed
        self.hidden = None

    def reset(self):
        self.hidden = None

    def compute_voltage(self, state, path, progress_s: float, remaining: float):
        features = build_direct_path_features(state, path, progress_s, remaining, self.cruise_speed)
        x = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            raw, self.hidden = self.model.step(x, self.hidden)
            raw = raw.squeeze(0)
        return (
            float(torch.tanh(raw[0]) * self.voltage_limit),
            float(torch.tanh(raw[1]) * self.voltage_limit),
        )


def evaluate_tracking_error(model, voltage_limit: float, path_count: int, duration: float, cruise_speed: float):
    # Walidacja zamknietej petli: GRU -> napiecia -> model robota.
    model.eval()
    mean_errors = []
    max_errors = []
    progress_ratios = []
    mean_speeds = []

    for seed in range(7000, 7000 + path_count):
        path = make_closed_spline_path(seed=seed)
        controller = InMemoryDirectController(model, voltage_limit, cruise_speed)
        log = simulate_direct_path_following(
            path=path,
            controller=controller,
            duration=duration,
            stop_at_end=False,
        )
        distances = [sample.nearest_distance for sample in log]
        mean_errors.append(sum(distances) / len(distances))
        max_errors.append(max(distances))

        travelled = travelled_progress(log, path)
        expected = max(1e-9, feasible_progress_for_duration(path, duration, cruise_speed))
        progress_ratios.append(travelled / expected)
        mean_speeds.append(travelled / max(1e-9, duration))

    model.train()
    return (
        sum(mean_errors) / len(mean_errors),
        max(max_errors),
        sum(progress_ratios) / len(progress_ratios),
        sum(mean_speeds) / len(mean_speeds),
    )


def travelled_progress(log, path) -> float:
    # Szacowany postep do przodu po torze.
    # Dla sciezki zamknietej rozpakowujemy przeskok s z konca petli na poczatek.
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

        # Cofanie nie zwieksza postepu.
        total += max(0.0, delta)
        previous_s = current_s

    return total


def train(args):
    if args.min_cruise_speed <= 0.0 or args.max_cruise_speed <= 0.0:
        raise ValueError("Predkosci treningowe musza byc dodatnie.")
    if args.min_cruise_speed > args.max_cruise_speed:
        raise ValueError("--min-cruise-speed nie moze byc wieksze niz --max-cruise-speed.")
    if not 0.0 <= args.slow_replay_fraction <= 1.0:
        raise ValueError("--slow-replay-fraction musi byc w zakresie 0.0 - 1.0.")

    cpu_threads = available_cpu_threads()
    worker_count = resolve_thread_count(args.workers)
    torch_threads = resolve_thread_count(args.torch_threads)
    torch.set_num_threads(torch_threads)
    eval_cruise_speeds = parse_speed_list(args.eval_cruise_speeds, args.eval_cruise_speed)

    print(f"wykryte logiczne watki CPU: {cpu_threads}")
    print(f"procesy do generowania danych: {worker_count}")
    print(f"watki PyTorch do uczenia: {torch_threads}")
    print(f"generowanie datasetu: workers={worker_count}, samples={args.samples}, sequence_length={args.sequence_length}")
    print(f"zakres predkosci treningowych: {args.min_cruise_speed:.3f} - {args.max_cruise_speed:.3f} m/s")
    print(
        f"slow replay: {args.slow_replay_fraction:.2f} "
        f"w zakresie {args.slow_cruise_speed_min:.3f} - {args.slow_cruise_speed_max:.3f} m/s"
    )
    print("predkosci walidacyjne: " + ", ".join(f"{speed:.3f}" for speed in eval_cruise_speeds) + " m/s")
    dataset_start = time.perf_counter()
    x, y = make_sequence_dataset(
        args.samples,
        args.sequence_length,
        args.seed,
        args.voltage_limit,
        worker_count,
        args.min_cruise_speed,
        args.max_cruise_speed,
        args.slow_replay_fraction,
        args.slow_cruise_speed_min,
        args.slow_cruise_speed_max,
    )
    print(f"dataset: {x.shape[0]} sekwencji, czas = {time.perf_counter() - dataset_start:.2f} s")

    model = DirectVoltagePathGRU()

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"wczytano model do douczenia: {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    history = []
    error_history = []
    best_score = None
    best_state = None
    best_epoch = None
    best_mean_error = None
    best_max_error = None
    best_progress_ratio = None
    best_mean_speed = None
    live_plot = LiveTrainingPlot(enabled=not args.no_live_plot)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        permutation = torch.randperm(x.shape[0])
        total_loss = 0.0

        for start in range(0, x.shape[0], args.batch):
            batch_idx = permutation[start : start + args.batch]
            xb = x[batch_idx]
            yb = y[batch_idx]

            pred = torch.tanh(model(xb))
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * xb.shape[0]

        epoch_loss = total_loss / x.shape[0]
        epoch_seconds = time.perf_counter() - epoch_start
        history.append((epoch, epoch_loss, epoch_seconds))
        message = f"epoch {epoch:03d} | loss = {epoch_loss:.6f} | {epoch_seconds:.2f} s"

        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            eval_results = [
                evaluate_tracking_error(
                    model,
                    args.voltage_limit,
                    args.eval_paths,
                    args.eval_duration,
                    eval_speed,
                )
                for eval_speed in eval_cruise_speeds
            ]
            mean_error = sum(result[0] for result in eval_results) / len(eval_results)
            max_error = max(result[1] for result in eval_results)
            progress_ratio = sum(result[2] for result in eval_results) / len(eval_results)
            mean_speed = sum(result[3] for result in eval_results) / len(eval_results)
            error_history.append((epoch, mean_error, max_error, progress_ratio, mean_speed))
            message += (
                f" | val mean = {mean_error:.4f} m"
                f" | val max = {max_error:.4f} m"
                f" | progress = {progress_ratio:.2f}"
                f" | speed = {mean_speed:.3f} m/s"
            )

            # Wybieramy model pod realne sledzenie sciezki, nie pod sam loss MSE.
            # score karze blad sredni, czesc bledu maksymalnego oraz brak postepu.
            progress_error = max(0.0, 1.0 - progress_ratio)
            score = mean_error + args.max_error_weight * max_error + args.progress_error_weight * progress_error
            if best_score is None or score < best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
                best_mean_error = mean_error
                best_max_error = max_error
                best_progress_ratio = progress_ratio
                best_mean_speed = mean_speed
                message += " | BEST"
                if not args.no_save_best_during_training:
                    torch.save(best_state, args.output)

        print(message)
        live_plot.update(history, error_history)

        if args.save_history and args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == args.epochs):
            save_training_history(history, args.history_csv, args.loss_plot)

    if best_state is not None:
        torch.save(best_state, args.output)
        print(
            f"zapisano najlepszy model: {args.output} | "
            f"epoka {best_epoch} | "
            f"val mean = {best_mean_error:.4f} m | "
            f"val max = {best_max_error:.4f} m | "
            f"progress = {best_progress_ratio:.2f} | "
            f"speed = {best_mean_speed:.3f} m/s"
        )
    else:
        torch.save(model.state_dict(), args.output)
        print(f"zapisano model z ostatniej epoki: {args.output}")

    if args.save_history:
        save_training_history(history, args.history_csv, args.loss_plot)
        print(f"zapisano historie: {args.history_csv}")
        print(f"zapisano wykres uczenia: {args.loss_plot}")

    live_plot.finish(keep_open=not args.close_plot_on_finish)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64000)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--voltage-limit", type=float, default=12.0)
    parser.add_argument("--min-cruise-speed", type=float, default=0.14, help="Minimalna predkosc docelowa w danych treningowych [m/s].")
    parser.add_argument("--max-cruise-speed", type=float, default=0.32, help="Maksymalna predkosc docelowa w danych treningowych [m/s].")
    parser.add_argument("--slow-replay-fraction", type=float, default=0.30, help="Udzial probek wolnej jazdy chroniacy przed zapominaniem niskich predkosci.")
    parser.add_argument("--slow-cruise-speed-min", type=float, default=0.12, help="Dolny zakres wolnej jazdy replay [m/s].")
    parser.add_argument("--slow-cruise-speed-max", type=float, default=0.22, help="Gorny zakres wolnej jazdy replay [m/s].")
    parser.add_argument("--eval-cruise-speed", type=float, default=0.30, help="Predkosc docelowa podczas walidacji [m/s].")
    parser.add_argument("--eval-cruise-speeds", type=str, default=None, help="Kilka predkosci walidacyjnych po przecinku, np. 0.16,0.35,0.55.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output", type=str, default="direct_path_nn.pt")
    parser.add_argument("--workers", type=int, default=0, help="Liczba procesow do generowania danych; 0 = wszystkie watki CPU.")
    parser.add_argument("--torch-threads", type=int, default=0, help="Liczba watkow PyTorch; 0 = wszystkie watki CPU.")
    parser.add_argument("--history-csv", type=str, default="training_history.csv")
    parser.add_argument("--loss-plot", type=str, default="training_loss.png")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--no-live-plot", action="store_true")
    parser.add_argument("--close-plot-on-finish", action="store_true")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-paths", type=int, default=3)
    parser.add_argument("--eval-duration", type=float, default=20.0)
    parser.add_argument("--max-error-weight", type=float, default=0.25, help="Waga bledu maksymalnego przy wyborze najlepszego modelu.")
    parser.add_argument("--progress-error-weight", type=float, default=0.20, help="Waga bledu postepu przy wyborze najlepszego modelu.")
    parser.add_argument("--no-save-best-during-training", action="store_true", help="Nie zapisuj najlepszego modelu w trakcie treningu, tylko na koncu.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
