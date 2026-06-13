from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import torch
from torch import nn

from controller import wrap_to_pi


DIRECT_LOOKAHEADS = (0.0, 0.15, 0.35, 0.70, 1.10, 1.60, 2.30)
DIRECT_FEATURE_SIZE = 5 + 3 * len(DIRECT_LOOKAHEADS)
ANGLE_SCALE = 3.141592653589793
FEATURE_CLIP = 3.0
DEFAULT_CRUISE_SPEED = 0.16
MAX_LATERAL_ACCEL = 0.18


def _preview_scales():
    scales = []
    for distance in DIRECT_LOOKAHEADS:
        # local_x rosnie razem z odlegloscia lookahead.
        # local_y to blad boczny/polozenie poprzeczne, zwykle mniejsze.
        # local_theta jest katem w radianach.
        scales.extend([max(0.25, distance), 0.8, ANGLE_SCALE])
    return scales


DIRECT_FEATURE_SCALES = [
    1.5,   # omega robota [rad/s]
    0.4,   # v robota [m/s]
    3.0,   # remaining obciete do 3 m
    0.25,  # target_speed [m/s]
    1.0,   # flaga dojazdu do mety
    *_preview_scales(),
]


def desired_speed_from_remaining(remaining: float, cruise_speed: float = DEFAULT_CRUISE_SPEED) -> float:
    # Profil predkosci dla otwartej trasy: jedziemy normalnie, a przed meta zwalniamy.
    if remaining <= 0.0:
        return 0.0
    return cruise_speed * min(1.0, remaining / 0.75)


def path_curvature(path, progress_s: float, ds: float = 0.05) -> float:
    # Przyblizona krzywizna toru w punkcie progress_s.
    # Duza krzywizna oznacza ostry zakret, wiec bezpieczna predkosc musi byc mniejsza.
    before = path.sample(progress_s - ds)
    after = path.sample(progress_s + ds)
    return wrap_to_pi(after.theta - before.theta) / (2.0 * ds)


def target_speed_for_path(
    state,
    path,
    progress_s: float,
    remaining: float,
    cruise_speed: float = DEFAULT_CRUISE_SPEED,
) -> float:
    # Predkosc docelowa dla szybkiego, ale stabilnego path following.
    # Na prostych dazymy do cruise_speed, na zakretach ograniczamy predkosc
    # z przyblizonego warunku przyspieszenia bocznego: a_y = v^2 * kappa.
    speed = desired_speed_from_remaining(remaining, cruise_speed)

    curvature = abs(path_curvature(path, progress_s))
    if curvature > 1e-6:
        curvature_speed = sqrt(MAX_LATERAL_ACCEL / curvature)
        speed = min(speed, curvature_speed)

    # Jesli robot juz ma blad boczny albo katowy, chwilowo zwalniamy,
    # bo wtedy priorytetem jest powrot na sciezke.
    _local_x, lateral_error, heading_error = path.local_preview(state, progress_s, (0.0,))[0:3]
    lateral_scale = max(0.55, 1.0 - 1.8 * max(0.0, abs(lateral_error) - 0.04))
    heading_scale = max(0.55, 1.0 - 1.2 * max(0.0, abs(heading_error) - 0.15))
    speed *= min(lateral_scale, heading_scale)

    if speed <= 0.0:
        return 0.0
    return max(0.05, min(cruise_speed, speed))


def feasible_progress_for_duration(path, duration: float, cruise_speed: float, dt: float = 0.02) -> float:
    # Oczekiwany postep po sciezce dla idealnego robota jadacego zgodnie
    # z tym samym profilem predkosci, ktory widzi siec.
    # To jest uczciwszy punkt odniesienia niz samo cruise_speed * czas,
    # bo profil automatycznie zwalnia na ostrych zakretach.
    t = 0.0
    progress_s = 0.0
    travelled = 0.0

    while t < duration:
        step = min(dt, duration - t)
        pose = path.sample(progress_s)
        state_on_path = (pose.theta, pose.x, pose.y, 0.0, 0.0)
        remaining = path.total_length if path.closed else max(0.0, path.total_length - progress_s)
        speed = target_speed_for_path(state_on_path, path, progress_s, remaining, cruise_speed)
        ds = speed * step

        travelled += ds
        progress_s += ds
        if not path.closed:
            progress_s = min(progress_s, path.total_length)
            travelled = min(travelled, path.total_length)

        t += step

    return travelled


def normalize_direct_features(raw_features):
    # Stala normalizacja wejsc sieci.
    # Uzywamy tych samych skal w treningu i podczas jazdy robota.
    normalized = []
    for value, scale in zip(raw_features, DIRECT_FEATURE_SCALES):
        scaled = value / scale
        scaled = max(-FEATURE_CLIP, min(FEATURE_CLIP, scaled))
        normalized.append(scaled)
    return normalized


def build_direct_path_features(
    state,
    path,
    progress_s: float,
    remaining: float,
    cruise_speed: float = DEFAULT_CRUISE_SPEED,
):
    # Cechy dla sieci napieciowej:
    # - aktualna predkosc katowa i liniowa,
    # - ile zostalo do konca,
    # - zadana predkosc wynikajaca z profilu dojazdu do mety,
    # - lokalne punkty trasy przed robotem.
    _theta, _x, _y, omega, v = state
    target_speed = target_speed_for_path(state, path, progress_s, remaining, cruise_speed)
    preview = path.local_preview(state, progress_s, DIRECT_LOOKAHEADS)

    raw_features = [
        omega,
        v,
        min(remaining, 3.0),
        target_speed,
        1.0 if remaining < 0.75 else 0.0,
        *preview,
    ]
    return normalize_direct_features(raw_features)


class DirectVoltagePathGRU(nn.Module):
    # GRU-policy dla path following.
    # Siec ma pamiec poprzednich krokow, co pomaga przy sterowaniu napieciami
    # bez podrzednego regulatora PI/PIC.
    def __init__(self, input_size=DIRECT_FEATURE_SIZE, hidden_size=96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x, hidden=None):
        # x moze miec ksztalt:
        # - [batch, features],
        # - [batch, sequence, features].
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)

        batch, sequence, features = x.shape
        encoded = self.encoder(x.reshape(batch * sequence, features))
        encoded = encoded.reshape(batch, sequence, -1)
        recurrent, hidden = self.gru(encoded, hidden)
        output = self.head(recurrent)

        if single_step:
            return output[:, 0, :]
        return output

    def step(self, x, hidden=None):
        # Jeden krok inferencji z przeniesieniem pamieci GRU.
        if x.dim() == 1:
            x = x.unsqueeze(0)
        encoded = self.encoder(x).unsqueeze(1)
        recurrent, hidden = self.gru(encoded, hidden)
        output = self.head(recurrent[:, 0, :])
        return output, hidden


# Alias dla starszych importow w skryptach treningowych.
DirectVoltagePathMLP = DirectVoltagePathGRU


@dataclass
class DirectNNVoltageController:
    # Regulator w 100% neuronowy w czasie jazdy.
    # Nie ma tu PI/PIC ani klasycznego regulatora platformy.
    model_path: str = "direct_path_nn.pt"
    voltage_limit: float = 12.0
    cruise_speed: float = DEFAULT_CRUISE_SPEED

    def __post_init__(self):
        self.model = DirectVoltagePathGRU()
        path = Path(self.model_path)
        if path.exists():
            try:
                self.model.load_state_dict(torch.load(path, map_location="cpu"))
            except RuntimeError:
                # Stary plik modelu mogl byc zapisany dla MLP.
                # Wtedy startujemy z losowym GRU i trzeba uruchomic trening.
                pass
        self.model.eval()
        self.hidden = None

    def reset(self):
        self.hidden = None

    def compute_voltage(self, state, path, progress_s: float, remaining: float):
        features = build_direct_path_features(state, path, progress_s, remaining, self.cruise_speed)
        x = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            raw, self.hidden = self.model.step(x, self.hidden)
            raw = raw.squeeze(0)

        voltage_right = float(torch.tanh(raw[0]) * self.voltage_limit)
        voltage_left = float(torch.tanh(raw[1]) * self.voltage_limit)
        return voltage_right, voltage_left
