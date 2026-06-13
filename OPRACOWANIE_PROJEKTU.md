# Opracowanie projektu: neuronowe sledzenie sciezki przez robota mobilnego 2.0

Ten dokument opisuje finalna wersje projektu tak, zeby mozna bylo obronic kod na pytania typu:

- co robi dana funkcja,
- co oznacza dana zmienna,
- dlaczego przyjeto taka strukture programu,
- dlaczego uzyto sieci GRU,
- jak odbywa sie uczenie,
- jak dziala model robota,
- jak dziala animacja,
- dlaczego robot nie ma klasycznego regulatora PI/PID/PIC w czasie jazdy.

Dokument nie opisuje prywatnego "toku myslenia" krok po kroku, tylko techniczne uzasadnienie projektu. To jest bezpieczniejsza i bardziej inzynierska forma: pokazuje decyzje projektowe, zaleznosci i sens implementacji.

## 1. Cel projektu

Celem projektu jest symulacja robota mobilnego roznicowego, ktory porusza sie po zadanej zamknietej sciezce. Robot ma dwa niezaleznie sterowane kola: prawe i lewe. Sterowanie odbywa sie bez klasycznego regulatora PI/PIC w czasie jazdy. Siec neuronowa dostaje aktualny stan robota oraz lokalny opis sciezki przed robotem i zwraca bezposrednio napiecia:

```text
U_p - napiecie prawego kola
U_l - napiecie lewego kola
```

Napiecia sa ograniczone do zakresu:

```text
-12 V ... +12 V
```

Czyli wyjscie sieci jest fizycznie zgodne z zalozona saturacja napiecia silnikow.

## 2. Ogolna architektura projektu

Projekt sklada sie z kilku warstw:

```text
road_path.py
    generuje i obsluguje sciezke

robot_model.py
    opisuje model kinematyczno-dynamiczny robota

direct_path_nn.py
    definiuje siec GRU i kontroler neuronowy

expert_voltage.py
    generuje przykladowe poprawne sterowania do uczenia sieci

train_direct_path_nn.py
    uczy siec na danych generowanych z eksperta

simulate_path_nn.py
    wykonuje symulacje zamknietej petli na potrzeby walidacji

animate_path_nn.py
    pokazuje animacje w czasie rzeczywistym

controller.py
    zawiera funkcje pomocnicze do katow i bledow

trajectory.py
    zawiera strukture punktu referencyjnego sciezki
```

Najwazniejszy przeplyw danych w czasie jazdy:

```text
stan robota + sciezka
        |
        v
cechy wejściowe sieci
        |
        v
GRU
        |
        v
U_p, U_l
        |
        v
model robota
        |
        v
nowy stan robota
```

## 3. Dlaczego siec GRU?

Uzyta siec to `GRU`, czyli Gated Recurrent Unit. Jest to siec rekurencyjna, czyli taka, ktora ma stan pamieciowy. W tym projekcie jest to istotne, bo sterowanie robotem nie zalezy tylko od aktualnego punktu na sciezce. Zalezy rowniez od tego, jak robot zachowywal sie chwile wczesniej:

- czy juz skrecal,
- czy mial predkosc obrotowa,
- czy doganial sciezke,
- czy byl po lewej/prawej stronie toru,
- czy napiecia powinny byc lagodne, czy agresywne.

Zwykla siec MLP traktowalaby kazdy krok niezaleznie. GRU pamieta kontekst sekwencji. W sterowaniu ruchem to ma duzy sens, bo robot ma dynamike i bezwladnosc.

GRU jest tez prostsza od LSTM:

- ma mniej bramek,
- ma mniej parametrow,
- szybciej sie uczy,
- latwiej dziala w malej symulacji,
- nadal przechowuje potrzebna pamiec.

Dlatego w tym projekcie GRU jest kompromisem: wystarczajaco mocna do sterowania sekwencyjnego, ale nie przesadnie skomplikowana.

## 4. Jak uczy sie siec?

Siec uczy sie nadzorowanie, czyli przez nasladowanie eksperta.

Ekspert to funkcja z pliku `expert_voltage.py`. Dla danego stanu robota i sciezki ekspert liczy napiecia, ktore powinny dobrze prowadzic robota po torze. Potem siec dostaje:

```text
wejscie = stan robota + lokalny podglad sciezki
cel = napiecie eksperta [U_p, U_l]
```

Siec przewiduje swoje napiecia. Potem liczony jest blad MSE:

```text
loss = srednia((U_siec - U_ekspert)^2)
```

Napiecia w treningu sa dzielone przez `12 V`, czyli uczenie odbywa sie na zakresie bliskim `[-1, 1]`. To stabilizuje uczenie.

Dodatkowo model jest walidowany w zamknietej petli:

```text
GRU -> napiecia -> model robota -> nowy stan -> GRU -> ...
```

Dzieki temu nie patrzymy tylko na to, czy siec kopiuje eksperta, ale tez czy robot naprawde jedzie po sciezce.

## 5. Najwazniejsze metryki

W logach treningu pojawiaja sie:

```text
loss
val mean
val max
progress
speed
```

Znaczenie:

```text
loss
    blad MSE napiec wzgledem eksperta

val mean
    sredni blad od sciezki w symulacji walidacyjnej

val max
    maksymalny chwilowy blad od sciezki

progress
    stosunek rzeczywistego postepu po sciezce do oczekiwanego postepu

speed
    srednia predkosc uzyskana w walidacji
```

Idealnie:

```text
loss maleje
val mean jest maly
val max jest maly
progress jest blisko 1.0
speed jest blisko zadanej predkosci
```

## 6. Plik `trajectory.py`

Ten plik jest najprostszy. Zawiera tylko klase danych `Reference`.

Kod:

```python
from dataclasses import dataclass
```

Import `dataclass` pozwala szybko zdefiniowac strukture danych bez pisania recznego konstruktora `__init__`.

```python
@dataclass(frozen=True)
class Reference:
```

`@dataclass` generuje konstruktor i czytelna reprezentacje obiektu. `frozen=True` oznacza, ze po utworzeniu obiektu nie powinno sie zmieniac jego pol. To pasuje do punktu referencyjnego, bo referencja jest wynikiem obliczen dla konkretnego miejsca na sciezce.

Pola:

```python
x: float
y: float
theta: float
v: float
omega: float
s: float = 0.0
```

Znaczenie:

```text
x, y
    polozenie punktu referencyjnego na sciezce

theta
    orientacja stycznej do sciezki w tym punkcie

v
    predkosc liniowa referencyjna

omega
    predkosc katowa wynikajaca z krzywizny sciezki

s
    dlugosc luku od poczatku sciezki
```

Po co to jest? Ekspert potrzebuje informacji, jaki kierunek i jaka krzywizna sa w najblizszym punkcie toru. `Reference` jest wygodnym kontenerem na te dane.

## 7. Plik `controller.py`

Ten plik zawiera funkcje pomocnicze matematyczne.

### Importy

```python
from math import atan2, cos, sin
```

Potrzebne sa funkcje trygonometryczne:

- `sin`, `cos` do transformacji ukladow wspolrzednych,
- `atan2` do normalizacji kata.

```python
from trajectory import Reference
```

`Reference` jest typem punktu referencyjnego uzywanym w funkcji liczacej blad sledzenia.

### `wrap_to_pi(angle)`

```python
def wrap_to_pi(angle: float) -> float:
    return atan2(sin(angle), cos(angle))
```

Ta funkcja sprowadza kat do zakresu:

```text
[-pi, pi]
```

Dlaczego to jest potrzebne? Kat `3.14 rad` i `-3.14 rad` sa prawie tym samym kierunkiem, ale zwykle odejmowanie katow moze dac duzy skok. Normalizacja zapobiega sytuacji, w ktorej robot probuje wykonac prawie pelny obrot, mimo ze roznica kierunkow jest mala.

Dlaczego `atan2(sin(angle), cos(angle))`? To standardowa metoda normalizacji kata. Dziala dla dowolnie duzych wartosci, bo sinus i cosinus sa okresowe, a `atan2` odtwarza kat w zakresie `[-pi, pi]`.

### `tracking_errors(state, reference)`

```python
def tracking_errors(state, reference: Reference):
```

Funkcja liczy blad pozycji robota wzgledem punktu referencyjnego.

```python
theta, x, y, _omega, _v = state
```

Stan robota ma postac:

```text
theta, x, y, omega, v
```

Tutaj potrzebne sa tylko `theta`, `x`, `y`. Zmienna `_omega` i `_v` maja podkreslenie, bo sa czescia stanu, ale w tej funkcji nie sa uzywane.

```python
dx = reference.x - x
dy = reference.y - y
```

To blad w globalnym ukladzie wspolrzednych.

```python
e1 = cos(theta) * dx + sin(theta) * dy
e2 = -sin(theta) * dx + cos(theta) * dy
```

To przeksztalcenie bledu z ukladu globalnego do lokalnego ukladu robota:

- `e1` to blad w osi przod-tyl robota,
- `e2` to blad boczny.

Jest to wazne, bo dla robota bardziej naturalne jest pytanie: "czy punkt jest przede mna, czy z boku?", a nie tylko "jaka jest roznica x/y w mapie".

```python
e3 = wrap_to_pi(reference.theta - theta)
```

`e3` to blad orientacji. Uzycie `wrap_to_pi` zabezpiecza przed skokiem kata.

```python
return e1, e2, e3
```

Funkcja zwraca trzy skladowe bledu.

W finalnym projekcie ta funkcja nie steruje robotem bezposrednio. Jest pomocnicza i pozostala jako czytelna definicja bledow lokalnych.

## 8. Plik `robot_model.py`

Ten plik jest modelem kinematyczno-dynamicznym robota.

### Typy danych

```python
State = Tuple[float, float, float, float, float]
Vector2 = Tuple[float, float]
```

`State` oznacza:

```text
(theta, x, y, omega, v)
```

gdzie:

```text
theta
    orientacja robota [rad]
x, y
    polozenie robota [m]
omega
    predkosc katowa platformy [rad/s]
v
    predkosc liniowa platformy [m/s]
```

`Vector2` oznacza dwuelementowy wektor. W projekcie jest uzywany m.in. dla:

```text
(U_p, U_l)
(omega_prawego_kola, omega_lewego_kola)
(moment_prawy, moment_lewy)
```

### `RobotParams`

`RobotParams` przechowuje parametry fizyczne robota.

```python
m = 24.0
```

Masa platformy w kilogramach.

```python
b = 0.15
```

Polowa rozstawu kol. Pelny rozstaw kol to `2*b`.

```python
r = 0.0845
```

Promien kola.

```python
xi = 0.1
Ik = 1.014e-5
```

`xi` to tlumienie w lozyskach kol, `Ik` to moment bezwladnosci kola.

```python
Im = 4.22e-3 * 1e-4
xi_m = 1.85e-8
ng = 1.0
R = 3.78
km = 0.855
```

Parametry silnika:

- `Im` - bezwladnosc wirnika,
- `xi_m` - tlumienie w lozyskach silnika,
- `ng` - przelozenie,
- `R` - rezystancja uzwojen,
- `km` - stala momentowa/maszynowa.

```python
g = 9.81
Cr = 5e-4
Daw = 5e-4
Dav = 5e-3
nu = 10.0
```

Parametry oporow:

- `Cr` - wspolczynnik oporu toczenia,
- `Daw` - opor aerodynamiczny obrotu,
- `Dav` - opor aerodynamiczny ruchu postepowego,
- `nu` - wspolczynnik wygladzania funkcja `tanh`.

### `Ic`

```python
@property
def Ic(self) -> float:
    B = 2.0 * self.b
    return self.m * (B**2 + B**2) / 12.0
```

To moment bezwladnosci platformy wzgledem osi pionowej. Przyjeto przyblizenie dla prostokatnej platformy:

```text
Ic = m * (B^2 + B^2) / 12
```

Poniewaz w projekcie szerokosc i dlugosc potraktowano jako `B`, wzor jest uproszczony.

### `wheel_speeds(u, p)`

```python
omega, v = u
return (p.b / p.r * omega + v / p.r, -p.b / p.r * omega + v / p.r)
```

Funkcja zamienia predkosc platformy:

```text
omega - obrot platformy
v     - ruch postepowy
```

na predkosci katowe prawego i lewego kola.

Dla robota roznicowego:

```text
omega_p = (v + b*omega)/r
omega_l = (v - b*omega)/r
```

Jesli robot jedzie prosto, `omega = 0`, oba kola maja te sama predkosc. Jesli robot obraca sie w miejscu, jedno kolo przyspiesza, drugie zwalnia albo jedzie w druga strone.

### `motor_shaft_speeds`

```python
return omega_p / p.ng, omega_l / p.ng
```

Uwzglednia przelozenie przekladni. W tym projekcie `ng = 1`, wiec wartosc sie nie zmienia, ale funkcja zostaje, bo model jest ogolny.

### `motor_currents`

```python
return u_me[0] / p.R, u_me[1] / p.R
```

Prawo Ohma:

```text
i = U / R
```

`u_me` to napiecie skuteczne po odjeciu sily elektromotorycznej silnika.

### `motor_torques`

```python
return p.km * i_m[0], p.km * i_m[1]
```

Moment silnika jest proporcjonalny do pradu:

```text
T = km * i
```

### `back_emf`

```python
return p.km * omega_m[0], p.km * omega_m[1]
```

Silnik obracajac sie generuje sile elektromotoryczna przeciwna do zasilania. Im szybciej obraca sie silnik, tym mniej efektywnego napiecia zostaje na wytwarzanie momentu.

To dlatego robot ma ograniczona predkosc maksymalna mimo stalego `12 V`.

### `b_tilde_times_torque`

```python
wsp = 1.0 / (p.r * p.ng)
return p.b * wsp * t_p - p.b * wsp * t_l, wsp * t_p + wsp * t_l
```

Funkcja przelicza momenty silnikow/kol na wymuszenia platformy:

```text
skladowa 1 - moment obracajacy platforme
skladowa 2 - sila postepowa platformy
```

Jesli oba momenty sa takie same, robot jedzie do przodu. Jesli sa przeciwne, robot sie obraca.

### `h_tilde_times_u`

```python
h11 = ...
h22 = ...
return h11 * omega, h22 * v
```

To model tlumienia lepkiego zalezne od predkosci. Osobno liczony jest wplyw na ruch obrotowy i postepowy.

### `resistance_forces`

Funkcja liczy opory ruchu:

```python
n_a = p.Daw * abs(omega) * omega
f_a = p.Dav * abs(v) * v
```

To opory aerodynamiczne, rosnace mniej wiecej kwadratowo z predkoscia.

```python
n_r = ...
f_r = ...
```

To opor toczenia. Uzyto `tanh`, zeby wygladzic przejscie przez zero. Bez wygladzania znak oporu moglby skakac, co pogarszaloby stabilnosc numeryczna.

### `m_tilde_inverse_diagonal`

Ta funkcja liczy odwrotnosc efektywnej macierzy bezwladnosci. W modelu macierz jest diagonalna, wiec wystarczy odwrocic dwa elementy:

```text
1 / m11
1 / m22
```

`m11` dotyczy obrotu, `m22` ruchu postepowego.

### `derivatives`

To najwazniejsza funkcja modelu ciaglego.

Wejscia:

```text
t       - czas, w tym modelu nieuzywany jawnie
state   - theta, x, y, omega, v
voltage - U_p, U_l
p       - parametry robota
```

Najpierw rozpakowuje stan:

```python
theta, _x, _y, omega, v = state
```

`_x`, `_y` nie sa potrzebne do dynamiki predkosci, ale sa czescia stanu.

Potem liczony jest blok silnikow:

```python
omega_wheels = wheel_speeds(u, p)
omega_m = motor_shaft_speeds(omega_wheels, p)
e_m = back_emf(omega_m, p)
u_me = voltage[0] - e_m[0], voltage[1] - e_m[1]
i_m = motor_currents(u_me, p)
t_m = motor_torques(i_m, p)
```

Interpretacja:

1. z predkosci platformy liczysz predkosci kol,
2. z predkosci kol liczysz predkosci silnikow,
3. liczysz SEM,
4. odejmujesz SEM od napiecia,
5. liczysz prad,
6. liczysz moment.

Potem blok mechaniki:

```python
btm = b_tilde_times_torque(t_m, p)
hu = h_tilde_times_u(u, p)
forces = resistance_forces(u, p)
inv_m11, inv_m22 = m_tilde_inverse_diagonal(p)
```

To daje wymuszenia, tlumienia, opory i odwrotnosc masy/bezwladnosci.

Kinematyka:

```python
dtheta = omega
dx = v * cos(theta)
dy = v * sin(theta)
```

To klasyczny model ruchu robota w plaszczyznie.

Dynamika:

```python
domega = inv_m11 * (btm[0] - hu[0] - forces[0])
dv = inv_m22 * (btm[1] - hu[1] - forces[1])
```

Przyspieszenie wynika z:

```text
wymuszenie od silnikow - tlumienie - opory
```

Podzielone przez efektywna bezwladnosc.

### `rk4_step`

To calkowanie Rungego-Kutty 4. rzedu.

Dlaczego RK4, a nie Euler?

Euler jest prostszy, ale mniej dokladny. Przy dynamice silnikow i predkosci RK4 daje stabilniejsza symulacje bez drastycznego zmniejszania kroku czasowego.

Schemat:

```python
k1 = derivatives(t, state, voltage, p)
k2 = derivatives(t + dt / 2.0, state + k1*dt/2, voltage, p)
k3 = derivatives(t + dt / 2.0, state + k2*dt/2, voltage, p)
k4 = derivatives(t + dt, state + k3*dt, voltage, p)
```

Wynik:

```python
state_next = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
```

To standardowy wzor RK4.

## 9. Plik `road_path.py`

Ten plik odpowiada za geometrie sciezki.

### `PathPose`

```python
@dataclass(frozen=True)
class PathPose:
    x, y, theta, s
```

Jest to punkt na torze:

- `x`, `y` - polozenie,
- `theta` - kierunek stycznej,
- `s` - dlugosc luku od startu.

### `RoadPath.__init__`

Konstruktor przyjmuje liste punktow i informacje, czy tor jest zamkniety.

```python
if closed and self.points[0] != self.points[-1]:
    self.points.append(self.points[0])
```

Jesli tor jest zamkniety, ostatni punkt zostaje dopisany jako pierwszy punkt, aby petla byla domknieta.

```python
self.lengths = [0.0]
```

`lengths` przechowuje skumulowane dlugosci segmentow. Dzieki temu mozna probkowac sciezke po dlugosci luku `s`, a nie po indeksie punktu.

```python
self.total_length = self.lengths[-1]
```

Calkowita dlugosc sciezki.

### `sample(s)`

Funkcja zwraca punkt na sciezce dla zadanej dlugosci luku `s`.

Dla toru zamknietego:

```python
s = s % self.total_length
```

To powoduje zawijanie. Jesli robot przejedzie koniec petli, wraca na poczatek.

Dla toru otwartego:

```python
s = min(max(s, 0.0), self.total_length)
```

Ograniczenie do zakresu sciezki.

```python
i = min(max(1, bisect_right(self.lengths, s)), len(self.points) - 1)
```

`bisect_right` szybko znajduje segment, w ktorym znajduje sie zadane `s`.

```python
alpha = (s - s0) / segment_length
```

`alpha` mowi, jak daleko jestesmy w danym segmencie.

```python
x = x0 + alpha * (x1 - x0)
y = y0 + alpha * (y1 - y0)
theta = atan2(y1 - y0, x1 - x0)
```

Interpolacja liniowa pozycji i kierunek segmentu.

### `nearest_s`

Funkcja szuka najblizszego punktu na sciezce dla pozycji robota `(x, y)`.

Dla kazdego segmentu liczy rzut punktu robota na odcinek:

```python
alpha = ((x - x0) * dx + (y - y0) * dy) / segment2
alpha = min(max(alpha, 0.0), 1.0)
```

Ograniczenie `alpha` do `[0, 1]` oznacza, ze rzut nie wychodzi poza segment.

Potem liczona jest odleglosc:

```python
distance2 = (x - px) ** 2 + (y - py) ** 2
```

Najmniejsza odleglosc wygrywa. Funkcja zwraca:

```text
best_s, sqrt(best_distance2)
```

Czyli pozycje na torze i blad geometryczny.

### `reference_at_s`

Funkcja buduje punkt referencyjny na sciezce.

```python
pose = self.sample(s)
```

Pobiera punkt na torze.

```python
ds = 0.02
pose_before = self.sample(pose.s - ds)
pose_after = self.sample(pose.s + ds)
dtheta = wrap_to_pi(pose_after.theta - pose_before.theta)
curvature = dtheta / (2.0 * ds)
```

Krzywizna jest liczona numerycznie jako zmiana kata stycznej na malej odleglosci.

```python
omega = v_ref * curvature
```

Dla jazdy po krzywej predkosc katowa wynika z:

```text
omega = v * kappa
```

gdzie `kappa` to krzywizna.

### `local_preview`

To jedna z najwazniejszych funkcji dla sieci.

Siec nie dostaje calej mapy. Dostaje kilka punktow toru przed robotem, przeliczonych do lokalnego ukladu robota.

```python
local_x = cos(theta) * dx + sin(theta) * dy
local_y = -sin(theta) * dx + cos(theta) * dy
local_theta = wrap_to_pi(pose.theta - theta)
```

Znaczenie:

- `local_x` - punkt toru przed/za robotem,
- `local_y` - punkt toru po lewej/prawej stronie robota,
- `local_theta` - roznica orientacji toru i robota.

To jest lepsze niz globalne `x/y`, bo siec nie musi uczyc sie translacji i rotacji calego swiata. Dla niej wazne jest lokalne polozenie toru wzgledem robota.

### `_catmull_rom_point`

To interpolacja Catmulla-Roma. Tworzy gladka krzywa przechodzaca przez punkty kontrolne.

Dlaczego Catmull-Rom?

- nie wymaga SciPy,
- daje gladkie trasy,
- przechodzi przez punkty kontrolne,
- latwo zrobic zamknieta petle.

### `make_closed_spline_path`

Funkcja generuje losowy zamkniety tor.

```python
rng = Random(seed)
```

Seed pozwala odtworzyc ten sam tor.

Punkty kontrolne sa rozmieszczone wokol elipsy:

```python
radius_x = 2.0 + rng.uniform(-0.35, 0.35)
radius_y = 1.25 + rng.uniform(-0.25, 0.25)
```

Losowe zaburzenia sprawiaja, ze tor nie jest idealnym okregiem.

Potem dla kazdego odcinka generowane sa punkty splajnu Catmulla-Roma.

## 10. Plik `direct_path_nn.py`

Ten plik definiuje wejscia sieci, normalizacje, profil predkosci, architekture GRU i kontroler runtime.

### Stale

```python
DIRECT_LOOKAHEADS = (0.0, 0.15, 0.35, 0.70, 1.10, 1.60, 2.30)
```

To odleglosci przed robotem, dla ktorych pobieramy lokalne punkty sciezki. `0.0` to punkt najblizszy, dalsze wartosci daja podglad zakretu przed robotem.

```python
DIRECT_FEATURE_SIZE = 5 + 3 * len(DIRECT_LOOKAHEADS)
```

Wejscie sieci ma:

- 5 cech globalnych/profilowych,
- dla kazdego lookahead 3 cechy: `local_x`, `local_y`, `local_theta`.

```python
ANGLE_SCALE = pi
FEATURE_CLIP = 3.0
```

Skale do normalizacji. `FEATURE_CLIP` ogranicza ekstremalne wartosci, zeby siec nie dostawala ogromnych liczb.

```python
DEFAULT_CRUISE_SPEED = 0.16
MAX_LATERAL_ACCEL = 0.18
```

Domyslna predkosc i ograniczenie przyspieszenia bocznego. Przy ostrych zakretach predkosc musi spadac, bo inaczej robot nie utrzyma sciezki przy ograniczonym napieciu.

### `_preview_scales`

Buduje skale normalizacji dla punktow podgladu sciezki.

Dla kazdego lookahead dodaje:

```text
skala local_x
skala local_y
skala local_theta
```

Normalizacja jest wazna, bo siec neuronowa uczy sie stabilniej, gdy wejscia maja podobne zakresy.

### `DIRECT_FEATURE_SCALES`

Lista skal dla wszystkich cech:

```text
omega robota
v robota
remaining
target_speed
flaga konca trasy
preview points
```

Kazda cecha jest dzielona przez swoja skale.

### `desired_speed_from_remaining`

Dla otwartej trasy funkcja zwalnia przed koncem. Dla zamknietej trasy `remaining` jest stale rowne dlugosci toru, wiec predkosc jest zasadniczo `cruise_speed`.

### `path_curvature`

Liczy krzywizne toru numerycznie:

```python
before = path.sample(progress_s - ds)
after = path.sample(progress_s + ds)
return wrap_to_pi(after.theta - before.theta) / (2.0 * ds)
```

To pozwala ocenic, czy przed robotem jest ostry zakret.

### `target_speed_for_path`

To profil predkosci: szybciej na prostych, wolniej na zakretach.

Najpierw bierze predkosc bazowa:

```python
speed = desired_speed_from_remaining(remaining, cruise_speed)
```

Potem ogranicza ja przez krzywizne:

```python
curvature_speed = sqrt(MAX_LATERAL_ACCEL / curvature)
speed = min(speed, curvature_speed)
```

Wynika to z zaleznosci:

```text
a_y = v^2 * kappa
v = sqrt(a_y / kappa)
```

Jesli robot ma duzy blad boczny albo katowy, funkcja dodatkowo zwalnia. To poprawia stabilnosc.

### `feasible_progress_for_duration`

Ta funkcja liczy, ile powinien przejechac idealny robot jadacy zgodnie z profilem predkosci.

Dlaczego nie uzywamy prostego:

```text
cruise_speed * czas
```

Bo profil sam zwalnia na zakretach. Gdyby porownywac do stalej predkosci, `progress` bylby niesprawiedliwie zanizony.

### `normalize_direct_features`

Kazda cecha jest dzielona przez skale i ograniczana do `[-3, 3]`.

Dlaczego?

- stabilniejsze uczenie,
- brak dominacji jednej cechy,
- mniejsze ryzyko nasycenia sieci przez skrajne wartosci.

### `build_direct_path_features`

Buduje wektor wejscia sieci.

Zawiera:

```text
omega
v
remaining
target_speed
flaga konca
lokalne punkty sciezki
```

Najwazniejszy fragment:

```python
preview = path.local_preview(state, progress_s, DIRECT_LOOKAHEADS)
```

To daje sieci informacje o ksztalcie toru przed robotem.

### `DirectVoltagePathGRU`

Architektura:

```text
features -> Linear -> Tanh -> GRU -> Linear -> Tanh -> Linear -> 2 wyjscia
```

Pierwsza warstwa:

```python
self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
```

Koduje surowe cechy do rozmiaru `hidden_size`.

GRU:

```python
self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
```

Przetwarza sekwencje. `batch_first=True` oznacza format:

```text
[batch, sequence, features]
```

Glowica:

```python
self.head = nn.Sequential(...)
```

Zamienia stan GRU na dwa wyjscia: prawe i lewe napiecie.

W `forward` funkcja obsluguje dwa przypadki:

- pojedynczy krok,
- cala sekwencje.

To jest wygodne, bo trening uzywa sekwencji, a runtime dziala krok po kroku.

### `step`

`step` sluzy do jazdy online. Przyjmuje jeden wektor cech i poprzedni stan ukryty GRU. Zwraca:

```text
output, hidden
```

`hidden` jest pamiecia sieci przenoszona miedzy krokami.

### `DirectNNVoltageController`

To wrapper uzywany w animacji i symulacji.

```python
self.model = DirectVoltagePathGRU()
```

Tworzy model.

```python
self.model.load_state_dict(torch.load(path, map_location="cpu"))
```

Wczytuje wyuczone wagi z pliku `.pt`.

```python
self.model.eval()
```

Przelacza model w tryb inferencji.

```python
self.hidden = None
```

Pamiec GRU startuje pusta.

`compute_voltage`:

1. buduje cechy,
2. robi tensor PyTorch,
3. uruchamia `model.step`,
4. przepuszcza wyjscie przez `tanh`,
5. mnozy przez `12 V`.

Dlaczego `tanh`?

Bo `tanh` daje zakres `[-1, 1]`. Po pomnozeniu przez `12` dostajemy saturacje napiecia:

```text
-12 V ... +12 V
```

## 11. Plik `expert_voltage.py`

Ekspert sluzy tylko do generowania danych treningowych. W finalnej jezdzie robot jest sterowany siecia.

### `_clamp`

Ogranicza wartosc do przedzialu.

Uzywane do:

- ograniczenia predkosci,
- ograniczenia predkosci katowej,
- saturacji napiec.

### `_steady_voltage_for_platform`

Funkcja robi przyblizone odwrocenie modelu.

Wejscie:

```text
target_platform = (omega_cmd, v_cmd)
```

Wyjscie:

```text
voltage_right, voltage_left
```

Najpierw liczone sa tlumienia i opory dla zadanej predkosci:

```python
q_h = h_tilde_times_u(...)
q_f = resistance_forces(...)
```

Potem wymagane wymuszenia sa zamieniane na momenty prawego i lewego kola.

Na koncu dodawana jest SEM:

```python
voltage = R * torque / km + e_m
```

To odpowiada rownaniu silnika:

```text
U = R*i + e
T = km*i
```

### `expert_voltage_for_path`

Ekspert:

1. znajduje docelowa predkosc,
2. liczy blad boczny i katowy,
3. wyznacza zadane `omega_cmd` i `v_cmd`,
4. zamienia to na napiecia kol,
5. dodaje korekcje predkosci kol,
6. ogranicza napiecia do `[-12, 12]`.

Najwazniejsze linie:

```python
target_speed = target_speed_for_path(...)
```

Ekspert uzywa tego samego profilu predkosci, co siec w runtime. To jest wazne, bo dane treningowe i dzialanie sa spojne.

```python
_local_x, lateral_error, heading_error = path.local_preview(...)[0:3]
```

Pobiera blad boczny i katowy wzgledem najblizszego punktu sciezki.

```python
v_cmd = target_speed * clamp(cos(heading_error), 0.35, 1.0)
```

Jesli robot jest zle ustawiony, predkosc liniowa spada. To zabezpiecza przed jechaniem szybko bokiem.

```python
omega_cmd = reference.omega + 2.2 * heading_error + 3.0 * atan2(lateral_error, 0.35)
```

Skladowe:

- `reference.omega` - feedforward z krzywizny sciezki,
- `heading_error` - korekcja orientacji,
- `lateral_error` - dociaganie do sciezki.

`atan2(lateral_error, 0.35)` daje gladna nieliniowa korekcje. Dla malych bledow jest prawie liniowa, dla duzych sie ogranicza.

```python
speed_kp = 2.2
```

Prosta korekcja predkosci kol. Ekspert nie jest finalnym regulatorem, ale ma dawac dobre przyklady napiec.

## 12. Plik `simulate_path_nn.py`

Ten plik robi symulacje zamknietej petli na potrzeby walidacji treningu.

### `DirectPathSample`

Zapisuje tylko to, co jest potrzebne w walidacji:

```text
t
progress_s
nearest_distance
```

Czyli czas, pozycje na sciezce i blad od sciezki.

### `initial_state_for_path`

Ustawia robota na poczatku sciezki:

```python
return start.theta, start.x, start.y, 0.0, 0.0
```

Robot startuje z:

- pozycja na poczatku toru,
- orientacja styczna do toru,
- zerowa predkoscia.

### `simulate_direct_path_following`

Pętla:

1. znajdz najblizszy punkt sciezki,
2. policz ile zostalo,
3. pobierz napiecia z kontrolera,
4. zapisz probke,
5. wykonaj krok RK4.

```python
controller = controller or DirectNNVoltageController()
```

Jesli nie podasz kontrolera, uzywany jest domyslny model NN.

```python
if hasattr(controller, "reset"):
    controller.reset()
```

Resetuje pamiec GRU przed nowa symulacja.

```python
state = initial_state if initial_state is not None else initial_state_for_path(path)
```

Pozwala albo podac wlasny stan startowy, albo startowac z poczatku toru.

## 13. Plik `animate_path_nn.py`

Ten plik pokazuje symulacje w czasie rzeczywistym.

### Stale

```python
DEFAULT_ANIMATION_SPEEDUP = 2.0
DEFAULT_CRUISE_SPEED = 0.55
DEFAULT_INTERVAL_MS = 20
```

Domyslnie:

- animacja idzie x2,
- zadana predkosc to `0.55 m/s`,
- odswiezanie okna co `20 ms`.

### `transform_path_to_pose`

Po okrazeniu generowana jest nowa trasa. Ta funkcja obraca i przesuwa nowy spline tak, zeby zaczynal sie dokladnie w aktualnej pozycji robota i z aktualnym kierunkiem.

Bez tego robot dostalby nagly skok ukladu odniesienia.

Matematycznie:

```python
x_new = x_target + ca*dx - sa*dy
y_new = y_target + sa*dx + ca*dy
```

To standardowy obrot i translacja punktu.

### `unwrapped_forward_delta`

Dla toru zamknietego `s` skacze z konca na poczatek. Funkcja rozpoznaje taki skok i poprawnie liczy postep.

Przyklad:

```text
poprzednio s = 9.9
teraz s = 0.1
```

Bez poprawki wyszloby `-9.8`, a faktycznie robot przejechal do przodu `0.2`.

### `RealtimePathAnimation.__init__`

Tworzy cala animacje:

- model robota,
- kontroler NN,
- poczatkowa sciezke,
- stan robota,
- listy do rysowania historii,
- okno Matplotlib.

```python
self.running = False
```

Symulacja nie startuje od razu. Czeka na przycisk `Start`.

### `_build_static_plot`

Tworzy elementy wykresu:

- linie sciezki,
- slad robota,
- punkt startu,
- kolko robota,
- linie kierunku,
- tekst informacyjny,
- wykres bledu,
- wykres napiec,
- przycisk start.

```python
self.ax_path.set_aspect("equal", adjustable="box")
```

Dzieki temu metr na osi X i metr na osi Y maja taka sama skale.

### `start_simulation`

Po kliknieciu:

```python
self.running = True
self.start_button_axis.set_visible(False)
```

Symulacja rusza, a przycisk znika.

### `_refresh_path_plot`

Aktualizuje narysowana sciezke. Jest wywolywana na starcie i po zmianie toru.

### `_change_path_after_lap`

Po jednym okrazeniu:

1. zwieksza seed,
2. generuje nowy spline,
3. dopasowuje go do pozycji robota,
4. resetuje pamiec GRU,
5. zeruje postep okrazenia.

Reset pamieci GRU jest potrzebny, bo kontekst sciezki zmienia sie nagle.

### `_simulate_one_step`

To jeden krok fizyki:

```python
progress_s, nearest_distance = path.nearest_s(...)
```

Mierzy blad i pozycje na torze.

```python
voltage = controller.compute_voltage(...)
```

Siec wyznacza napiecia.

```python
state = rk4_step(...)
```

Model robota przechodzi do nowego stanu.

### `_append_live_history`

Dopisuje dane do wykresow live i ucina stare probki, zeby listy nie rosly bez konca.

### `_refresh_robot_plot`

Aktualizuje:

- polozenie kolka robota,
- kierunek robota,
- slad,
- tekst informacyjny.

### `_refresh_history_plots`

Aktualizuje wykres bledu i napiec.

Pokazuje ostatnie 30 sekund symulacji, zeby wykres byl czytelny.

### `update`

Funkcja wywolywana przez `FuncAnimation`.

Jesli `running = True`, wykonuje kilka krokow symulacji na jedna klatke animacji. To pozwala przyspieszyc wizualizacje.

### `main`

Parsuje argumenty z terminala. Jesli `steps-per-frame = 0`, liczba krokow na klatke jest wyliczana z `speedup`.

Dlatego domyslnie wystarczy:

```powershell
python animate_path_nn.py
```

## 14. Plik `train_direct_path_nn.py`

To najwiekszy plik, bo zawiera generowanie danych, trening, walidacje i wykresy live.

### `available_cpu_threads`

Zwraca liczbe logicznych watkow CPU:

```python
return max(1, os.cpu_count() or 1)
```

`max(1, ...)` zabezpiecza przed wartoscia `None`.

### `resolve_thread_count`

Jesli uzytkownik poda `0`, funkcja uzywa wszystkich watkow. Jesli poda liczbe dodatnia, ogranicza ja do liczby dostepnych watkow.

### `parse_speed_list`

Zmienia tekst:

```text
0.16,0.35,0.55
```

na liste:

```python
[0.16, 0.35, 0.55]
```

To pozwala walidowac model na kilku predkosciach.

### `choose_cruise_speed`

Losuje predkosc treningowa.

Jesli wylosuje sie slow replay, wybiera wolna predkosc. Inaczej wybiera predkosc z glownego zakresu.

Po co slow replay? Gdy model jest douczany tylko na wysokich predkosciach, moze pogorszyc zachowanie przy niskich predkosciach. Slow replay temu przeciwdziala.

### `random_state_near_path`

Losuje stan robota w okolicy sciezki.

70% probek jest blisko toru, bo chcemy duzej precyzji w normalnej jezdzie.

30% probek ma wieksze odchylenia, zeby siec umiala wracac na tor.

Pozycja jest tworzona jako:

```text
punkt na sciezce
+ przesuniecie wzdluz stycznej
+ przesuniecie boczne
```

Orientacja jest zaburzana wzgledem stycznej toru.

### `_make_sequence_chunk`

Generuje fragment datasetu. Jest osobna funkcja, bo moze byc uruchamiana w wielu procesach.

Dla kazdej sekwencji:

1. generuje tor,
2. losuje predkosc,
3. losuje stan startowy,
4. przez `sequence_length` krokow:
   - liczy najblizszy punkt toru,
   - pyta eksperta o napiecia,
   - buduje cechy sieci,
   - zapisuje cel treningowy,
   - przesuwa stan robota modelem RK4.

Cele sa normalizowane:

```python
voltage / voltage_limit
```

czyli `12 V` odpowiada wartosci `1.0`.

### `make_sequence_dataset`

Dzieli generowanie danych na procesy.

```python
ProcessPoolExecutor(max_workers=workers)
```

Uzywa wielu rdzeni CPU do szybszego generowania datasetu.

Na koncu zwraca tensory:

```text
x - wejscia sieci
y - zadane napiecia
```

### `save_training_history`

Opcjonalnie zapisuje historie treningu do CSV i PNG. W normalnym finalnym uzyciu nie trzeba tego wlaczac, bo masz wykres live.

### `LiveTrainingPlot`

Tworzy okno z trzema wykresami:

1. loss MSE,
2. blad sledzenia,
3. progress.

`plt.ion()` wlacza tryb interaktywny.

`update` aktualizuje dane wykresow po kazdej epoce.

### `InMemoryDirectController`

Kontroler walidacyjny, ktory uzywa modelu w pamieci RAM, bez zapisywania go na dysk.

To wazne, bo podczas treningu chcemy sprawdzac aktualne wagi, zanim ostatecznie zapiszemy model.

### `evaluate_tracking_error`

Uruchamia symulacje zamknietej petli na kilku torach walidacyjnych.

Dla kazdego toru liczy:

- sredni blad,
- maksymalny blad,
- progress,
- srednia predkosc.

Po walidacji robi:

```python
model.train()
```

bo na czas walidacji model byl przelaczony na `eval()`.

### `travelled_progress`

Liczy rzeczywisty postep po torze. Dla zamknietej petli obsluguje przeskok `s` z konca na poczatek.

Uzywane do metryki `progress`.

### `train`

Glowne kroki:

1. sprawdza argumenty,
2. ustawia liczbe watkow,
3. generuje dataset,
4. tworzy model,
5. ewentualnie wczytuje model do douczenia,
6. trenuje przez zadana liczbe epok,
7. waliduje model,
8. zapisuje najlepsze wagi.

```python
optimizer = torch.optim.AdamW(...)
```

AdamW to stabilny optymalizator dla sieci neuronowych. `weight_decay` lekko ogranicza rozrost wag.

```python
loss_fn = nn.MSELoss()
```

Siec uczy sie napiec eksperta, wiec naturalnym bledem jest sredni blad kwadratowy.

```python
pred = torch.tanh(model(xb))
```

Tanh ogranicza wyjscie do `[-1, 1]`, tak jak cele treningowe.

```python
loss.backward()
optimizer.step()
```

Klasyczny krok uczenia:

1. licz gradient,
2. popraw wagi.

### Wybieranie najlepszego modelu

Model nie jest wybierany tylko po `loss`. Jest wybierany po:

```python
score = mean_error + max_error_weight * max_error + progress_error_weight * progress_error
```

Czyli dobry model ma:

- maly blad sredni,
- maly blad maksymalny,
- postep blisko oczekiwanego.

To jest bardzo wazne, bo sam `loss` moze malec, a robot w zamknietej petli nie musi jechac lepiej.

### `main`

Definiuje argumenty terminala. Najwazniejsze:

```text
--samples
    liczba probek treningowych

--sequence-length
    dlugosc sekwencji dla GRU

--epochs
    liczba epok

--batch
    rozmiar batcha

--lr
    learning rate

--min-cruise-speed / --max-cruise-speed
    zakres predkosci treningowych

--slow-replay-fraction
    procent danych wolnej jazdy

--eval-cruise-speeds
    predkosci walidacyjne

--workers
    procesy do generowania danych

--torch-threads
    watki PyTorch
```

## 15. Jak odpowiadac na pytania obronne

### Dlaczego nie PI/PID?

Bo celem projektu jest sterowanie neuronowe. PI/PID moglby byc warstwa pomocnicza, ale w finalnej wersji siec zwraca bezposrednio napiecia kol. Dzieki temu sprawdzamy, czy NN potrafi nauczyc sie polityki sterowania.

### Dlaczego ekspert, skoro sterowanie ma byc neuronowe?

Ekspert jest uzywany tylko do uczenia. W czasie jazdy nie jest uzywany. To podejscie nazywa sie uczeniem przez imitacje.

### Dlaczego GRU?

Bo robot jest ukladem dynamicznym. Aktualna decyzja zalezy od poprzednich stanow. GRU ma pamiec i jest prostsze od LSTM.

### Dlaczego normalizacja wejsc?

Siec uczy sie stabilniej, gdy cechy maja podobne zakresy. Bez normalizacji cechy o duzych wartosciach moglyby dominowac gradient.

### Dlaczego `tanh` na wyjsciu?

Bo napiecie jest ograniczone. `tanh` daje zakres `[-1, 1]`, a po pomnozeniu przez `12 V` dostajemy fizyczny zakres napiec.

### Dlaczego RK4?

Bo model ma dynamike silnikow i platformy. RK4 jest dokladniejsze od Eulera przy tym samym kroku czasowym.

### Dlaczego tor jest splajnem?

Bo robot ma sledzic nieregularna, ale gladka sciezke. Catmull-Rom daje gladki zamkniety tor bez dodatkowych bibliotek.

### Dlaczego progress moze byc mniejszy niz 1?

Bo robot moze jechac wolniej niz oczekiwany profil, np. z powodu ograniczenia napiecia albo zakretow. `progress = 1` oznacza nadazanie za profilem.

### Dlaczego model czasem zwalnia na zakretach?

Bo predkosc maksymalna na zakrecie jest ograniczona przez przyspieszenie boczne i napiecie. Gdyby robot jechal za szybko, blad od toru wzroslby.

### Dlaczego po okrazeniu resetowana jest pamiec GRU?

Bo zmienia sie sciezka. Ukryty stan GRU zawiera kontekst poprzedniej trasy, wiec po zmianie toru moglby przeszkadzac.

## 16. Minimalne komendy

Animacja:

```powershell
python animate_path_nn.py
```

Trening:

```powershell
python train_direct_path_nn.py --resume direct_path_nn.pt --output direct_path_nn.pt --samples 1536000 --sequence-length 64 --epochs 250 --batch 64 --lr 0.00018 --min-cruise-speed 0.24 --max-cruise-speed 0.68 --slow-replay-fraction 0.35 --slow-cruise-speed-min 0.12 --slow-cruise-speed-max 0.24 --eval-cruise-speeds 0.16,0.35,0.55,0.65 --eval-every 1 --eval-paths 5 --eval-duration 30 --max-error-weight 0.70 --progress-error-weight 0.70
```

## 17. Co jest najwazniejsze w projekcie

Najwazniejsze punkty do obrony:

1. Robot jest modelem roznicowym z dynamika silnikow DC.
2. Wejscie sterujace to napiecia prawego i lewego kola.
3. Siec GRU generuje napiecia bezposrednio.
4. Wejscie sieci to stan robota i lokalny podglad sciezki.
5. Uczenie odbywa sie przez imitacje eksperta.
6. Walidacja odbywa sie w zamknietej petli, nie tylko po MSE.
7. Predkosc jest ograniczana na zakretach przez profil zalezy od krzywizny.
8. Animacja pokazuje rzeczywista petle: NN -> model robota -> nowy stan.

## 18. Krotka wersja odpowiedzi "co robi projekt?"

Projekt symuluje robota mobilnego roznicowego, ktory sledzi zamknieta sciezke typu spline. Sterowanie jest realizowane przez siec neuronowa GRU, ktora na podstawie stanu robota i lokalnego podgladu toru wyznacza napiecia dla prawego i lewego kola. Model robota uwzglednia kinematyke, dynamike platformy, silniki DC, SEM, opory i ograniczenie napiecia. Siec jest uczona nadzorowanie na danych generowanych przez eksperta i oceniana w zamknietej petli przez blad od sciezki, postep i predkosc. Animacja pokazuje dzialanie w czasie rzeczywistym oraz zmienia tor po wykonaniu okrazenia.
