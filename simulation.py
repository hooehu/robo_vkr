"""
Сравнение методов робастного управления 2-DOF манипулятором.

Методы: PID, PID + наблюдатель возмущений (DOB), метод скользящих режимов (SMC).
Возмущения: механический удар, сенсорный шум, вибрации оборудования.
"""

import numpy as np
import matplotlib.pyplot as plt


# ─── Параметры робота ─────────────────────────────────────────────────────────

ROBOT = {
    "m1": 1.5,   # масса первого звена [кг]
    "m2": 1.0,   # масса второго звена [кг]
    "l1": 0.5,   # длина первого звена [м]
    "l2": 0.4,   # длина второго звена [м]
    "lc1": 0.25, # расстояние до центра масс, звено 1 [м]
    "lc2": 0.20, # расстояние до центра масс, звено 2 [м]
    "I1": 0.10,  # момент инерции, звено 1 [кг·м²]
    "I2": 0.05,  # момент инерции, звено 2 [кг·м²]
    "g":  9.81,  # ускорение свободного падения [м/с²]
}


# ─── Динамика манипулятора ────────────────────────────────────────────────────

def compute_dynamics(q, dq):
    """
    Возвращает матрицы динамики Лагранжа: M(q), C(q,dq), G(q).

    Уравнение движения: M·q̈ + C·q̇ + G = τ + d

    Параметры
    ---------
    q  : углы суставов [рад]
    dq : угловые скорости [рад/с]

    Возвращает
    ----------
    M : матрица инерции (2×2)
    C : матрица Кориолиса/центробежных сил (2×2)
    G : вектор гравитационных моментов (2,)
    """
    p = ROBOT
    q1, q2   = q
    dq1, dq2 = dq

    # Матрица инерции
    M = np.array([
        [p["m1"]*p["lc1"]**2 + p["I1"]
         + p["m2"]*(p["l1"]**2 + p["lc2"]**2 + 2*p["l1"]*p["lc2"]*np.cos(q2))
         + p["I2"],
         p["m2"]*(p["lc2"]**2 + p["l1"]*p["lc2"]*np.cos(q2)) + p["I2"]],
        [p["m2"]*(p["lc2"]**2 + p["l1"]*p["lc2"]*np.cos(q2)) + p["I2"],
         p["m2"]*p["lc2"]**2 + p["I2"]],
    ])

    # Матрица Кориолиса
    h = -p["m2"] * p["l1"] * p["lc2"] * np.sin(q2)
    C = np.array([
        [h * dq2,  h * (dq1 + dq2)],
        [-h * dq1, 0.0],
    ])

    # Гравитационный вектор
    G = np.array([
        (p["m1"]*p["lc1"] + p["m2"]*p["l1"]) * p["g"] * np.cos(q1)
        + p["m2"] * p["lc2"] * p["g"] * np.cos(q1 + q2),
        p["m2"] * p["lc2"] * p["g"] * np.cos(q1 + q2),
    ])

    return M, C, G


# ─── Желаемая траектория ──────────────────────────────────────────────────────

def desired_trajectory(t):
    """
    Синусоидальная эталонная траектория для обоих суставов.

    Возвращает позицию, скорость и ускорение в момент времени t.
    """
    q_d   = np.array([ 0.500 * np.sin(0.8 * t),  0.300 * np.sin(1.2 * t)])
    dq_d  = np.array([ 0.400 * np.cos(0.8 * t),  0.360 * np.cos(1.2 * t)])
    ddq_d = np.array([-0.320 * np.sin(0.8 * t), -0.432 * np.sin(1.2 * t)])
    return q_d, dq_d, ddq_d


# ─── Внешние возмущения ───────────────────────────────────────────────────────

def get_disturbance(t, kind):
    """
    Возвращает вектор внешнего возмущения d(t) [Н·м] для момента времени t.

    Виды возмущений
    ---------------
    'mechanical'   — кратковременный механический удар при t = 3 с
    'sensor_noise' — высокочастотный шум измерений
    'vibration'    — синусоидальные вибрации лабораторного оборудования
    """
    if kind == "mechanical":
        if 3.0 <= t <= 3.12:
            return np.array([6.0, 4.0])
        return np.zeros(2)

    if kind == "sensor_noise":
        return 2.0 * np.array([
            np.sin(18*t) + 0.4 * np.cos(30*t),
            np.cos(20*t) + 0.3 * np.sin(28*t),
        ])

    if kind == "vibration":
        return np.array([
            1.8 * np.sin(7*t) + 0.7 * np.sin(14*t),
            1.2 * np.sin(7*t + 0.4),
        ])

    return np.zeros(2)


# ─── Регуляторы ───────────────────────────────────────────────────────────────

# Коэффициенты PID (слабые — чтобы разница с другими методами была заметна)
PID_Kp = np.array([30.0, 25.0])
PID_Kd = np.array([ 7.0,  5.0])
PID_Ki = np.array([ 1.5,  1.2])

# Коэффициенты PID+DOB и SMC (сильнее)
ROBUST_Kp = np.array([60.0, 50.0])
ROBUST_Kd = np.array([14.0, 11.0])
ROBUST_Ki = np.array([ 3.0,  2.5])

# Параметры SMC
SMC_lambda = np.array([8.0, 7.0])  # коэффициент поверхности скольжения
SMC_K_sw   = np.array([5.0, 4.0])  # амплитуда переключающего члена
SMC_phi    = 0.05                  # ширина граничного слоя (tanh)

# Полоса пропускания фильтра DOB [рад/с]
DOB_BANDWIDTH = 25.0


def pid_torque(e, de, integral):
    """Момент ПИД-регулятора."""
    return PID_Kp * e + PID_Kd * de + PID_Ki * integral


def pid_dob_torque(e, de, integral, d_hat):
    """Момент ПИД-регулятора с компенсацией оценённого возмущения."""
    return ROBUST_Kp * e + ROBUST_Kd * de + ROBUST_Ki * integral - d_hat


def smc_torque(e, de, q, dq, ddq_d):
    """
    Момент по методу скользящих режимов.

    Поверхность скольжения: s = ė + λ·e
    Закон управления: τ = τ_ff + Kp·e + Kd·ė − K_sw·tanh(s/φ)
    """
    M, C, G = compute_dynamics(q, dq)
    s       = de + SMC_lambda * e
    tau_ff  = M @ (ddq_d + SMC_lambda * de) + C @ dq + G
    return tau_ff + ROBUST_Kp * e + ROBUST_Kd * de - SMC_K_sw * np.tanh(s / SMC_phi)


def update_dob(d_hat, M, C, G, dq, tau_prev, ddq_prev, dt):
    """
    Обновление оценки наблюдателя возмущений (DOB).

    Оценка строится через обратную динамику и фильтр низких частот:
        d_raw = M·q̈ − τ + C·q̇ + G
        d̂˙   = ω · (d_raw − d̂)
    """
    d_raw = M @ ddq_prev - tau_prev + C @ dq + G
    return d_hat + DOB_BANDWIDTH * (d_raw - d_hat) * dt


# ─── Симуляция ────────────────────────────────────────────────────────────────

def simulate(controller, disturbance_kind, T=8.0, dt=0.002):
    """
    Запускает симуляцию манипулятора с заданным регулятором и возмущением.

    Параметры
    ---------
    controller       : 'PID', 'PID+DOB' или 'SMC'
    disturbance_kind : 'mechanical', 'sensor_noise' или 'vibration'
    T                : длительность симуляции [с]
    dt               : шаг интегрирования [с]

    Возвращает
    ----------
    t     : вектор времени
    q     : реальные углы суставов
    q_d   : желаемые углы суставов
    error : ошибка слежения (q_d − q)
    """
    time_steps = np.linspace(0, T, int(T / dt))

    # Начальное состояние
    q        = np.zeros(2)
    dq       = np.zeros(2)
    integral = np.zeros(2)
    d_hat    = np.zeros(2)
    tau_prev = np.zeros(2)
    ddq_prev = np.zeros(2)

    # Логи
    q_log     = []
    q_d_log   = []
    error_log = []

    for t in time_steps:
        q_d, dq_d, ddq_d = desired_trajectory(t)
        e  = q_d - q
        de = dq_d - dq
        d  = get_disturbance(t, disturbance_kind)

        M, C, G = compute_dynamics(q, dq)

        # Вычисление управляющего момента
        if controller == "PID":
            integral += e * dt
            tau = pid_torque(e, de, integral)

        elif controller == "PID+DOB":
            integral += e * dt
            d_hat = update_dob(d_hat, M, C, G, dq, tau_prev, ddq_prev, dt)
            tau   = pid_dob_torque(e, de, integral, d_hat)

        elif controller == "SMC":
            tau = smc_torque(e, de, q, dq, ddq_d)

        # Интегрирование динамики (метод Эйлера)
        ddq = np.linalg.solve(M, tau + d - C @ dq - G)
        ddq = np.clip(ddq, -50, 50)  # защита от численного взрыва
        dq += ddq * dt
        q  += dq  * dt

        # Сохраняем для следующего шага DOB
        tau_prev = tau.copy()
        ddq_prev = ddq.copy()

        q_log.append(q.copy())
        q_d_log.append(q_d.copy())
        error_log.append(e.copy())

    return (
        time_steps,
        np.array(q_log),
        np.array(q_d_log),
        np.array(error_log),
    )


# ─── Запуск всех комбинаций ───────────────────────────────────────────────────

CONTROLLERS = ["PID", "PID+DOB", "SMC"]

DISTURBANCES = {
    "mechanical":   "Механическое возмущение (удар)",
    "sensor_noise": "Сенсорный шум",
    "vibration":    "Вибрации оборудования",
}

COLORS = {
    "PID":     "#E53935",
    "PID+DOB": "#1E88E5",
    "SMC":     "#43A047",
}

print("Запуск симуляций...")
results = {}
for dist_key in DISTURBANCES:
    results[dist_key] = {}
    for ctrl in CONTROLLERS:
        t, q, q_d, error = simulate(ctrl, dist_key)
        results[dist_key][ctrl] = {
            "t":     t,
            "q":     q,
            "q_d":   q_d,
            "error": error,
            "rmse":  np.sqrt(np.mean(error**2, axis=0)),
        }
        print(f"  {ctrl:8s} | {DISTURBANCES[dist_key]}")

print("\nСимуляции завершены.\n")


# ─── Графики ──────────────────────────────────────────────────────────────────

def plot_tracking():
    """График слежения за траекторией по суставу q₁."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.patch.set_facecolor("white")

    for ax, (dist_key, dist_label) in zip(axes, DISTURBANCES.items()):
        t = results[dist_key]["PID"]["t"]

        ax.plot(t, results[dist_key]["PID"]["q_d"][:, 0],
                "k--", lw=1.8, alpha=0.6, label="Желаемая")

        for ctrl in CONTROLLERS:
            ax.plot(t, results[dist_key][ctrl]["q"][:, 0],
                    color=COLORS[ctrl], lw=1.8, label=ctrl)

        ax.set_title(dist_label, fontsize=13, fontweight="bold",
                     color="#1a1a2e", pad=6)
        ax.set_ylabel("q₁ [рад]", fontsize=11)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.set_xlim(0, 8)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#f8f9ff")

    axes[-1].set_xlabel("Время [с]", fontsize=11)
    plt.tight_layout(pad=1.5)
    plt.savefig("plot_tracking.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Сохранён: plot_tracking.png")


def plot_errors():
    """График абсолютной ошибки слежения по суставу q₁."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("white")

    for ax, (dist_key, dist_label) in zip(axes, DISTURBANCES.items()):
        t = results[dist_key]["PID"]["t"]
        for ctrl in CONTROLLERS:
            ax.plot(t, np.abs(results[dist_key][ctrl]["error"][:, 0]),
                    color=COLORS[ctrl], lw=1.6, label=ctrl)

        ax.set_title(dist_label, fontsize=10.5, fontweight="bold",
                     color="#1a1a2e")
        ax.set_ylabel("|e₁| [рад]", fontsize=10)
        ax.set_xlabel("Время [с]", fontsize=10)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#f8f9ff")

    plt.tight_layout(pad=1.5)
    plt.savefig("plot_errors.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Сохранён: plot_errors.png")


def plot_rmse():
    """Столбчатая диаграмма RMSE по всем методам и возмущениям."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    x      = np.arange(len(DISTURBANCES))
    width  = 0.25
    labels = ["Механическое\nвозмущение", "Сенсорный\nшум",
              "Вибрации\nоборудования"]

    for i, ctrl in enumerate(CONTROLLERS):
        values = [results[dk][ctrl]["rmse"][0] for dk in DISTURBANCES]
        bars   = ax.bar(x + i * width, values, width,
                        label=ctrl, color=COLORS[ctrl],
                        alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("RMSE по q₁ [рад]", fontsize=12)
    ax.set_title("Сравнение точности методов управления (RMSE)",
                 fontsize=13, fontweight="bold", color="#1a1a2e")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#f8f9ff")

    plt.tight_layout()
    plt.savefig("plot_rmse.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Сохранён: plot_rmse.png")


def print_rmse_table():
    """Вывод сводной таблицы RMSE в консоль."""
    print(f"\n{'Метод':<12} {'Механическое':>15} {'Сенс. шум':>12} {'Вибрации':>12}")
    print("-" * 54)
    for ctrl in CONTROLLERS:
        row = [f"{results[dk][ctrl]['rmse'][0]:.4f}" for dk in DISTURBANCES]
        print(f"{ctrl:<12} {row[0]:>15} {row[1]:>12} {row[2]:>12}")


# ─── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_tracking()
    plot_errors()
    plot_rmse()
    print_rmse_table()
