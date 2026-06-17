"""
Сравнение методов компенсации возмущений при управлении манипулятором,
переносящим открытый сосуд с жидким реагентом (slosh-аффектируемая нагрузка).

Методы: PID, PID + наблюдатель возмущений (DOB), метод скользящих режимов (SMC).

Возмущения:
  1) эксплуатационные — механический удар, сенсорный шум, вибрации оборудования
     (внешние, не зависящие от движения манипулятора);
  2) расплёскивание жидкого реагента — самовозбуждаемое возмущение, возникающее
     из-за ускорения второго звена (того, что несёт сосуд) при движении по
     заданной траектории.

Все метрики приводятся по второму суставу (q2), так как именно он несёт
технологическую нагрузку (сосуд с реагентом).
"""

import numpy as np
import matplotlib.pyplot as plt


# ─── Параметры манипулятора ───────────────────────────────────────────────────

ROBOT = {
    "m1": 1.5,   # масса первого звена [кг]
    "m2": 1.0,   # масса второго звена (без груза) [кг]
    "l1": 0.5,   # длина первого звена [м]
    "l2": 0.4,   # длина второго звена [м]
    "lc1": 0.25, # расстояние до центра масс, звено 1 [м]
    "lc2": 0.20, # расстояние до центра масс, звено 2 [м]
    "I1": 0.10,  # момент инерции, звено 1 [кг·м²]
    "I2": 0.05,  # момент инерции, звено 2 [кг·м²]
    "g":  9.81,  # ускорение свободного падения [м/с²]
}

# ─── Параметры технологической операции: перенос сосуда с реагентом ──────────
#
# Объём 500 мл выбран как реалистичная нагрузка для лабораторного манипулятора:
# отношение массы жидкости к собственной массе звеньев (0.5/2.5 = 20%)
# сопоставимо с отношением payload/масса у промышленных коботов
# (UR3 — 27%, UR5 — 27%, ABB IRB120 — 12%), то есть является технически
# реализуемой нагрузкой, а не превышающей грузоподъёмность манипулятора.

PAYLOAD = {
    "m_liquid":   0.50,    # масса жидкого реагента [кг] (500 мл воды)
    "r_vessel":   0.0425,  # внутренний радиус сосуда [м] (лабораторный стакан 500 мл)
    "h_liquid":   0.088,   # высота столба жидкости [м]: V/(пи*r^2) = 8.8 см —
                            # физически согласовано с объёмом и радиусом сосуда
    "slosh_damp": 0.08,    # коэффициент демпфирования (вязкое затухание)
}

# Собственная частота первой (антисимметричной) моды расплёскивания жидкости
# в цилиндрическом сосуде — стандартная формула гидродинамики свободной
# поверхности (Abramson, 1966):
#     omega_n = sqrt(1.84 * g / r_vessel * tanh(1.84 * h_liquid / r_vessel))
PAYLOAD["slosh_freq"] = np.sqrt(
    1.84 * ROBOT["g"] / PAYLOAD["r_vessel"]
    * np.tanh(1.84 * PAYLOAD["h_liquid"] / PAYLOAD["r_vessel"])
)

# Эквивалентная маятниковая модель расплёскивания (slosh-as-pendulum model):
# жидкость представлена эквивалентным маятником массой m_liquid, подвешенным
# в точке крепления сосуда к эффектору. При угловом ускорении второго звена
# манипулятора маятник отклоняется на угол theta_slosh и создаёт реактивный
# момент на сустав — это и есть возмущение, самовозбуждаемое движением робота.


# ─── Динамика манипулятора (метод Лагранжа) ───────────────────────────────────

def compute_dynamics(q, dq):
    """
    Возвращает матрицы динамики Лагранжа: M(q), C(q,dq), G(q) для
    манипулятора, несущего сосуд с жидким реагентом (масса жидкости
    учтена в эффективной массе второго звена).
    """
    p = ROBOT
    m2_eff = p["m2"] + PAYLOAD["m_liquid"]

    q1, q2   = q
    dq1, dq2 = dq

    M = np.array([
        [p["m1"]*p["lc1"]**2 + p["I1"]
         + m2_eff*(p["l1"]**2 + p["lc2"]**2 + 2*p["l1"]*p["lc2"]*np.cos(q2))
         + p["I2"],
         m2_eff*(p["lc2"]**2 + p["l1"]*p["lc2"]*np.cos(q2)) + p["I2"]],
        [m2_eff*(p["lc2"]**2 + p["l1"]*p["lc2"]*np.cos(q2)) + p["I2"],
         m2_eff*p["lc2"]**2 + p["I2"]],
    ])

    h = -m2_eff * p["l1"] * p["lc2"] * np.sin(q2)
    C = np.array([
        [h * dq2,  h * (dq1 + dq2)],
        [-h * dq1, 0.0],
    ])

    G = np.array([
        (p["m1"]*p["lc1"] + m2_eff*p["l1"]) * p["g"] * np.cos(q1)
        + m2_eff * p["lc2"] * p["g"] * np.cos(q1 + q2),
        m2_eff * p["lc2"] * p["g"] * np.cos(q1 + q2),
    ])

    return M, C, G


# ─── Динамика расплёскивания жидкости (slosh dynamics) ───────────────────────

def slosh_step(theta, dtheta, ddq2_excitation, dt):
    """
    Один шаг интегрирования угла отклонения жидкости (эквивалентного
    маятника), возбуждаемого угловым ускорением второго звена ddq2:

        theta'' + 2*zeta*omega_n*theta' + omega_n^2*theta = -ddq2

    Возвращает обновлённые (theta, dtheta) и реактивный момент на сустав:

        d_slosh = m_liquid * g * r_vessel * sin(theta)
    """
    omega_n = PAYLOAD["slosh_freq"]
    zeta    = PAYLOAD["slosh_damp"]
    m_l     = PAYLOAD["m_liquid"]
    r_v     = PAYLOAD["r_vessel"]
    g       = ROBOT["g"]

    ddtheta = -2*zeta*omega_n*dtheta - omega_n**2*theta - ddq2_excitation
    dtheta_new = dtheta + ddtheta * dt
    theta_new  = theta + dtheta_new * dt

    d_slosh = m_l * g * r_v * np.sin(theta_new)
    return theta_new, dtheta_new, d_slosh


# ─── Желаемая траектория переноса сосуда ──────────────────────────────────────

def desired_trajectory(t):
    """
    Технологическая траектория переноса сосуда с реагентом между двумя
    позициями (забор / дозирование). Профиль плавный (синусоидальный),
    что соответствует типичному требованию при переносе жидких сред —
    минимизировать инерционное возбуждение расплёскивания.
    """
    q_d   = np.array([ 0.500 * np.sin(0.8 * t),  0.300 * np.sin(1.2 * t)])
    dq_d  = np.array([ 0.400 * np.cos(0.8 * t),  0.360 * np.cos(1.2 * t)])
    ddq_d = np.array([-0.320 * np.sin(0.8 * t), -0.432 * np.sin(1.2 * t)])
    return q_d, dq_d, ddq_d


# ─── Внешние эксплуатационные возмущения ──────────────────────────────────────

def get_disturbance(t, kind):
    """
    Эксплуатационные возмущения, характерные для лабораторной среды:

    'mechanical'   — кратковременный механический удар (контакт с оборудованием)
    'sensor_noise' — высокочастотный шум измерительной системы манипулятора
    'vibration'    — вибрации смежного оборудования (центрифуги, насосы)
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

PID_Kp = np.array([30.0, 25.0])
PID_Kd = np.array([ 7.0,  5.0])
PID_Ki = np.array([ 1.5,  1.2])

ROBUST_Kp = np.array([60.0, 50.0])
ROBUST_Kd = np.array([14.0, 11.0])
ROBUST_Ki = np.array([ 3.0,  2.5])

SMC_lambda = np.array([8.0, 7.0])
# K_sw[1] и phi подобраны так, чтобы переключающий член не возбуждал
# резонанс расплёскивания жидкости (см. примечание в отчёте о настройке):
# слишком агрессивное переключение по q2 накачивает энергией маятник жидкости.
SMC_K_sw = np.array([6.5, 1.5])
SMC_phi  = 0.30

DOB_BANDWIDTH = 25.0


def pid_torque(e, de, integral):
    """Момент ПИД-регулятора."""
    return PID_Kp * e + PID_Kd * de + PID_Ki * integral


def pid_dob_torque(e, de, integral, d_hat):
    """Момент ПИД-регулятора с компенсацией оценённого возмущения."""
    return ROBUST_Kp * e + ROBUST_Kd * de + ROBUST_Ki * integral - d_hat


def smc_torque(e, de, q, dq, ddq_d):
    """Момент по методу скользящих режимов."""
    M, C, G = compute_dynamics(q, dq)
    s       = de + SMC_lambda * e
    tau_ff  = M @ (ddq_d + SMC_lambda * de) + C @ dq + G
    return tau_ff + ROBUST_Kp * e + ROBUST_Kd * de - SMC_K_sw * np.tanh(s / SMC_phi)


def update_dob(d_hat, M, C, G, dq, tau_prev, ddq_prev, dt):
    """Обновление оценки наблюдателя возмущений (DOB)."""
    d_raw = M @ ddq_prev - tau_prev + C @ dq + G
    return d_hat + DOB_BANDWIDTH * (d_raw - d_hat) * dt


# ─── Симуляция ────────────────────────────────────────────────────────────────

def simulate(controller, disturbance_kind, T=8.0, dt=0.002):
    """
    Запускает симуляцию переноса сосуда с реагентом манипулятором с
    заданным регулятором под действием эксплуатационного возмущения
    И возмущения от расплёскивания жидкости (возбуждается собственным
    движением манипулятора, действует всегда, независимо от сценария).
    """
    time_steps = np.linspace(0, T, int(T / dt))

    q        = np.zeros(2)
    dq       = np.zeros(2)
    integral = np.zeros(2)
    d_hat    = np.zeros(2)
    tau_prev = np.zeros(2)
    ddq_prev = np.zeros(2)
    theta_slosh, dtheta_slosh = 0.0, 0.0

    q_log, q_d_log, error_log, slosh_log = [], [], [], []

    for t in time_steps:
        q_d, dq_d, ddq_d = desired_trajectory(t)
        e  = q_d - q
        de = dq_d - dq

        d_env = get_disturbance(t, disturbance_kind)

        # Расплёскивание возбуждается фактическим ускорением второго сустава
        # на предыдущем шаге
        theta_slosh, dtheta_slosh, d_slosh_val = slosh_step(
            theta_slosh, dtheta_slosh, ddq_prev[1], dt
        )

        d = d_env + np.array([0.0, d_slosh_val])

        M, C, G = compute_dynamics(q, dq)

        if controller == "PID":
            integral += e * dt
            tau = pid_torque(e, de, integral)

        elif controller == "PID+DOB":
            integral += e * dt
            d_hat = update_dob(d_hat, M, C, G, dq, tau_prev, ddq_prev, dt)
            tau   = pid_dob_torque(e, de, integral, d_hat)

        elif controller == "SMC":
            tau = smc_torque(e, de, q, dq, ddq_d)

        ddq = np.linalg.solve(M, tau + d - C @ dq - G)
        ddq = np.clip(ddq, -50, 50)
        dq += ddq * dt
        q  += dq  * dt

        tau_prev = tau.copy()
        ddq_prev = ddq.copy()

        q_log.append(q.copy())
        q_d_log.append(q_d.copy())
        error_log.append(e.copy())
        slosh_log.append(theta_slosh)

    return (
        time_steps,
        np.array(q_log),
        np.array(q_d_log),
        np.array(error_log),
        np.array(slosh_log),
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

# Метрики приводятся по второму суставу (JOINT) — он несёт сосуд с реагентом
JOINT = 1  # индекс: 0 = q1, 1 = q2

print("Запуск симуляций переноса сосуда с реагентом (500 мл)...")
results = {}
for dist_key in DISTURBANCES:
    results[dist_key] = {}
    for ctrl in CONTROLLERS:
        t, q, q_d, error, slosh = simulate(ctrl, dist_key)
        results[dist_key][ctrl] = {
            "t":     t,
            "q":     q,
            "q_d":   q_d,
            "error": error,
            "slosh": slosh,
            "rmse":  np.sqrt(np.mean(error**2, axis=0)),
        }
        print(f"  {ctrl:8s} | {DISTURBANCES[dist_key]}")

print("\nСимуляции завершены.\n")


# ─── Графики ──────────────────────────────────────────────────────────────────

def plot_tracking():
    """
    График слежения за траекторией второго сустава (q2) — несущего сосуд
    с реагентом — под действием эксплуатационных возмущений и
    расплёскивания жидкости.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.patch.set_facecolor("white")

    for ax, (dist_key, dist_label) in zip(axes, DISTURBANCES.items()):
        t = results[dist_key]["PID"]["t"]
        ax.plot(t, results[dist_key]["PID"]["q_d"][:, JOINT],
                "k--", lw=1.8, alpha=0.6, label="Желаемая")
        for ctrl in CONTROLLERS:
            ax.plot(t, results[dist_key][ctrl]["q"][:, JOINT],
                    color=COLORS[ctrl], lw=1.8, label=ctrl)
        ax.set_title(dist_label, fontsize=13, fontweight="bold", color="#1a1a2e", pad=6)
        ax.set_ylabel("q₂ [рад]", fontsize=11)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.set_xlim(0, 8)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#f8f9ff")

    axes[-1].set_xlabel("Время [с]", fontsize=11)
    plt.tight_layout(pad=1.5)
    plt.savefig("plot_tracking.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Сохранён: plot_tracking.png")


def plot_rmse():
    """Столбчатая диаграмма RMSE по q2 для всех методов и возмущений."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    x      = np.arange(len(DISTURBANCES))
    width  = 0.25
    labels = ["Механическое\nвозмущение", "Сенсорный\nшум", "Вибрации\nоборудования"]

    for i, ctrl in enumerate(CONTROLLERS):
        values = [results[dk][ctrl]["rmse"][JOINT] for dk in DISTURBANCES]
        bars   = ax.bar(x + i * width, values, width,
                        label=ctrl, color=COLORS[ctrl], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("RMSE по q₂ [рад]", fontsize=12)
    ax.set_title("Точность слежения сустава, несущего сосуд с реагентом (RMSE)",
                 fontsize=13, fontweight="bold", color="#1a1a2e")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#f8f9ff")

    plt.tight_layout()
    plt.savefig("plot_rmse.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Сохранён: plot_rmse.png")


def plot_slosh():
    """График угла расплёскивания реагента в сосуде для каждого метода."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("white")

    for ax, (dist_key, dist_label) in zip(axes, DISTURBANCES.items()):
        t = results[dist_key]["PID"]["t"]
        for ctrl in CONTROLLERS:
            ax.plot(t, np.degrees(results[dist_key][ctrl]["slosh"]),
                    color=COLORS[ctrl], lw=1.6, label=ctrl)
        ax.set_title(dist_label, fontsize=10.5, fontweight="bold", color="#1a1a2e")
        ax.set_ylabel("Угол расплёскивания [°]", fontsize=10)
        ax.set_xlabel("Время [с]", fontsize=10)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 8)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("#f8f9ff")

    plt.tight_layout(pad=1.5)
    plt.savefig("plot_slosh.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Сохранён: plot_slosh.png")


def print_summary_table():
    """Сводная таблица RMSE по q2 (сустав, несущий сосуд с реагентом)."""
    print(f"\n{'Метод':<12} {'Механическое':>15} {'Сенс. шум':>12} {'Вибрации':>12}")
    print("-" * 54)
    for ctrl in CONTROLLERS:
        row = [f"{results[dk][ctrl]['rmse'][JOINT]:.4f}" for dk in DISTURBANCES]
        print(f"{ctrl:<12} {row[0]:>15} {row[1]:>12} {row[2]:>12}")


# ─── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_tracking()
    plot_rmse()
    plot_slosh()
    print_summary_table()