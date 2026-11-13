"""
Actividad formativa 3 - Implementación y evaluación de filtros digitales
=========================================================================

Diseña y evalúa filtros digitales pasa bajos, pasa altos y pasa banda con
SciPy, usando tres métodos distintos para cada tipo:

    * IIR Butterworth      (respuesta plana en la banda de paso)
    * IIR Chebyshev tipo I (transición más abrupta, con rizado en la banda de paso)
    * FIR con ventana Hamming (fase lineal, siempre estable)

Estructura del programa (la misma que pide la actividad):
    1. Definición de la señal de entrada
    2. Diseño de los filtros
    3. Aplicación de los filtros a la señal
    4. Visualización y evaluación de los resultados
    5. Aplicación real: limpieza de una señal de ECG

Uso:
    python filtros_digitales.py

Salidas:
    figuras/*.png            gráficas para el informe y la presentación
    resultados/metricas.json métricas numéricas (SNR, ganancias en los tonos)
    resultados/metricas.csv  las mismas métricas en formato tabla
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: guarda las figuras en archivo
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Estilo de las gráficas (paleta de alto contraste, rejilla discreta)
# ---------------------------------------------------------------------------
C_BUTTER = "#2a78d6"   # azul   -> Butterworth
C_CHEBY = "#eb6834"    # naranja -> Chebyshev I
C_FIR = "#1baf7a"      # aqua   -> FIR con ventana
COLORES = [C_BUTTER, C_CHEBY, C_FIR]
C_RAW = "#898781"      # gris   -> señal sin filtrar
C_INK = "#0b0b0b"      # tinta  -> referencia ideal
C_INK2 = "#52514e"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": C_SURF, "axes.facecolor": C_SURF,
    "savefig.facecolor": C_SURF, "savefig.dpi": 160,
    "axes.edgecolor": C_AXIS, "axes.labelcolor": C_INK2,
    "axes.titlecolor": C_INK, "axes.titlesize": 11, "axes.titlelocation": "left",
    "xtick.color": C_INK2, "ytick.color": C_INK2, "text.color": C_INK,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "font.size": 10, "legend.frameon": False,
    "legend.fontsize": 9,
})

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_FIG = os.path.join(DIR_BASE, "figuras")
DIR_RES = os.path.join(DIR_BASE, "resultados")

# ===========================================================================
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ===========================================================================
FS = 1000.0          # frecuencia de muestreo [Hz]  (Nyquist = 500 Hz)
DURACION = 4.0       # [s]
t = np.arange(0, DURACION, 1 / FS)
rng = np.random.default_rng(42)  # semilla fija: resultados reproducibles

# Señal compuesta por tres tonos: baja, media y alta frecuencia
TONOS = {5: 1.0, 50: 0.8, 150: 0.6}          # {frecuencia [Hz]: amplitud}
componentes = {f0: a * np.sin(2 * np.pi * f0 * t) for f0, a in TONOS.items()}
x_limpia = sum(componentes.values())

# Ruido blanco gaussiano aditivo (desviación estándar 0.5)
SIGMA_RUIDO = 0.5
x = x_limpia + SIGMA_RUIDO * rng.standard_normal(t.size)

# Para cada tipo de filtro: qué tono queremos conservar, banda de paso nominal
OBJETIVO = {"Pasa bajos": 5, "Pasa altos": 150, "Pasa banda": 50}
REGION_PASO = {"Pasa bajos": (0, 20), "Pasa altos": (100, FS / 2), "Pasa banda": (40, 60)}
# Ventana temporal que se grafica (unos cuantos ciclos del tono objetivo)
VENTANAS = {"Pasa bajos": (1.0, 1.6), "Pasa altos": (1.0, 1.04), "Pasa banda": (1.0, 1.12)}
ARCHIVOS = {"Pasa bajos": "fig05_pasa_bajos.png", "Pasa altos": "fig06_pasa_altos.png",
            "Pasa banda": "fig07_pasa_banda.png"}


# ===========================================================================
# 2. DISEÑO DE LOS FILTROS
# ===========================================================================
def disenar_filtros():
    """Devuelve {tipo: {método: (clase, coeficientes)}}.

    clase = "sos"  -> IIR en secciones de segundo orden (numéricamente estable)
    clase = "fir"  -> coeficientes b de un filtro FIR
    """
    return {
        # --- Pasa bajos: fc = 20 Hz (conserva 5 Hz, elimina 50 y 150 Hz) ---
        "Pasa bajos": {
            "Butterworth (N=4)": ("sos", signal.butter(4, 20, "lowpass", fs=FS, output="sos")),
            "Chebyshev I (N=4, 1 dB)": ("sos", signal.cheby1(4, 1, 20, "lowpass", fs=FS, output="sos")),
            "FIR Hamming (201 coef.)": ("fir", signal.firwin(201, 20, window="hamming", fs=FS)),
        },
        # --- Pasa altos: fc = 100 Hz (conserva 150 Hz, elimina 5 y 50 Hz) ---
        "Pasa altos": {
            "Butterworth (N=4)": ("sos", signal.butter(4, 100, "highpass", fs=FS, output="sos")),
            "Chebyshev I (N=4, 1 dB)": ("sos", signal.cheby1(4, 1, 100, "highpass", fs=FS, output="sos")),
            "FIR Hamming (201 coef.)": ("fir", signal.firwin(201, 100, window="hamming", pass_zero=False, fs=FS)),
        },
        # --- Pasa banda: 40-60 Hz (conserva 50 Hz, elimina 5 y 150 Hz) ---
        # La banda es estrecha, así que el FIR necesita más coeficientes
        # para lograr una transición angosta (ancho ~ 3.3*fs/N con Hamming).
        "Pasa banda": {
            "Butterworth (N=4)": ("sos", signal.butter(4, [40, 60], "bandpass", fs=FS, output="sos")),
            "Chebyshev I (N=4, 1 dB)": ("sos", signal.cheby1(4, 1, [40, 60], "bandpass", fs=FS, output="sos")),
            "FIR Hamming (401 coef.)": ("fir", signal.firwin(401, [40, 60], window="hamming", pass_zero=False, fs=FS)),
        },
    }


def respuesta(clase, coef, n=4096):
    """Respuesta en frecuencia de un filtro: (frecuencias [Hz], H complejo)."""
    if clase == "sos":
        return signal.sosfreqz(coef, worN=n, fs=FS)
    return signal.freqz(coef, worN=n, fs=FS)


def ganancia_db(clase, coef, f0):
    """Ganancia de diseño (una pasada) del filtro en la frecuencia f0, en dB."""
    f, h = respuesta(clase, coef, n=16384)
    return 20 * np.log10(np.interp(f0, f, np.abs(h)) + 1e-12)


# ===========================================================================
# 3. APLICACIÓN DE LOS FILTROS A LA SEÑAL
# ===========================================================================
def aplicar(clase, coef, senal):
    """Filtrado de fase cero (hacia adelante y hacia atrás).

    Con filtfilt la señal de salida queda alineada con la de entrada (sin
    retardo), por lo que se puede comparar punto a punto. El costo es que la
    respuesta efectiva es |H(f)|^2 (el doble de atenuación en dB).
    """
    if clase == "sos":
        return signal.sosfiltfilt(coef, senal)
    return signal.filtfilt(coef, [1.0], senal)


def snr_db(y, ref, recorte=0.1):
    """Relación señal-ruido (dB) de y respecto a la referencia ideal ref.

    Se descarta el 10 % inicial y final para evitar transitorios de borde.
    """
    n = int(len(y) * recorte)
    s = slice(n, len(y) - n)
    error = y[s] - ref[s]
    return 10 * np.log10(np.mean(ref[s] ** 2) / np.mean(error ** 2))


# ===========================================================================
# 4. VISUALIZACIÓN
# ===========================================================================
def espectro_db(senal):
    """Espectro de amplitud (dB) con ventana Hann. Un tono de amplitud A se ve en 20*log10(A)."""
    w = np.hanning(len(senal))
    X = np.fft.rfft(senal * w)
    f = np.fft.rfftfreq(len(senal), 1 / FS)
    amp = 2 * np.abs(X) / np.sum(w)
    return f, 20 * np.log10(amp + 1e-12)


def guardar(fig, nombre):
    fig.savefig(os.path.join(DIR_FIG, nombre), bbox_inches="tight")
    plt.close(fig)


def fig_senal_entrada():
    fig, axs = plt.subplots(2, 1, figsize=(10, 7.2))
    m = t < 0.5
    ax = axs[0]
    ax.plot(t[m], x[m], color=C_RAW, lw=1.0, label="Señal con ruido")
    ax.plot(t[m], x_limpia[m], color=C_INK, lw=1.6, label="Señal limpia (5 + 50 + 150 Hz)")
    ax.set(title="Señal de entrada en el dominio del tiempo", xlabel="Tiempo [s]", ylabel="Amplitud")
    ax.legend(loc="upper right", ncol=2)

    ax = axs[1]
    f, X = espectro_db(x)
    ax.plot(f, X, color=C_RAW, lw=1.0, label="Espectro de la señal con ruido")
    for f0, a in TONOS.items():
        ax.annotate(f"{f0} Hz", xy=(f0, 20 * np.log10(a)), xytext=(f0 + 8, 20 * np.log10(a) + 6),
                    color=C_INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=C_INK2, lw=0.8))
    ax.axhline(20 * np.log10(0.0194), color=C_INK2, lw=0.8, ls=":")
    ax.text(300, -31, "piso de ruido ≈ -34 dB", color=C_INK2, fontsize=9)
    ax.set(title="Espectro de amplitud: tres tonos sobre un piso de ruido blanco",
           xlabel="Frecuencia [Hz]", ylabel="Amplitud [dB]", xlim=(0, 500), ylim=(-80, 10))
    ax.legend(loc="upper right")
    fig.tight_layout()
    guardar(fig, "fig01_senal_entrada.png")


def fig_ruido_blanco(filtros):
    """Ruido blanco puro antes y después de cada filtro: muestra cómo cada filtro 'da forma' al ruido."""
    w = rng.standard_normal(int(40 * FS))
    f, p = signal.welch(w, fs=FS, nperseg=1024)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (tipo, banco) in zip(axs, filtros.items()):
        ax.plot(f, 10 * np.log10(p), color=C_RAW, label="Ruido blanco original")
        for (nombre, (clase, coef)), col in zip(banco.items(), COLORES):
            fy, py = signal.welch(aplicar(clase, coef, w), fs=FS, nperseg=1024)
            ax.plot(fy, 10 * np.log10(py + 1e-20), color=col, lw=1.3, label=nombre)
        ax.set(title=tipo, xlabel="Frecuencia [Hz]", xlim=(0, 500), ylim=(-140, -15))
    axs[0].set_ylabel("Densidad espectral de potencia [dB/Hz]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    guardar(fig, "fig02_ruido_blanco_filtrado.png")


def fig_respuesta_frecuencia(filtros):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (tipo, banco) in zip(axs, filtros.items()):
        for (nombre, (clase, coef)), col in zip(banco.items(), COLORES):
            f, h = respuesta(clase, coef)
            ax.plot(f, 20 * np.log10(np.abs(h) + 1e-12), color=col, label=nombre)
        ax.axhline(-3, color=C_INK2, lw=0.8, ls=":")
        for f0 in TONOS:
            ax.axvline(f0, color=C_INK2, lw=0.7, ls="--", alpha=0.6)
            ax.text(f0 + 2, 1, f"{f0} Hz", color=C_INK2, fontsize=8, va="bottom")
        ax.set(title=f"{tipo}: magnitud", xlabel="Frecuencia [Hz]", xlim=(0, 300), ylim=(-100, 8))
    axs[0].set_ylabel("Ganancia [dB]  (una pasada)")
    axs[0].text(295, -2.2, "-3 dB", color=C_INK2, fontsize=8, ha="right", va="top")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    guardar(fig, "fig03_respuesta_frecuencia.png")


def fig_retardo_grupo(filtros):
    """Retardo de grupo: el FIR simétrico tiene fase lineal (retardo constante); el IIR no."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (tipo, banco) in zip(axs, filtros.items()):
        for (nombre, (clase, coef)), col in zip(banco.items(), COLORES):
            if clase == "fir":
                # Un FIR simétrico de N coeficientes retarda exactamente (N-1)/2 muestras
                ax.axhline((len(coef) - 1) / 2 / FS * 1000, color=col, label=nombre)
            else:
                f, h = respuesta(clase, coef, n=8192)
                fase = np.unwrap(np.angle(h))
                gd = -np.gradient(fase, 2 * np.pi * f) * 1000  # [ms]
                ax.plot(f, gd, color=col, label=nombre)
        ax.set(title=f"{tipo}: retardo de grupo", xlabel="Frecuencia [Hz]", xlim=(0, 300), ylim=(0, 260))
    axs[0].set_ylabel("Retardo de grupo [ms]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    guardar(fig, "fig04_retardo_grupo.png")


def fig_por_tipo(tipo, objetivo, y_dict, snr_dict):
    """Antes (fila superior) y después (fila inferior) del filtrado, en tiempo y en frecuencia."""
    t0, t1 = VENTANAS[tipo]
    m = (t >= t0) & (t < t1)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
    ylim_t = (-3.8, 3.8)

    # (a) tiempo, antes
    ax = axs[0, 0]
    ax.plot(t[m], x[m], color=C_RAW, lw=1.0, label=f"Con ruido (SNR {snr_dict['antes']:.1f} dB)")
    ax.plot(t[m], componentes[objetivo][m], color=C_INK, lw=1.6, ls="--", label=f"Componente objetivo ({objetivo} Hz)")
    ax.set(title="Antes: dominio del tiempo", ylabel="Amplitud", ylim=ylim_t)
    ax.legend(loc="upper right", fontsize=8)

    # (b) espectro, antes
    ax = axs[0, 1]
    f, X = espectro_db(x)
    ax.plot(f, X, color=C_RAW, lw=1.0, label="Sin filtrar")
    ax.set(title="Antes: dominio de la frecuencia", ylabel="Amplitud [dB]", xlim=(0, 500), ylim=(-100, 10))
    ax.legend(loc="upper right", fontsize=8)

    # (c) tiempo, después
    ax = axs[1, 0]
    ax.plot(t[m], componentes[objetivo][m], color=C_INK, lw=1.4, ls="--", label="Componente objetivo")
    for (nombre, y), col in zip(y_dict.items(), COLORES):
        ax.plot(t[m], y[m], color=col, lw=1.5, label=f"{nombre} (SNR {snr_dict[nombre]:.1f} dB)")
    ax.set(title="Después: dominio del tiempo", xlabel="Tiempo [s]", ylabel="Amplitud", ylim=ylim_t)
    ax.legend(loc="upper right", fontsize=8)

    # (d) espectro, después
    ax = axs[1, 1]
    ax.plot(f, X, color=C_RAW, lw=0.8, alpha=0.6, label="Sin filtrar")
    for (nombre, y), col in zip(y_dict.items(), COLORES):
        fy, Y = espectro_db(y)
        ax.plot(fy, Y, color=col, lw=1.2, label=nombre)
    lo, hi = REGION_PASO[tipo]
    ax.axvspan(lo, hi, color=C_GRID, alpha=0.7, lw=0)
    ax.set(title="Después: dominio de la frecuencia (zona gris = banda de paso)",
           xlabel="Frecuencia [Hz]", ylabel="Amplitud [dB]", xlim=(0, 500), ylim=(-100, 10))
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{tipo}: señal antes y después del filtrado (filtrado de fase cero)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    guardar(fig, ARCHIVOS[tipo])


def fig_snr(resultados_snr, filtros):
    """Barras agrupadas: SNR antes y después para cada tipo y método."""
    tipos = list(filtros.keys())
    metodos = ["Butterworth", "Chebyshev I", "FIR Hamming"]
    colores = [C_RAW] + COLORES
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ancho = 0.2
    xs = np.arange(len(tipos))
    series = ["Sin filtrar"] + metodos
    for k, (serie, col) in enumerate(zip(series, colores)):
        vals = []
        for tipo in tipos:
            d = resultados_snr[tipo]
            if k == 0:
                vals.append(d["antes"])
            else:
                clave = [n for n in d if n.startswith(metodos[k - 1])][0]
                vals.append(d[clave])
        barras = ax.bar(xs + (k - 1.5) * ancho, vals, ancho * 0.9, color=col, label=serie,
                        edgecolor=C_SURF, linewidth=2)
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.4 if v >= 0 else -0.4), f"{v:.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=C_INK2)
    ax.axhline(0, color=C_AXIS, lw=1)
    ax.set_xticks(xs, tipos)
    ax.set(ylabel="SNR respecto a la componente objetivo [dB]",
           title="Mejora de la relación señal-ruido con cada filtro")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-12, 26)
    fig.tight_layout()
    guardar(fig, "fig08_snr_comparacion.png")


# ===========================================================================
# 5. APLICACIÓN REAL: LIMPIEZA DE UNA SEÑAL DE ECG
# ===========================================================================
def ecg_sintetico(tiempo, fc_latido=75 / 60):
    """ECG sintético: cada latido es la suma de 5 gaussianas (ondas P, Q, R, S y T)."""
    ecg = np.zeros_like(tiempo)
    periodo = 1 / fc_latido
    # (desfase respecto al pico R [s], amplitud [mV], ancho [s])
    ondas = [(-0.20, 0.12, 0.025), (-0.04, -0.10, 0.010), (0.0, 1.00, 0.012),
             (0.04, -0.20, 0.012), (0.25, 0.30, 0.040)]
    for k0 in np.arange(0.5, tiempo[-1] + periodo, periodo):
        for dt, a, s in ondas:
            ecg += a * np.exp(-0.5 * ((tiempo - (k0 + dt)) / s) ** 2)
    return ecg


def aplicacion_ecg():
    fs = 500.0
    te = np.arange(0, 20, 1 / fs)
    ecg = ecg_sintetico(te)

    # Contaminación típica de un electrocardiograma:
    deriva = 0.6 * np.sin(2 * np.pi * 0.3 * te) + 0.2 * np.sin(2 * np.pi * 0.1 * te + 1)  # respiración / movimiento
    red = 0.25 * np.sin(2 * np.pi * 60 * te)                                             # interferencia de la red (60 Hz en México)
    emg = 0.05 * rng.standard_normal(te.size)                                            # ruido muscular de banda ancha
    ecg_ruidoso = ecg + deriva + red + emg

    # Pasa altos 0.5 Hz (quita deriva) en cascada con pasa bajos 40 Hz (quita 60 Hz y EMG)
    sos_hp = signal.butter(2, 0.5, "highpass", fs=fs, output="sos")
    sos_lp = signal.butter(4, 40, "lowpass", fs=fs, output="sos")
    ecg_filtrado = signal.sosfiltfilt(sos_lp, signal.sosfiltfilt(sos_hp, ecg_ruidoso))

    # El pasa altos elimina por diseño el nivel DC; como el ECG es una señal
    # acoplada en AC, el SNR se calcula quitando la media a todas las señales.
    ecg_ac = ecg - np.mean(ecg)
    n = int(len(te) * 0.05)
    s = slice(n, len(te) - n)

    def snr_ac(y):
        yc = y[s] - np.mean(y[s])
        return 10 * np.log10(np.mean(ecg_ac[s] ** 2) / np.mean((yc - ecg_ac[s]) ** 2))

    snr_antes = snr_ac(ecg_ruidoso)
    snr_despues = snr_ac(ecg_filtrado)
    # Conservación del pico R: amplitud promedio de la señal filtrada en los
    # instantes de los latidos, dividida entre la ideal (1.0 = sin distorsión)
    idx_r = np.round(np.arange(0.5, te[-1] - 0.5, 1 / (75 / 60)) * fs).astype(int)
    idx_r = idx_r[(idx_r > n) & (idx_r < len(te) - n)]
    ratio_r = float(np.mean(ecg_filtrado[idx_r]) / np.mean(ecg_ac[idx_r]))

    # --- Gráficas ---
    m = (te >= 2) & (te < 8)
    fig, axs = plt.subplots(3, 1, figsize=(11, 11))
    ax = axs[0]
    ax.plot(te[m], ecg_ruidoso[m], color=C_RAW, lw=1.0, label=f"ECG con deriva, 60 Hz y ruido (SNR {snr_antes:.1f} dB)")
    ax.plot(te[m], ecg[m], color=C_INK, lw=1.2, ls="--", label="ECG ideal")
    ax.set(title="Antes: ECG contaminado", ylabel="Amplitud [mV]", ylim=(-1.3, 2.4))
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[1]
    ax.plot(te[m], ecg_ac[m], color=C_INK, lw=1.2, ls="--", label="ECG ideal (sin nivel DC)")
    ax.plot(te[m], ecg_filtrado[m], color=C_BUTTER, lw=1.5,
            label=f"Filtrado: pasa altos 0.5 Hz + pasa bajos 40 Hz (SNR {snr_despues:.1f} dB)")
    ax.set(title="Después: ECG filtrado", xlabel="Tiempo [s]", ylabel="Amplitud [mV]", ylim=(-0.45, 1.5))
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[2]
    f, p0 = signal.welch(ecg_ruidoso, fs=fs, nperseg=4096)
    f, p1 = signal.welch(ecg_filtrado, fs=fs, nperseg=4096)
    ax.plot(f, 10 * np.log10(p0 + 1e-20), color=C_RAW, label="Antes")
    ax.plot(f, 10 * np.log10(p1 + 1e-20), color=C_BUTTER, label="Después")
    ax.annotate("deriva (0.3 Hz)", xy=(0.3, 10 * np.log10(p0[np.argmin(abs(f - 0.3))])),
                xytext=(0.5, 5), fontsize=9, arrowprops=dict(arrowstyle="-", color=C_INK2, lw=0.8))
    ax.annotate("red eléctrica (60 Hz)", xy=(60, 10 * np.log10(p0[np.argmin(abs(f - 60))])),
                xytext=(80, -10), fontsize=9, arrowprops=dict(arrowstyle="-", color=C_INK2, lw=0.8))
    ax.axvspan(0.5, 40, color=C_GRID, alpha=0.7, lw=0)
    ax.set_xscale("log")
    ax.set(title="Densidad espectral de potencia (zona gris = banda conservada 0.5-40 Hz)",
           xlabel="Frecuencia [Hz]", ylabel="PSD [dB/Hz]", xlim=(0.1, 250), ylim=(-110, 20))
    ax.legend(loc="lower left")
    fig.tight_layout()
    guardar(fig, "fig09_aplicacion_ecg.png")

    return {"snr_antes_db": float(snr_antes), "snr_despues_db": float(snr_despues),
            "conservacion_pico_R": ratio_r,
            "ganancia_60Hz_efectiva_db": float(2 * 20 * np.log10(abs(
                signal.sosfreqz(sos_lp, worN=[60], fs=fs)[1][0])))}


# ===========================================================================
# PROGRAMA PRINCIPAL
# ===========================================================================
def main():
    os.makedirs(DIR_FIG, exist_ok=True)
    os.makedirs(DIR_RES, exist_ok=True)

    filtros = disenar_filtros()

    # Aplicar todos los filtros y calcular métricas
    resultados_snr, ganancias, salidas = {}, {}, {}
    for tipo, banco in filtros.items():
        obj = OBJETIVO[tipo]
        ref = componentes[obj]
        resultados_snr[tipo] = {"antes": float(snr_db(x, ref))}
        ganancias[tipo], salidas[tipo] = {}, {}
        for nombre, (clase, coef) in banco.items():
            y = aplicar(clase, coef, x)
            salidas[tipo][nombre] = y
            resultados_snr[tipo][nombre] = float(snr_db(y, ref))
            ganancias[tipo][nombre] = {
                f"{f0} Hz": float(ganancia_db(clase, coef, f0)) for f0 in TONOS}

    # Gráficas
    fig_senal_entrada()
    fig_ruido_blanco(filtros)
    fig_respuesta_frecuencia(filtros)
    fig_retardo_grupo(filtros)
    for tipo in filtros:
        fig_por_tipo(tipo, OBJETIVO[tipo], salidas[tipo], resultados_snr[tipo])
    fig_snr(resultados_snr, filtros)
    ecg = aplicacion_ecg()

    # Guardar métricas
    resultados = {
        "parametros": {"fs_hz": FS, "duracion_s": DURACION, "tonos_hz_amplitud": TONOS,
                       "sigma_ruido": SIGMA_RUIDO},
        "snr_db": resultados_snr,
        "ganancia_diseno_db": ganancias,
        "ecg": ecg,
    }
    with open(os.path.join(DIR_RES, "metricas.json"), "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(DIR_RES, "metricas.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tipo", "metodo", "snr_antes_db", "snr_despues_db", "mejora_db",
                    "ganancia_5Hz_db", "ganancia_50Hz_db", "ganancia_150Hz_db"])
        for tipo in filtros:
            for nombre in filtros[tipo]:
                a, d = resultados_snr[tipo]["antes"], resultados_snr[tipo][nombre]
                g = ganancias[tipo][nombre]
                w.writerow([tipo, nombre, f"{a:.2f}", f"{d:.2f}", f"{d - a:.2f}",
                            f"{g['5 Hz']:.1f}", f"{g['50 Hz']:.1f}", f"{g['150 Hz']:.1f}"])

    # Resumen en consola
    print(f"{'Filtro':<12}{'Método':<28}{'SNR antes':>10}{'SNR después':>13}"
          f"{'5 Hz':>9}{'50 Hz':>9}{'150 Hz':>9}")
    for tipo in filtros:
        for nombre in filtros[tipo]:
            g = ganancias[tipo][nombre]
            print(f"{tipo:<12}{nombre:<28}{resultados_snr[tipo]['antes']:>10.2f}"
                  f"{resultados_snr[tipo][nombre]:>13.2f}"
                  f"{g['5 Hz']:>9.1f}{g['50 Hz']:>9.1f}{g['150 Hz']:>9.1f}")
    print("\nECG:", json.dumps(ecg, indent=2))


if __name__ == "__main__":
    main()
