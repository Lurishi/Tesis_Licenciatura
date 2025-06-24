'''
Author: Lara Sofía Barreiro
Date: 24/06/2025
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#%matplotlib qt5 # Descomentar si se quiere usar una ventana gráfica interactiva
# import time  # Comentado ya que no se usa actualmente

custom_params = {

    'font.family': 'serif',
}

plt.rcParams.update(custom_params)
#%%
def test_gaussiano(a:float,med:np.ndarray,sigma:float):
    '''
    Test de hipótesis gaussiano para detectar píxeles con media significativamente menor a 1.
    
    Hipótesis:
    H₀: μ = 1 (hipótesis nula)
    H₁: μ < 1 (hipótesis alternativa, test de cola izquierda)
    
    Necesita:
    numpy, scipy.stats.norm
    -------
    a: float
        Nivel de significancia (típicamente 0.05 o 0.01)
    med: np.ndarray
        Matriz de datos con forma (N, M, samples) donde N y M son las dimensiones de la 
        imagen y samples es el número de muestras por píxel.
    sigma: float
        Error experimental del sensor (σ experimental para una única sample), debe ser un 
        número positivo.
    Returns
    -------
    med_p: np.ndarray (N, M, samples)
        Matriz de datos procesados con valores hasta el punto de parada
    mu_at_stop: np.ndarray (N, M)
        Matriz de medias μ calculadas en la iteración donde se detuvo el test
    p_value: np.ndarray (N, M, samples)
        Matriz de valores p-value del test de hipótesis para cada muestra acumulativa
    samples: np.ndarray (N, M)
        Matriz de conteos de samples utilizados antes de la detección (o máximo si no se detectó)
    mask: np.ndarray (N, M, samples)
        Máscara booleana que indica si el p-value es menor que el nivel de significancia
    -------
    '''
    # Verificaciones de entrada
    if not isinstance(med, np.ndarray):
        raise TypeError("med debe ser un numpy.ndarray")
    
    if med.ndim != 3:
        raise ValueError(f"med debe tener 3 dimensiones (N, M, samples), pero tiene {med.ndim}")
    
    if not isinstance(a, (int, float)) or not (0 < a < 1):
        raise ValueError("a (nivel de significancia) debe ser un número entre 0 y 1")
    
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("sigma debe ser un número positivo")
    
    if med.shape[-1] < 2:
        raise ValueError("Se necesitan al menos 2 muestras para realizar el test")
    
    # Conversión de tipos y extracción de dimensiones
    sigma = np.float64(sigma)
    s = med.shape[-1]
    N, M = med.shape[:2]
    alpha = np.float64(a)
    # Cálculo del estadístico de prueba y el p-value
    # μ_n = (1/n) * Σ(x_i) para cada n = 1, 2, ..., s
    mu = np.cumsum(med, axis=-1) / np.arange(1, s + 1)
    
    # Error estándar para cada tamaño de muestra: σ/√n
    scales = sigma / np.sqrt(np.arange(1, s + 1))
    
    # P-value para test de cola izquierda: P(Z ≤ (μ_n - 1)/(σ/√n))
    # donde Z ~ N(0,1) bajo H₀: μ = 1
    p_value = norm.cdf(mu, loc=1, scale=scales[None, None, :])

    # Aplicación del criterio de parada
    mask = p_value < alpha
    first_pass = np.argmax(mask, axis=-1)  # Índice del primer p-value < α
    found = np.any(mask, axis=-1)  # Verificar si se encontró alguna detección
    
    # Si no se detectó significancia, usar todas las muestras disponibles
    counts = np.where(found, first_pass, s-1)  # Índices de 0 a s-1 
   
    # Obtener la media en el punto de parada para cada píxel
    mu_at_stop = mu[np.arange(N)[:, None], np.arange(M), counts]

    # Construcción de la matriz resultante
    med_p = np.full_like(med, np.nan)

    # Condición especial: si μ < 0, asignar ceros a esos píxeles
    mask_neg = mu_at_stop < 0
    med_p[mask_neg, :] = 0
    
    # Para μ ≥ 0, copiar los valores originales hasta el punto de parada
    for l in range(s):
        mask_l = (counts >= l) & (~mask_neg)
        med_p[mask_l, l] = med[mask_l, l]
    
    # Número de muestras utilizadas (counts + 1 porque counts es 0-indexado)
    samples = counts + 1
    
    return med_p, mu_at_stop, p_value, samples, mask
