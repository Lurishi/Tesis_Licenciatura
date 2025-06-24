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
def test_z(a:float,med:np.ndarray,sigma:float):
    '''
    Test de razón de verosimilitud (Likelihood Ratio Test) para detectar píxeles 
    con media significativamente diferente de 1.
    
    Hipótesis:
    H₀: μ = 1 (hipótesis nula)
    H₁: μ ≠ 1 (hipótesis alternativa, test de dos colas)
    
    Utiliza el estadístico de razón de verosimilitud:
    q₀ = -2 * ln(L(μ=1)/L(μ_hat) ≈ χ²(1) bajo H₀
    z_0 = √q₀
    
    Necesita:
    numpy, scipy.stats.norm
    -------
    a: float
        Nivel de significancia (típicamente 0.05 o 0.01)
    med: np.ndarray
        Matriz de datos con forma (N, M, samples) donde N y M son las dimensiones 
        de la imagen y samples es el número de muestras por píxel.
    sigma: float
        Error experimental del sensor (desviación estándar conocida)
    Returns
    -------
    med_p: np.ndarray (N, M, samples)
        Matriz de datos procesados con valores hasta el punto de parada
    z: np.ndarray (N, M, samples)
        Matriz de valores z = √q₀ calculados del estadístico de razón de verosimilitud
    p_value: np.ndarray (N, M, samples)
        Matriz de valores p-value calculados a partir del estadístico z (test de cola derecha)
    counts: np.ndarray (N, M)
        Matriz de conteos de samples utilizados antes de la detección (o máximo si no se detectó)
    q0: np.ndarray (N, M, samples)
        Matriz del estadístico de razón de verosimilitud q₀
    mask: np.ndarray (N, M, samples)
        Máscara booleana que indica dónde se cumplen las condiciones de rechazo
    -------
    Nota sobre la máscara de rechazo:
    La máscara incluye dos zonas de rechazo:
    1. p_value < α: Rechazo clásico cuando el estadístico es significativo
    2. p_value ≈ 0.5: Casos donde μ̂ < 0 (físicamente implausibles en este contexto)
       Se usa 0.5 - ε para evitar problemas de precisión numérica, donde μ̂ < 0 
       resulta en q₀ = 0 y por tanto z = 0, dando p_value = 0.5
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
    samples = med.shape[-1]
    N, M = med.shape[:2]
    alpha = np.float64(a)
    
    # Cálculo de la media muestral acumulativa
    # μ̂_n = (1/n) * Σ(x_i) para cada n = 1, 2, ..., samples
    mu_hat = np.cumsum(med, axis=-1) / np.arange(1, samples + 1)
    
    # Cálculo de la log-verosimilitud bajo H₀: μ = 1
    # L₀ = Π f(x_i; μ=1, σ) → log L₀ = Σ log f(x_i; μ=1, σ)
    logL_null = np.cumsum(np.log(norm.pdf(med, loc=1, scale=sigma)), axis=-1)
    
    # Inicialización para log-verosimilitud bajo H₁: μ = μ̂
    logL_hat = np.zeros_like(med)
    
    # Cálculo de la log-verosimilitud bajo H₁ para cada tamaño de muestra
    for l in range(samples):    
        mu = mu_hat[:, :, l][:, :, None]  # Media estimada hasta el sample l
        slice_l = med[:, :, :l+1]          # Datos hasta el sample l
        # L₁ = Π f(x_i; μ=μ̂, σ) → log L₁ = Σ log f(x_i; μ=μ̂, σ)
        logpdfs = np.log(norm.pdf(slice_l, loc=mu, scale=sigma))
        logL_hat[:, :, l] = np.sum(logpdfs, axis=-1)
    
    # Cálculo del estadístico de razón de verosimilitud
    # q₀ = -2 * ln(L₀/L₁) = -2 * (log L₀ - log L₁)
    q0_ = -2 * (logL_null - logL_hat)
    
    # Aplicar restricción: q₀ = 0 cuando μ̂ ≤ 0 (casos no físicos)
    q0 = np.where(mu_hat > 0, q0_, 0)
    
    # Transformación a estadístico z: z = √q₀
    z = np.sqrt(q0)
    
    # P-value para test de cola derecha: P(Z > z) = 1 - Φ(z)
    # donde Z ~ N(0,1) bajo H₀
    p_value = 1 - norm.cdf(z, loc=0, scale=1) 

    # Aplicación del criterio de parada con doble condición
    # Condición 1: p_value < α (rechazo estadístico clásico)
    # Condición 2: p_value ≈ 0.5 (casos con μ̂ < 0, físicamente implausibles)
    epsilon = np.finfo(float).eps
    mask = (p_value < alpha) | (p_value > 0.5 - epsilon)
    
    first_pass = np.argmax(mask, axis=-1)  # Índice del primer True en el eje samples
    found = np.any(mask, axis=-1)          # Verificar si hay al menos una detección
    
    # Si no se detectó significancia, usar todas las muestras disponibles
    counts = np.where(found, first_pass, samples-1)  # Índices de 0 a samples-1
    
    # Construcción de la matriz resultante
    med_p = np.full_like(med, np.nan)  # Inicializar con NaN
                             
    # Copiar valores originales hasta el punto de parada
    for l in range(samples):
        mask_l = counts >= l  # Incluir samples donde l <= count            
        med_p[mask_l, l] = med[mask_l, l]  # Preservar valores originales
    
    # Convertir counts a número de muestras (1-indexado)
    counts = counts + 1  # Incrementar para que el primer sample sea 1 en lugar de 0
    
    return med_p, z, p_value, counts, q0, mask
# %%
