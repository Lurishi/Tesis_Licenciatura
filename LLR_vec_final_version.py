#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#%matplotlib qt5 # Descomentar si se quiere usar una ventana gráfica interactiva
import time

custom_params = {

    'font.family': 'serif',
}

plt.rcParams.update(custom_params)
#%%
def test_z(a:float,med:np.ndarray,sigma:float):
    '''
    Necesita:
    numpy, scipy.stats.norm
    -------
    a: float
        Nivel de significancia
    med: np.ndarray
        Matriz de datos con forma (N, N, samples)
    sigma: float
        Error experimental del sensor
    Returns
    -------
    med_p: np.ndarray
        Matriz de datos con número de samples 
        con valores de med que cumplen la condición de p-value < a
    z: np.ndarray
        Matriz de valores z calculados a partir del estadístico q0
    p_value: np.ndarray
        Matriz de valores p-value calculados a partir del estadístico q0
    counts: np.ndarray
        Matriz que devuelve el número correspondiente al primer sample que cumple
        la condición (p_value < a) | (p_value > 0.5 - np.finfo(float).eps), donde 
        p_value es el valor de p calculado a partir del estadístico z0.
    q0: np.ndarray
        Matriz del estadístico q0 calculado a partir de los datos med.
    mask: np.ndarray
        Máscara booleana que indica dónde se cumplen las condiciones de p-value, 
        sin importar cuando se cumple la condición por primera vez.
    -------
    Nota:
    La máscara de rechazo incluye dos zonas:
        p_value < alpha: rechazo clásico de hipótesis nula.
        p_value ≈ 0.5: corresponde a casos donde mu_hat es negativo, lo que en este 
        contexto experimental también se considera zona de rechazo. Como no podemos 
        poner explicitamente p_value == 0.5, ya que podría pasar que por errores de 
        mantiza no fuera ese el valor exacto, se utiliza un valor muy cercano a 0.5
        usando np.finfo(float).eps para evitar problemas de precisión.
    '''
    samples = med.shape[-1]
    # Cálculo del estadístico y el p-value
    mu_hat = np.cumsum(med, axis = -1) / np.arange(1,samples+ 1)
    logL_null = np.cumsum(np.log(norm.pdf(med, loc = 1, scale = sigma)), axis = -1)
    logL_hat = np.zeros_like(med)
    
    for l in range(samples):    
        mu = mu_hat[:, :, l][:, :, None]
        slice_l = med[:, :, :l+1]
        logpdfs = np.log(norm.pdf(slice_l, loc = mu, scale = sigma))
        logL_hat[:, :, l] = np.sum(logpdfs, axis = -1)
    
    
    # Calculo del estadístico q0, z0 y el p-value 
    q0_ = -2 * (logL_null-logL_hat)
    q0 = np.where(mu_hat > 0,q0_,0 )
    z = np.sqrt(q0)
    p_value = (1 - (norm.cdf(z,loc = 0, scale = 1))) 

    #Máscara  
    alpha = a
    mask = (p_value< alpha) | (p_value > 0.5 - np.finfo(float).eps) # Maskeo en dos lugares
    first_pass = np.argmax(mask, axis=-1)  # Busca el primer True en el eje de samples
    found = np.any(mask, axis=-1)  # Busca si hay al menos un True la máscara anterior para cada píxel      
    counts = np.where(found, first_pass, samples-1) # Devuelve el índice del primer True, caso contrario, impone samples-1
    med_p = np.full_like(med, np.nan) # Shape: (N, N, samples)
                             
    for l in range(samples):
        mask_l = counts >= l # Solo copia donde l =< count            
        med_p[mask_l, l] = med[mask_l, l]  # Copia los valores originales donde se cumple la condición

    counts = counts + 1  # Incrementa en 1 para que el primer sample sea 1 en lugar de 0
    return med_p, z, p_value,counts,q0,mask
# %%
