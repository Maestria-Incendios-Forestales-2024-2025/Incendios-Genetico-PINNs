import cupy as cp # type: ignore


def spread_infection(S, I, R, dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B):
    
    """

    La función actualiza el estado de una cuadrícula de vegetación susceptible (S), incendiándose (I) y quemada (R) 
    en función de varios parámetros que incluyen el tiempo, la velocidad del viento, la pendiente del terreno y 
    las constantes de propagación del fuego. Utiliza un esquema de diferencias finitas para modelar la difusión 
    del fuego y un esquema upwind para considerar la influencia del viento y la pendiente.

    Args:
      S: vegetación susceptible
      I: vegetación incendiándose
      R: vegetación quemada
      dt: paso temporal
      d: tamaño de celda
      beta: fracción de vegetación incéndiandose por hora
      gamma: fracción de vegetación incéndiandose por hora.
              1/gamma es el tiempo promedio del incendio
      wx: velocidad del viento en la dirección x
      wy: velocidad del viento en la dirección y
      h_dx: pendiente en la dirección x
      h_dy: pendiente en la dirección y
      A: constante adimensional que modifica la intensidad del viento
      B: constante con unidades de velocidad que modifica la intensidad de la altura

    Returns:
      S_new: nuevos árboles susceptibles
      I_new: nuevos árboles quemándose
      R_new: nuevos árboles quemados

    """
    S_new = S.copy()
    I_new = I.copy()
    R_new = R.copy()

    # Actualización del estado S
    S_new -= dt * beta * I * S

    # Actualización del estado I
    I_laplacian = (cp.roll(I, 1, axis=0) + cp.roll(I, -1, axis=0) +
                  cp.roll(I, 1, axis=1) + cp.roll(I, -1, axis=1) - 4 * I )

    # Número de difusión
    s = D*dt/(d*d)

    # Esquema upwind para la altura
    I_dx = ((A*wx+B*h_dx)>0)*(I - cp.roll(I, 1, axis=1)) + ((A*wx+B*h_dx)<0)*(cp.roll(I, -1, axis=1) - I)
    I_dy = ((A*wy+B*h_dy)>0)*(I - cp.roll(I, 1, axis=0)) + ((A*wy+B*h_dy)<0)*(cp.roll(I, -1, axis=0) - I)

    I_new += dt * (beta * I * S - gamma * I) + s * I_laplacian - dt/d * ((A*wx+B*h_dx) * I_dx + (A*wy+B*h_dy) * I_dy)

    # Actualización del estado R
    R_new += dt * gamma * I

    # Condiciones de borde de Dirichlet
    S_new[0, :] = S_new[-1, :] = S_new[:, 0] = S_new[:, -1] = 0
    I_new[0, :] = I_new[-1, :] = I_new[:, 0] = I_new[:, -1] = 0
    R_new[0, :] = R_new[-1, :] = R_new[:, 0] = R_new[:, -1] = 0

    # Creamos una máscara para las celdas no combustibles
    no_fuel = (beta == 0)

    S_new[no_fuel] = 1
    I_new[no_fuel] = 0
    R_new[no_fuel] = 0

    return S_new, I_new, R_new