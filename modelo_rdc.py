import cupy as cp # type: ignore
import numpy as np # type: ignore

############################## SPREAD INFECTION CON ELEMENTWISE KERNEL ###############################################

spread_kernel = cp.ElementwiseKernel(
    """
    float32 S, float32 I, float32 R, float32 dt, float32 d, float32 beta, float32 gamma, 
    float32 D, float32 wx, float32 wy, float32 h_dx, float32 h_dy, float32 A, float32 B,
    float32 I_top, float32 I_bottom, float32 I_left, float32 I_right
    """,
    """
    float32 S_new, float32 I_new, float32 R_new
    """,
    """
    S_new = S - dt * beta * I * S;

    float laplacian_I = (I_top + I_bottom + I_left + I_right - 4 * I);
    float s = D * dt / (d * d);

    float I_dx = ((A * wx + B * h_dx) > 0) ? (I - I_left) : (I_right - I);
    float I_dy = ((A * wy + B * h_dy) > 0) ? (I - I_bottom) : (I_top - I);

    I_new = I + dt * (beta * I * S - gamma * I) + s * laplacian_I 
            - dt / d * ((A * wx + B * h_dx) * I_dx + (A * wy + B * h_dy) * I_dy);

    R_new = R + dt * gamma * I;
    """,
    "spread_kernel"
)

############################## LLAMADO A ELEMENTWISE KERNEL ###############################################

def spread_infection(S, I, R, S_new, I_new, R_new, dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B):
    I_top = cp.roll(I, -1, axis=0)
    I_bottom = cp.roll(I, 1, axis=0)
    I_left = cp.roll(I, +1, axis=1)
    I_right = cp.roll(I, -1, axis=1)

    spread_kernel(S, I, R, dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B,
                  I_top, I_bottom, I_left, I_right,
                  S_new, I_new, R_new)

    # Aplicar condiciones de borde (esto se hace fuera del kernel)
    S_new[0, :] = S_new[-1, :] = S_new[:, 0] = S_new[:, -1] = 0
    I_new[0, :] = I_new[-1, :] = I_new[:, 0] = I_new[:, -1] = 0
    R_new[0, :] = R_new[-1, :] = R_new[:, 0] = R_new[:, -1] = 0

    # Celdas no combustibles
    no_fuel = (beta == 0)
    S_new[no_fuel] = 1
    I_new[no_fuel] = 0
    R_new[no_fuel] = 0

############################## SPREAD INFECTION CON RAW KERNEL ###############################################

kernel_code = r'''
extern "C" {

__global__ void spread_infection_kernel_raw(const float* S, const float* I, const float* R,
                                  float* S_new, float* I_new, float* R_new,
                                  const float* beta, const float* gamma,
                                  const float dt, const float d, const float D, 
                                  const float* wx, const float* wy,
                                  const float* h_dx, const float* h_dy,
                                  const float A, const float B,
                                  const int ny, const int nx, const int n_batch) 
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y
    int b = blockIdx.z;                             // batch_id

    if (i >= ny || j >= nx || b >= n_batch) return;

    int idx2D = i * nx + j;
    int idx = b * ny * nx + idx2D;

    if (i == 0 || i == ny-1 || j == 0 || j == nx-1) {
        S_new[idx] = 0.0f;
        I_new[idx] = 0.0f;
        R_new[idx] = 0.0f;
        return;
    }

    float S_val = S[idx];
    float I_val = I[idx];
    float R_val = R[idx];
    float beta_val = beta[idx];
    float gamma_val = gamma[idx];

    float I_top    = I[b * ny * nx + (i+1)*nx + j];
    float I_bottom = I[b * ny * nx + (i-1)*nx + j];
    float I_left   = I[b * ny * nx + i*nx + (j-1)];
    float I_right  = I[b * ny * nx + i*nx + (j+1)];

    float laplacian_I = I_top + I_bottom + I_left + I_right - 4.0f * I_val;
    float s = D * dt / (d * d);

    float adv_x = A * wx[idx] + B * h_dx[idx];
    float adv_y = A * wy[idx] + B * h_dy[idx];

    float I_dx = (adv_x > 0.0f) ? (I_val - I_left) : (I_right - I_val);
    float I_dy = (adv_y > 0.0f) ? (I_val - I_bottom) : (I_top - I_val);

    float S_temp = S_val - dt * beta_val * I_val * S_val;
    float I_temp = I_val + dt * (beta_val * I_val * S_val - gamma_val * I_val)
                   + s * laplacian_I - dt / d * (adv_x * I_dx + adv_y * I_dy);
    float R_temp = R_val + dt * gamma_val * I_val;

    if (beta_val == 0.0f) {
        S_temp = 1.0f;
        I_temp = 0.0f;
    }

    S_new[idx] = S_temp;
    I_new[idx] = I_temp;
    R_new[idx] = R_temp;
}
}
'''

mod = cp.RawModule(code=kernel_code)
spread_kernel_raw = mod.get_function('spread_infection_kernel_raw')

def spread_infection_raw(S, I, R, S_new, I_new, R_new, 
                                dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B):

    n_batch, ny, nx = S.shape
    threads = (16, 16)
    blocks_x = (nx + threads[0] - 1) // threads[0]
    blocks_y = (ny + threads[1] - 1) // threads[1]
    blocks = (blocks_x, blocks_y, n_batch)

    spread_kernel_raw(
        blocks, threads,
        (
            S.ravel(), I.ravel(), R.ravel(),
            S_new.ravel(), I_new.ravel(), R_new.ravel(),
            beta.ravel(), gamma.ravel(),
            cp.float32(dt), cp.float32(d), cp.float32(D),
            wx.ravel(), wy.ravel(),
            h_dx.ravel(), h_dy.ravel(),
            cp.float32(A), cp.float32(B),
            cp.int32(ny), cp.int32(nx), cp.int32(n_batch)
        )
    )

############################## SPREAD INFECTION CON CUPY ###############################################

def spread_infection_cupy(S, I, R, S_new, I_new, R_new, dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B):
    
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

    # Actualización del estado S
    S_new[...] = S - dt * beta * I * S

    # Actualización del estado I
    I_laplacian = (cp.roll(I, 1, axis=0) + cp.roll(I, -1, axis=0) +
                  cp.roll(I, 1, axis=1) + cp.roll(I, -1, axis=1) - 4 * I )

    # Número de difusión
    s = D*dt/(d*d)

    # Esquema upwind para la altura
    I_dx = ((A*wx+B*h_dx)>0)*(I - cp.roll(I, 1, axis=1)) + ((A*wx+B*h_dx)<0)*(cp.roll(I, -1, axis=1) - I)
    I_dy = ((A*wy+B*h_dy)>0)*(I - cp.roll(I, 1, axis=0)) + ((A*wy+B*h_dy)<0)*(cp.roll(I, -1, axis=0) - I)

    I_new[...] = I + dt * (beta * I * S - gamma * I) + s * I_laplacian - dt/d * ((A*wx+B*h_dx) * I_dx + (A*wy+B*h_dy) * I_dy)

    # Actualización del estado R
    R_new[...] = R + dt * gamma * I

    # Condiciones de borde de Dirichlet
    S_new[0, :] = S_new[-1, :] = S_new[:, 0] = S_new[:, -1] = 0
    I_new[0, :] = I_new[-1, :] = I_new[:, 0] = I_new[:, -1] = 0
    R_new[0, :] = R_new[-1, :] = R_new[:, 0] = R_new[:, -1] = 0

    # Creamos una máscara para las celdas no combustibles
    no_fuel = (beta == 0)

    S_new[no_fuel] = 1
    I_new[no_fuel] = 0
    R_new[no_fuel] = 0

############################## SPREAD INFECTION CON NUMPY ###############################################

def spread_infection_numpy(S, I, R, S_new, I_new, R_new, dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B):
    
    # Actualización del estado S
    S_new[...] = S - dt * beta * I * S

    # Actualización del estado I
    I_laplacian = (np.roll(I, 1, axis=0) + np.roll(I, -1, axis=0) +
                  np.roll(I, 1, axis=1) + np.roll(I, -1, axis=1) - 4 * I)

    # Número de difusión
    s = D * dt / (d * d)

    # Esquema upwind para la altura
    I_dx = ((A * wx + B * h_dx) > 0) * (I - np.roll(I, 1, axis=1)) + ((A * wx + B * h_dx) < 0) * (np.roll(I, -1, axis=1) - I)
    I_dy = ((A * wy + B * h_dy) > 0) * (I - np.roll(I, 1, axis=0)) + ((A * wy + B * h_dy) < 0) * (np.roll(I, -1, axis=0) - I)

    I_new[...] = I + dt * (beta * I * S - gamma * I) + s * I_laplacian - dt / d * ((A * wx + B * h_dx) * I_dx + (A * wy + B * h_dy) * I_dy)

    # Actualización del estado R
    R_new[...] = R + dt * gamma * I

    # Condiciones de borde de Dirichlet
    S_new[0, :] = S_new[-1, :] = S_new[:, 0] = S_new[:, -1] = 0
    I_new[0, :] = I_new[-1, :] = I_new[:, 0] = I_new[:, -1] = 0
    R_new[0, :] = R_new[-1, :] = R_new[:, 0] = R_new[:, -1] = 0

    # Celdas no combustibles
    no_fuel = (beta == 0)
    S_new[no_fuel] = 1
    I_new[no_fuel] = 0
    R_new[no_fuel] = 0

############################## CONDICIÓN DE COURANT ###############################################

def courant(dt, D, A, B, d, wx, wy, h_dx, h_dy):
    
    '''
    Verifica la condición de estabilidad de Courant para un modelo de difusión y advección.
    
    Args:
      dt (float): Intervalo de tiempo.
      D (float): Coeficiente de difusión.
      A (float): Coeficiente relacionado con la velocidad del viento.
      B (float): Coeficiente relacionado con la pendiente del terreno.
      d (float): Tamaño de la celda en el mapa.
      wx (float): Componente x de la velocidad del viento.
      wy (float): Componente y de la velocidad del viento.
      h_dx (float): Derivada parcial de la altura del terreno en la dirección x.
      h_dy (float): Derivada parcial de la altura del terreno en la dirección y.
    
    Returns:
      bool: True si la condición de estabilidad de Courant se cumple, False en caso contrario.
    '''

    # Courant para difusión
    courant_difusion = d**2 / (2 * D)
    
    # Courant para advección
    velocidad = cp.sqrt((A * wx + B * h_dx)**2 + (A * wy + B * h_dy)**2)
    courant_advectivo = d / (cp.sqrt(2) * cp.max(velocidad))

    # Retorna un booleano indicando si la condición de estabilidad se cumple
    return dt < cp.minimum(courant_difusion, courant_advectivo)