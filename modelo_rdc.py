import cupy as cp # type: ignore

############################## SPREAD INFECTION CON RAW KERNEL ###############################################

kernel_code_adi = r'''
extern "C" {

__global__ void compute_rhs_y(const float* I, float* rhs,
                              const float* D, const float dt, const float d,
                              const int ny, const int nx, const int n_batch,
                              const float* vegetacion) {

    int j = blockIdx.x * blockDim.x + threadIdx.x; // x
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y
    int b = blockIdx.z;                            // n_batch

    if (i >= ny || j >= nx || b >= n_batch) return;

    int idx2d = i * nx + j;
    int idx = b * nx * ny + idx2d;

    // No computar celdas sin combustible
    if (vegetacion[idx2d] <= 2.0f) {
        rhs[idx] = 0.0f;
        return;
    }

    // Condiciones de frontera
    if (i == 0 || i == ny-1 || j == 0 || j == nx-1) {
        rhs[idx] = 0.0f;
        return;
    }

    // boundary-safe neighbor indices (clamped)
    int i_top = min(i+1, ny-1);
    int i_bottom = max(i-1, 0);

    float I_idx = I[idx];
    float I_top = I[b*ny*nx + i_top * nx + j];
    float I_bottom = I[b*ny*nx + i_bottom * nx + j];

    float alpha = D[b] * dt / (d * d);

    // RHS para paso ADI en y: I + alpha/2 * Laplace_y(I)
    rhs[idx] = 2.0f * I_idx + alpha * (I_top - 2.0f * I_idx + I_bottom);
}

__global__ void compute_rhs_x(const float* I, float* rhs,
                              const float* D, const float dt, const float d,
                              const int ny, const int nx, const int n_batch,
                              const float* vegetacion) {

    int j = blockIdx.x * blockDim.x + threadIdx.x; // x
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y
    int b = blockIdx.z;                            // n_batch

    if (i >= ny || j >= nx || b >= n_batch) return;

    int idx2d = i * nx + j;
    int idx = b * nx * ny + idx2d;

    // No computar celdas sin combustible
    if (vegetacion[idx2d] <= 2.0f) {
        rhs[idx] = 0.0f;
        return;
    }

    // Condiciones de frontera
    if (i == 0 || i == ny-1 || j == 0 || j == nx-1) {
        rhs[idx] = 0.0f;
        return;
    }

    // boundary-safe neighbor indices (clamped)
    int j_left = max(j-1, 0);
    int j_right = min(j+1, nx-1);

    float I_idx = I[idx];
    float I_left = I[b*ny*nx + i * nx + j_left];
    float I_right = I[b*ny*nx + i * nx + j_right];

    float alpha = D[b] * dt / (d * d);

    // RHS para paso ADI en x: I + alpha/2 * Laplace_x(I)
    rhs[idx] = 2.0f * I_idx + alpha * (I_right - 2.0f * I_idx + I_left);
}

// Solver tridiagonal usando memoria global (para dominios grandes)
__global__ void solve_tridiagonal_x_global(const float* rhs, float* I, 
                                           float* c_prime_global, float* d_prime_global,
                                           const float* D, const float dt, const float d,
                                           const int ny, const int nx, const int n_batch,
                                           const float* vegetacion) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x; // indice fila
    int b = blockIdx.z;                            // n_batch

    if (i >= ny || b >= n_batch) return;
    if (i == 0 || i == ny-1) return; // condiciones de frontera

    // punteros base para fila i
    int row_offset2d = i * nx;
    int row_offset = b*ny*nx + row_offset2d;

    float alpha = D[b] * dt / (d * d);
    float a_val = -alpha;        // diagonal inferior
    float c_val = -alpha;        // diagonal superior

    // Usar memoria global - cada fila tiene su propio espacio
    float* c_prime = c_prime_global + i * nx + b*ny*nx;
    float* d_prime = d_prime_global + i * nx + b*ny*nx;

    // forward sweep
    int j = 0;
    int idx = row_offset + j;
    int idx2d = row_offset2d + j;
    if (vegetacion[idx2d] <= 2.0f) {
        I[idx] = 0.0f;
        c_prime[0] = 0.0f;
        d_prime[0] = 0.0f;
    } else {
        float b_val = 2.0f * (1.0f + alpha); // diagonal principal
        c_prime[0] = c_val / b_val;
        d_prime[0] = rhs[idx] / b_val;
    }

    for (int j = 1; j < nx; j++) {
        int idx2d = row_offset2d + j;
        int idx = row_offset + j;
        if (vegetacion[idx2d] <= 2.0f) {
            I[idx] = 0.0f;
            c_prime[j] = 0.0f;
            d_prime[j] = 0.0f;
            continue;
        }
        float b_val = 2.0f * (1.0f + alpha); // diagonal principal
        float denom = b_val - a_val * c_prime[j - 1];
        c_prime[j] = c_val / denom;
        d_prime[j] = (rhs[row_offset + j] - a_val * d_prime[j - 1]) / denom;
    }

    // backward substitution
    if (vegetacion[row_offset2d + nx - 1] <= 2.0f) {
        I[row_offset + nx - 1] = 0.0f;
    } else {
        I[row_offset + nx - 1] = d_prime[nx - 1];
    }
    for (int j = nx - 2; j >= 0; j--) {
        int idx2d = row_offset2d + j;
        int idx = row_offset + j;
        if (vegetacion[idx2d] <= 2.0f) {
            I[idx] = 0.0f;
            continue;
        }
        I[idx] = d_prime[j] - c_prime[j] * I[row_offset + j + 1];
    }
}

__global__ void solve_tridiagonal_y_global(const float* rhs, float* I,
                                           float* c_prime_global, float* d_prime_global,
                                           const float* D, const float dt, const float d,
                                           const int ny, const int nx, const int n_batch,
                                           const float* vegetacion) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // columna
    int b = blockIdx.z;                           // n_batch
    if (j >= nx || b >= n_batch) return;
    if (j == 0 || j == nx-1) return; // condiciones de frontera

    float alpha = D[b] * dt / (d * d);
    float a_val = -alpha;        // diagonal inferior
    float c_val = -alpha;        // diagonal superior

    // Usar memoria global - cada columna tiene su propio espacio
    float* c_prime = c_prime_global + j * ny + b*nx*ny;
    float* d_prime = d_prime_global + j * ny + b*nx*ny;

    // Forward sweep
    int k = 0;
    int idx2d = k * nx + j;
    int idx = b * nx * ny + idx2d;
    if (vegetacion[idx2d] <= 2.0f) {
        I[idx] = 0.0f;
        c_prime[0] = 0.0f;
        d_prime[0] = 0.0f;
    } else {
        float b_val = 2.0f * (1.0f + alpha); // diagonal principal
        c_prime[0] = c_val / b_val;
        d_prime[0] = rhs[idx] / b_val;
    }

    for (int k = 1; k < ny; k++) {
        int idx2d = k * nx + j;
        int idx = b * nx * ny + idx2d;
        if (vegetacion[idx2d] <= 2.0f) {
            I[idx] = 0.0f;
            c_prime[k] = 0.0f;
            d_prime[k] = 0.0f;
            continue;
        }
        float b_val = 2.0f * (1.0f + alpha); // diagonal principal
        float denom = b_val - a_val * c_prime[k-1];
        c_prime[k] = c_val / denom;
        d_prime[k] = (rhs[idx] - a_val * d_prime[k-1]) / denom;
    }

    // Back substitution
    if (vegetacion[(ny-1) * nx + j] <= 2.0f) {
        I[b*nx*ny + (ny-1) * nx + j] = 0.0f;
    } else {
        I[b*nx*ny + (ny-1) * nx + j] = d_prime[(ny-1)];
    }
    for (int k = ny-2; k >= 0; k--) {
        int idx2d = k * nx + j;
        int idx = b*nx*ny + idx2d;
        if (vegetacion[idx2d] <= 2.0f) {
            I[idx] = 0.0f;
            continue;
        }
        I[idx] = d_prime[k] - c_prime[k] * I[b*nx*ny + (k+1) * nx + j];
    }
}


__global__ void reaction_advection_kernel_raw(const float* S, const float* I, const float* R,
                                              float* S_tmp, float* I_tmp, float* R_tmp,
                                              const float* beta, const float* gamma,
                                              const float dt, const float d,
                                              const float* wx, const float* wy,
                                              const float* h_dx, const float* h_dy,
                                              const float* A, const float* B,
                                              const int ny, const int nx, const int n_batch,
                                              const float* vegetacion) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y
    int b = blockIdx.z;

    if (i >= ny || j >= nx || b >= n_batch) return;

    int idx2d = i * nx + j;
    int idx = b * ny * nx + idx2d;

    // No computar celdas sin combustible
    if (vegetacion[idx2d] <= 2.0f) {
        S_tmp[idx] = 0.0f;
        I_tmp[idx] = 0.0f;
        R_tmp[idx] = 0.0f;
        return;
    }

    // Bordes 
    if (i == 0 || i == ny-1 || j == 0 || j == nx-1) {
        S_tmp[idx] = 0.0f;
        I_tmp[idx] = 0.0f;
        R_tmp[idx] = 0.0f;
        return;
    }

    float S_val = S[idx];
    float I_val = I[idx];
    float R_val = R[idx];
    float beta_val = beta[idx];
    float gamma_val = gamma[idx];

    float adv_x = A[b] * wx[idx2d] + B[b] * h_dx[idx2d];
    float adv_y = A[b] * wy[idx2d] + B[b] * h_dy[idx2d];

    float I_top    = I[b*ny*nx + (i+1)*nx + j];
    float I_bottom = I[b*ny*nx + (i-1)*nx + j];
    float I_left   = I[b*ny*nx + i*nx + (j-1)];
    float I_right  = I[b*ny*nx + i*nx + (j+1)];

    float I_dx = (adv_x > 0.0f) ? (I_val - I_left) : (I_right - I_val);
    float I_dy = (adv_y > 0.0f) ? (I_val - I_bottom) : (I_top - I_val);

    // reaccion y adveccion *explicitos* (sin difusion)
    float S_new_val = S_val - dt * beta_val * I_val * S_val;
    float I_new_val = I_val + dt * beta_val * I_val * S_val - dt * gamma_val * I_val - dt / d * (adv_x * I_dx + adv_y * I_dy);
    float R_new_val = R_val + dt * gamma_val * I_val;

    S_tmp[idx] = S_new_val;
    I_tmp[idx] = I_new_val;
    R_tmp[idx] = R_new_val;
}

}
'''

# Compilar kernels ADI
mod_adi = cp.RawModule(code=kernel_code_adi)
compute_rhs_y_kernel = mod_adi.get_function('compute_rhs_y')
compute_rhs_x_kernel = mod_adi.get_function('compute_rhs_x')
solve_tridiagonal_y_global_kernel = mod_adi.get_function('solve_tridiagonal_y_global')
solve_tridiagonal_x_global_kernel = mod_adi.get_function('solve_tridiagonal_x_global')
reaction_advection_kernel_raw = mod_adi.get_function('reaction_advection_kernel_raw')

def spread_infection_adi(S, I, R, S_new, I_new, R_new,
                         dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B, vegetacion):
    """Paso ADI (Alternating Direction Implicit) para el sistema SIR con difusión + advección.

    Formato esperado de tensores: (n_batch, ny, nx).

    Esquema (dt total):
      1) Reacción + advección explícitas dt/2  -> (S_half, I_star, R_half)
      2) Difusión implícita en Y (sobre I_star)
      3) Reacción + advección explícitas dt/2  -> (S_new, I_new, R_new)
      4) Difusión implícita en X (sobre I_new)

    """
    if S.ndim != 3:
        raise ValueError("Se espera S con 3 dimensiones (n_batch, ny, nx)")

    n_batch, ny, nx = S.shape

    # Grid para kernels 2D (con batch en z)
    threads_2d = (16, 16)
    blocks_x = (nx + threads_2d[0] - 1) // threads_2d[0]
    blocks_y = (ny + threads_2d[1] - 1) // threads_2d[1]
    grid_diff_adv = (blocks_x, blocks_y, n_batch)

    # Buffers temporales
    rhs_y = cp.zeros_like(I)
    rhs_x = cp.zeros_like(I)
    I_star = cp.zeros_like(I)
    S_half = cp.zeros_like(S)
    R_half = cp.zeros_like(R)

    # --- Primer semi-paso (dt/2) ---
    reaction_advection_kernel_raw(
        grid_diff_adv, threads_2d,
        (
            S.ravel(), I.ravel(), R.ravel(),
            S_half.ravel(), I_star.ravel(), R_half.ravel(),
            beta.ravel(), gamma.ravel(),
            cp.float32(dt/2), cp.float32(d),
            wx.ravel(), wy.ravel(),
            h_dx.ravel(), h_dy.ravel(),
            A.ravel(), B.ravel(),
            cp.int32(ny), cp.int32(nx), cp.int32(n_batch),
            vegetacion.ravel()
        )
    )

    compute_rhs_y_kernel(
        grid_diff_adv, threads_2d,
        (
            I_star.ravel(), rhs_y.ravel(),
            D.ravel(), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx), cp.int32(n_batch),
            vegetacion.ravel()
        )
    )

    c_prime_x = cp.zeros((n_batch, ny, nx), dtype=cp.float32)
    d_prime_x = cp.zeros((n_batch, ny, nx), dtype=cp.float32)
    threads_x_solve = (16,)
    grid_x_solve = ((ny + threads_x_solve[0] - 1) // threads_x_solve[0], 1, n_batch)

    solve_tridiagonal_x_global_kernel(
        grid_x_solve, threads_x_solve,
        (
            rhs_y.ravel(), I_star.ravel(), 
            c_prime_x.ravel(), d_prime_x.ravel(),
            D.ravel(), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx), cp.int32(n_batch),
            vegetacion.ravel()
        )
    )

    # --- Segundo semi-paso (dt/2) ---
    reaction_advection_kernel_raw(
        grid_diff_adv, threads_2d,
        (
            S_half.ravel(), I_star.ravel(), R_half.ravel(),
            S_new.ravel(), I_new.ravel(), R_new.ravel(),
            beta.ravel(), gamma.ravel(),
            cp.float32(dt/2), cp.float32(d),
            wx.ravel(), wy.ravel(),
            h_dx.ravel(), h_dy.ravel(),
            A.ravel(), B.ravel(),
            cp.int32(ny), cp.int32(nx), cp.int32(n_batch),
            vegetacion.ravel()
        )
    )

    compute_rhs_x_kernel(
        grid_diff_adv, threads_2d,
        (
            I_new.ravel(), rhs_x.ravel(),
            D.ravel(), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx), cp.int32(n_batch),
            vegetacion.ravel()
        )
    )

    c_prime_y = cp.zeros((n_batch, nx, ny), dtype=cp.float32)
    d_prime_y = cp.zeros((n_batch, nx, ny), dtype=cp.float32)
    threads_y_solve = (16, 1)
    grid_y_solve = ((nx + threads_y_solve[0] - 1) // threads_y_solve[0], 1, n_batch)

    solve_tridiagonal_y_global_kernel(
        grid_y_solve, threads_y_solve,
        (
            rhs_x.ravel(), I_new.ravel(), 
            c_prime_y.ravel(), d_prime_y.ravel(),
            D.ravel(), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx), cp.int32(n_batch),
            vegetacion.ravel()
        )
    )

    # In-place; no return necesario.

############################## CONDICIÓN DE COURANT ###############################################

def courant_batch(dt, A, B, d, wx, wy, h_dx, h_dy):
    """
    Verifica la condición de estabilidad de Courant por batch.
    
    Args:
        dt (float): Intervalo de tiempo.
        A (cp.ndarray): Array de coeficientes A, shape (n_batch,)
        B (cp.ndarray): Array de coeficientes B, shape (n_batch,)
        d (float): Tamaño de la celda
        wx, wy, h_dx, h_dy: arrays de shape (n_batch, ny, nx)
    
    Returns:
        cp.ndarray: Booleano por batch, shape (n_batch,)
    """
    
    # Velocidad efectiva por batch y por celda
    velocidad = cp.sqrt(
        (A[:, None, None] * wx + B[:, None, None] * h_dx)**2 +
        (A[:, None, None] * wy + B[:, None, None] * h_dy)**2
    )

    # Max por batch
    velocidad_max = cp.max(velocidad, axis=(1,2))

    # Condición de Courant advección por batch
    courant_advectivo = d / (cp.sqrt(2) * velocidad_max)

    # Retorna booleano por batch
    return dt < courant_advectivo

def courant(dt, A, B, d, wx, wy, h_dx, h_dy):
    
    '''
    Verifica la condición de estabilidad de Courant para un modelo de difusión y advección.
    
    Args:
      dt (float): Intervalo de tiempo.
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
    
    # Courant para advección
    velocidad = cp.sqrt((A * wx + B * h_dx)**2 + (A * wy + B * h_dy)**2)
    courant_advectivo = d / (cp.sqrt(2) * cp.max(velocidad))

    # Retorna un booleano indicando si la condición de estabilidad se cumple
    #return dt < cp.minimum(courant_difusion, courant_advectivo)
    return dt < courant_advectivo

############################## SPREAD INFECTION CON RAW KERNEL ###############################################

kernel_code = r'''
extern "C" {

__global__ void spread_infection_kernel_raw(const float* S, const float* I, const float* R,
                                  float* S_new, float* I_new, float* R_new,
                                  const float* beta, const float* gamma,
                                  const float dt, const float d, const float* D, 
                                  const float* wx, const float* wy,
                                  const float* h_dx, const float* h_dy,
                                  const float* A, const float* B,
                                  const int ny, const int nx, const int n_batch, const float* vegetacion) 
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

    if (vegetacion[idx] <= 2.0f) {
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
    float s = D[b] * dt / (d * d);

    float adv_x = A[b] * wx[idx] + B[b] * h_dx[idx];
    float adv_y = A[b] * wy[idx] + B[b] * h_dy[idx];

    float I_dx = (adv_x > 0.0f) ? (I_val - I_left) : (I_right - I_val);
    float I_dy = (adv_y > 0.0f) ? (I_val - I_bottom) : (I_top - I_val);

    float S_temp = S_val - dt * beta_val * I_val * S_val;
    float I_temp = I_val + dt * (beta_val * I_val * S_val - gamma_val * I_val)
                   + s * laplacian_I - dt / d * (adv_x * I_dx + adv_y * I_dy);
    float R_temp = R_val + dt * gamma_val * I_val;

    S_new[idx] = S_temp;
    I_new[idx] = I_temp;
    R_new[idx] = R_temp;
}
}
'''

mod = cp.RawModule(code=kernel_code)
spread_kernel_raw = mod.get_function('spread_infection_kernel_raw')

def spread_infection_explicit_raw(S, I, R, S_new, I_new, R_new, 
                                dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B, vegetacion):

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
            cp.float32(dt), cp.float32(d), D.ravel(),
            wx.ravel(), wy.ravel(),
            h_dx.ravel(), h_dy.ravel(),
            A.ravel(), B.ravel(),
            cp.int32(ny), cp.int32(nx), cp.int32(n_batch), vegetacion.ravel()
        )
    )