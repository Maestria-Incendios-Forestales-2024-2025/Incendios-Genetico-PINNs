import cupy as cp # type: ignore

############################## SPREAD INFECTION CON RAW KERNEL ###############################################

kernel_code = r'''
extern "C" {

__global__ void compute_rhs_y(const float* I, float* rhs,
                              const float D, const float dt, const float d,
                              const int ny, const int nx) {

    int j = blockIdx.x * blockDim.x + threadIdx.x; // x
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y

    if (i >= ny || j >= nx) return;

    int idx = i * nx + j;

    // Condiciones de frontera
    if (i == 0 || i == ny-1 || j == 0 || j == nx-1) {
        rhs[idx] = 0.0f;
        return;
    }

    // boundary-safe neighbor indices (clamped)
    int i_top = min(i+1, ny-1);
    int i_bottom = max(i-1, 0);

    float I_idx = I[idx];
    float I_top = I[i_top * nx + j];
    float I_bottom = I[i_bottom * nx + j];

    float alpha = D * dt / (2.0f * d * d);

    // RHS para paso ADI en y: I + alpha/2 * Laplace_y(I)
    rhs[idx] = I_idx + alpha * (I_top - 2.0f * I_idx + I_bottom);
}

__global__ void compute_rhs_x(const float* I, float* rhs,
                              const float D, const float dt, const float d,
                              const int ny, const int nx) {

    int j = blockIdx.x * blockDim.x + threadIdx.x; // x
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y

    if (i >= ny || j >= nx) return;

    int idx = i * nx + j;

    // Condiciones de frontera
    if (i == 0 || i == ny-1 || j == 0 || j == nx-1) {
        rhs[idx] = 0.0f;
        return;
    }

    // boundary-safe neighbor indices (clamped)
    int j_left = max(j-1, 0);
    int j_right = min(j+1, nx-1);

    float I_idx = I[idx];
    float I_left = I[i * nx + j_left];
    float I_right = I[i * nx + j_right];

    float alpha = D * dt / (2.0f * d * d);

    // RHS para paso ADI en x: I + alpha/2 * Laplace_x(I)
    rhs[idx] = I_idx + alpha * (I_right - 2.0f * I_idx + I_left);
}

// Solver tridiagonal usando memoria global (para dominios grandes)
__global__ void solve_tridiagonal_x_global(const float* rhs, float* I, 
                                           float* c_prime_global, float* d_prime_global,
                                           const float D, const float dt, const float d,
                                           const int Ny, const int Nx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // indice fila

    if (i >= Ny) return;
    if (i == 0 || i == Ny-1) return; // condiciones de frontera

    float alpha = D * dt / (2.0f * d * d);
    float a_val = -alpha;        // diagonal inferior
    float b_val = 1.0f + 2.0f * alpha; // diagonal principal
    float c_val = -alpha;        // diagonal superior

    // Usar memoria global - cada fila tiene su propio espacio
    float* c_prime = c_prime_global + i * Nx;
    float* d_prime = d_prime_global + i * Nx;

    // punteros base para fila i
    int row_offset = i * Nx;

    // forward sweep
    c_prime[0] = c_val / b_val;
    d_prime[0] = rhs[row_offset] / b_val;

    for (int j = 1; j < Nx; j++) {
        float denom = b_val - a_val * c_prime[j - 1];
        c_prime[j] = c_val / denom;
        d_prime[j] = (rhs[row_offset + j] - a_val * d_prime[j - 1]) / denom;
    }

    // backward substitution
    I[row_offset + Nx - 1] = d_prime[Nx - 1];
    for (int j = Nx - 2; j >= 0; j--) {
        I[row_offset + j] = d_prime[j] - c_prime[j] * I[row_offset + j + 1];
    }
}

__global__ void solve_tridiagonal_y_global(const float* rhs, float* I, 
                                           float* c_prime_global, float* d_prime_global,
                                           const float D, const float dt, const float d,
                                           const int Ny, const int Nx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // columna
    if (j >= Nx) return;
    if (j == 0 || j == Nx-1) return; // condiciones de frontera

    float alpha = D * dt / (2.0f * d * d);
    float a_val = -alpha;        // diagonal inferior
    float b_val = 1.0f + 2.0f * alpha; // diagonal principal  
    float c_val = -alpha;        // diagonal superior

    // Usar memoria global - cada columna tiene su propio espacio
    float* c_prime = c_prime_global + j * Ny;
    float* d_prime = d_prime_global + j * Ny;

    // Forward sweep
    c_prime[0] = c_val / b_val;
    d_prime[0] = rhs[j] / b_val;

    for (int k = 1; k < Ny; k++) {
        int idx = k * Nx + j;
        float denom = b_val - a_val * c_prime[k-1];
        c_prime[k] = c_val / denom;
        d_prime[k] = (rhs[idx] - a_val * d_prime[k-1]) / denom;
    }

    // Back substitution
    I[(Ny-1) * Nx + j] = d_prime[Ny-1];
    for (int k = Ny-2; k >= 0; k--) {
        I[k * Nx + j] = d_prime[k] - c_prime[k] * I[(k+1) * Nx + j];
    }
}


__global__ void reaction_advection_kernel_raw(const float* S, const float* I, const float* R,
                                              float* S_tmp, float* I_tmp, float* R_tmp,
                                              const float* beta, const float* gamma,
                                              const float dt, const float d,
                                              const float* wx, const float* wy,
                                              const float* h_dx, const float* h_dy,
                                              const float A, const float B,
                                              const int ny, const int nx) {
    
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // y

    if (i >= ny || j >= nx) return;

    int idx = i * nx + j;

    // borde: mantenemos lo que hacias antes
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

    float adv_x = A * wx[idx] + B * h_dx[idx];
    float adv_y = A * wy[idx] + B * h_dy[idx];

    // upwind discretization para adveccion (como en tu kernel)
    float I_top    = I[(i+1)*nx + j];
    float I_bottom = I[(i-1)*nx + j];
    float I_left   = I[i*nx + (j-1)];
    float I_right  = I[i*nx + (j+1)];

    float I_dx = (adv_x > 0.0f) ? (I_val - I_left) : (I_right - I_val);
    float I_dy = (adv_y > 0.0f) ? (I_val - I_bottom) : (I_top - I_val);

    // reaccion y adveccion *explicitos* (sin difusion)
    float S_new_val = S_val - dt * beta_val * I_val * S_val;
    float I_new_val = I_val + dt * (beta_val * I_val * S_val - gamma_val * I_val)
                      - dt / d * (adv_x * I_dx + adv_y * I_dy);
    float R_new_val = R_val + dt * gamma_val * I_val;

    // condiciones fisicas: (mantene tu chequeo)
    if (beta_val == 0.0f) {
        S_new_val = 0.0f;
        I_new_val = 0.0f;
        R_new_val = 0.0f;
    }

    S_tmp[idx] = S_new_val;
    I_tmp[idx] = I_new_val;
    R_tmp[idx] = R_new_val;
}

}
'''

# Compilar kernels ADI
mod = cp.RawModule(code=kernel_code)
compute_rhs_y_kernel = mod.get_function('compute_rhs_y')
compute_rhs_x_kernel = mod.get_function('compute_rhs_x')
solve_tridiagonal_y_global_kernel = mod.get_function('solve_tridiagonal_y_global')
solve_tridiagonal_x_global_kernel = mod.get_function('solve_tridiagonal_x_global')
# Solo para dominios pequeños (comentado por ahora):
# solve_tridiagonal_y_kernel = mod.get_function('solve_tridiagonal_y')
reaction_advection_kernel_raw = mod.get_function('reaction_advection_kernel_raw')

def spread_infection_adi(S, I, R, S_new, I_new, R_new, 
                         dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B):
    """
    Implementación corregida del método ADI para el sistema SIR con difusión y convección.
    
    El método ADI divide el paso temporal en dos sub-pasos de dt/2 cada uno:
    1. Implícito en Y, explícito en X para difusión
    2. Implícito en X, explícito en Y para difusión
    La reacción y advección se tratan explícitamente
    """
    ny, nx = S.shape
    threads = (16, 16)
    blocks_x = (nx + threads[0] - 1) // threads[0]
    blocks_y = (ny + threads[1] - 1) // threads[1]
    blocks_2d = (blocks_x, blocks_y)

    # Arrays temporales para ADI
    rhs_y = cp.zeros_like(I)
    rhs_x = cp.zeros_like(I)
    I_star = cp.zeros_like(I)  # Resultado intermedio
    S_half = cp.zeros_like(S)  # Estados intermedios
    R_half = cp.zeros_like(R)

    # print(f"Ejecutando ADI en dominio {ny}x{nx}")

    ############################## PRIMER SUB-PASO (dt/2) ##############################
    # Paso 1a: Reacción y Advección explícitas por dt/2
    reaction_advection_kernel_raw(
        blocks_2d, threads,
        (
            S.ravel(), I.ravel(), R.ravel(),
            S_half.ravel(), I_star.ravel(), R_half.ravel(),
            beta.ravel(), gamma.ravel(),
            cp.float32(dt/2), cp.float32(d),  # dt/2 para el primer sub-paso
            wx.ravel(), wy.ravel(),
            h_dx.ravel(), h_dy.ravel(),
            cp.float32(A), cp.float32(B),
            cp.int32(ny), cp.int32(nx)
        )
    )

    # Paso 1b: Difusión implícita en Y por dt/2
    compute_rhs_y_kernel(
        blocks_2d, threads,
        (
            I_star.ravel(), rhs_y.ravel(), 
            cp.float32(D), cp.float32(dt), cp.float32(d),  # dt total para alpha
            cp.int32(ny), cp.int32(nx)
        )
    )
    
    # Resolver sistemas tridiagonales en Y
    c_prime_y = cp.zeros((nx, ny), dtype=cp.float32)
    d_prime_y = cp.zeros((nx, ny), dtype=cp.float32)
    blocks_y_solve = ((nx + 15) // 16,)
    
    solve_tridiagonal_y_global_kernel(
        blocks_y_solve, (16,),
        (
            rhs_y.ravel(), I_star.ravel(),
            c_prime_y.ravel(), d_prime_y.ravel(),
            cp.float32(D), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx)
        )
    )

    ############################## SEGUNDO SUB-PASO (dt/2) ##############################
    # Paso 2a: Reacción y Advección explícitas por dt/2 adicional
    reaction_advection_kernel_raw(
        blocks_2d, threads,
        (
            S_half.ravel(), I_star.ravel(), R_half.ravel(),
            S_new.ravel(), I_new.ravel(), R_new.ravel(),
            beta.ravel(), gamma.ravel(),
            cp.float32(dt/2), cp.float32(d),  # dt/2 para el segundo sub-paso
            wx.ravel(), wy.ravel(),
            h_dx.ravel(), h_dy.ravel(),
            cp.float32(A), cp.float32(B),
            cp.int32(ny), cp.int32(nx)
        )
    )

    # Paso 2b: Difusión implícita en X por dt/2
    compute_rhs_x_kernel(
        blocks_2d, threads,
        (
            I_new.ravel(), rhs_x.ravel(),
            cp.float32(D), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx)
        )
    )
    
    # Resolver sistemas tridiagonales en X
    c_prime_x = cp.zeros((ny, nx), dtype=cp.float32)
    d_prime_x = cp.zeros((ny, nx), dtype=cp.float32)
    blocks_x_solve = ((ny + 15) // 16,)
    
    solve_tridiagonal_x_global_kernel(
        blocks_x_solve, (16,),
        (
            rhs_x.ravel(), I_new.ravel(),
            c_prime_x.ravel(), d_prime_x.ravel(),
            cp.float32(D), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx)
        )
    )

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

############################## FUNCIONES AUXILIARES PARA ADI ###############################################

def create_adi_example():
    """
    Ejemplo de uso del método ADI para comparar con el método explícito original.
    """
    # Parámetros del dominio (probemos primero con un dominio más pequeño)
    nx, ny = 128, 128  # Dominio más pequeño para debugging
    d = 1.0
    
    print(f"Creando dominio para debugging: {nx}x{ny}")
    
    # Inicializar arrays
    S = cp.ones((ny, nx), dtype=cp.float32) * 0.9
    I = cp.zeros((ny, nx), dtype=cp.float32)
    R = cp.zeros((ny, nx), dtype=cp.float32)
    
    # Arrays de salida
    S_new = cp.zeros_like(S)
    I_new = cp.zeros_like(I)
    R_new = cp.zeros_like(R)
    
    # Condición inicial: foco de infección en el centro
    center_y, center_x = ny // 2, nx // 2
    I[center_y-5:center_y+6, center_x-5:center_x+6] = 0.1
    S[center_y-5:center_y+6, center_x-5:center_x+6] = 0.8
    
    # Parámetros del modelo
    beta = cp.ones((ny, nx), dtype=cp.float32) * 0.3
    gamma = cp.ones((ny, nx), dtype=cp.float32) * 0.1
    D = 0.01  # Coeficiente de difusión
    
    # Parámetros de viento y terreno (simplificados)
    wx = cp.ones((ny, nx), dtype=cp.float32) * 0.1
    wy = cp.ones((ny, nx), dtype=cp.float32) * 0.05
    h_dx = cp.zeros((ny, nx), dtype=cp.float32)
    h_dy = cp.zeros((ny, nx), dtype=cp.float32)
    A = 0.1
    B = 0.05
    
    # Condiciones de frontera (beta = 0 en bordes)
    beta[0, :] = 0.0
    beta[-1, :] = 0.0
    beta[:, 0] = 0.0
    beta[:, -1] = 0.0
    
    # Paso temporal
    dt = 0.01  # Paso más pequeño para debugging
    
    print(f"Configuración del problema:")
    print(f"Dominio: {nx}x{ny}")
    print(f"Paso temporal: {dt}")
    print(f"Coeficiente de difusión: {D}")
    print(f"Memoria requerida por dominio: {nx*ny*4/1024/1024:.2f} MB por array")
    
    # Diagnósticos iniciales
    print(f"\nEstado inicial:")
    print(f"  S total: {cp.sum(S):.6f}")
    print(f"  I total: {cp.sum(I):.6f}")
    print(f"  R total: {cp.sum(R):.6f}")
    print(f"  Total: {cp.sum(S+I+R):.6f}")
    print(f"  Puntos con I>0: {cp.sum(I > 0)}")
    
    # Ejecutar un paso ADI
    print("\nEjecutando un paso con método ADI...")
    import time
    start_time = time.time()
    
    spread_infection_adi(S, I, R, S_new, I_new, R_new, 
                        dt, d, beta, gamma, D, wx, wy, h_dx, h_dy, A, B)
    
    cp.cuda.Stream.null.synchronize()  # Esperar a que termine
    end_time = time.time()
    
    print(f"\nTiempo de ejecución: {end_time - start_time:.4f} segundos")
    
    # Diagnósticos finales
    print(f"\nEstado final:")
    print(f"  S total: {cp.sum(S_new):.6f}")
    print(f"  I total: {cp.sum(I_new):.6f}")
    print(f"  R total: {cp.sum(R_new):.6f}")
    print(f"  Total: {cp.sum(S_new+I_new+R_new):.6f}")
    print(f"  Puntos con I>0: {cp.sum(I_new > 0)}")
    
    print(f"\nCambios:")
    print(f"  Max I antes: {cp.max(I):.6f}")
    print(f"  Max I después: {cp.max(I_new):.6f}")
    print(f"  Conservación de masa: {abs(cp.sum(S+I+R) - cp.sum(S_new+I_new+R_new)):.8f}")
    
    # Verificar si hay valores negativos (problema común)
    print(f"\nVerificación de valores negativos:")
    print(f"  S negativo: {cp.sum(S_new < 0)}")
    print(f"  I negativo: {cp.sum(I_new < 0)}")
    print(f"  R negativo: {cp.sum(R_new < 0)}")
    
    if cp.sum(S_new < 0) > 0:
        print(f"  Min S: {cp.min(S_new):.8f}")
    if cp.sum(I_new < 0) > 0:
        print(f"  Min I: {cp.min(I_new):.8f}")
    if cp.sum(R_new < 0) > 0:
        print(f"  Min R: {cp.min(R_new):.8f}")
    
    return S, I, R, S_new, I_new, R_new

def test_diffusion_only_adi(S, I, R, S_new, I_new, R_new, 
                            dt, d, D):
    """
    Prueba solo la difusión usando el esquema ADI, sin reacción ni advección.
    Esto nos ayuda a verificar la conservación de masa.
    """
    ny, nx = S.shape
    threads = (16, 16)
    blocks_x = (nx + threads[0] - 1) // threads[0]
    blocks_y = (ny + threads[1] - 1) // threads[1]
    blocks_2d = (blocks_x, blocks_y)

    # Arrays temporales
    rhs_y = cp.zeros_like(I)
    rhs_x = cp.zeros_like(I)
    I_star = cp.zeros_like(I)

    print(f"Probando solo difusión ADI en dominio {ny}x{nx}")

    # Copiar S y R sin cambios (no hay reacción)
    S_new[:] = S
    R_new[:] = R
    I_star[:] = I  # Comenzar con I original

    ############################## PRIMER SUB-PASO (dt/2) ##############################
    # Solo difusión implícita en Y por dt/2
    compute_rhs_y_kernel(
        blocks_2d, threads,
        (
            I_star.ravel(), rhs_y.ravel(), 
            cp.float32(D), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx)
        )
    )
    
    # Resolver sistemas tridiagonales en Y
    c_prime_y = cp.zeros((nx, ny), dtype=cp.float32)
    d_prime_y = cp.zeros((nx, ny), dtype=cp.float32)
    blocks_y_solve = ((nx + 15) // 16,)
    
    solve_tridiagonal_y_global_kernel(
        blocks_y_solve, (16,),
        (
            rhs_y.ravel(), I_star.ravel(),
            c_prime_y.ravel(), d_prime_y.ravel(),
            cp.float32(D), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx)
        )
    )

    ############################## SEGUNDO SUB-PASO (dt/2) ##############################
    # Solo difusión implícita en X por dt/2
    compute_rhs_x_kernel(
        blocks_2d, threads,
        (
            I_star.ravel(), rhs_x.ravel(),
            cp.float32(D), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx)
        )
    )
    
    # Resolver sistemas tridiagonales en X
    c_prime_x = cp.zeros((ny, nx), dtype=cp.float32)
    d_prime_x = cp.zeros((ny, nx), dtype=cp.float32)
    blocks_x_solve = ((ny + 15) // 16,)
    
    solve_tridiagonal_x_global_kernel(
        blocks_x_solve, (16,),
        (
            rhs_x.ravel(), I_new.ravel(),
            c_prime_x.ravel(), d_prime_x.ravel(),
            cp.float32(D), cp.float32(dt), cp.float32(d),
            cp.int32(ny), cp.int32(nx)
        )
    )

def test_diffusion_only():
    """
    Prueba solo la difusión para verificar conservación de masa.
    """
    # Dominio real para ver el rendimiento
    nx, ny = 1768, 1768  # Tu dominio real
    d = 1.0
    
    print(f"=== Prueba de solo difusión ===")
    print(f"Dominio: {nx}x{ny}")
    
    # Crear una distribución inicial simple
    I = cp.zeros((ny, nx), dtype=cp.float32)
    S = cp.zeros((ny, nx), dtype=cp.float32)
    R = cp.zeros((ny, nx), dtype=cp.float32)
    
    # Arrays de salida
    S_new = cp.zeros_like(S)
    I_new = cp.zeros_like(I)
    R_new = cp.zeros_like(R)
    
    # Condición inicial: gaussiana en el centro (más pequeña para dominio grande)
    center_y, center_x = ny // 2, nx // 2
    for i in range(max(0, center_y-20), min(ny, center_y+21)):
        for j in range(max(0, center_x-20), min(nx, center_x+21)):
            r_sq = (i - center_y)**2 + (j - center_x)**2
            if r_sq <= 100:  # Radio 10
                I[i, j] = 1.0 * cp.exp(-r_sq / 50.0)
    
    # Parámetros
    D = 0.01  # Coeficiente de difusión como en tu ejemplo original
    dt = 0.02  # Paso temporal más grande (ventaja del ADI)
    
    print(f"Configuración:")
    print(f"  D = {D}")
    print(f"  dt = {dt}")
    print(f"  α = D*dt/(d²) = {D*dt/(d*d):.6f}")
    print(f"  Memoria por array: {nx*ny*4/1024/1024:.2f} MB")
    
    # Estado inicial
    masa_inicial = cp.sum(I)
    print(f"\nEstado inicial:")
    print(f"  Masa total I: {masa_inicial:.8f}")
    print(f"  Max I: {cp.max(I):.8f}")
    print(f"  Puntos con I>0: {cp.sum(I > 0)}")
    
    # Ejecutar solo difusión
    print(f"\nEjecutando difusión ADI en dominio {nx}x{ny}...")
    start_time = cp.cuda.Event()
    end_time = cp.cuda.Event()
    start_time.record()
    
    test_diffusion_only_adi(S, I, R, S_new, I_new, R_new, dt, d, D)
    
    end_time.record()
    end_time.synchronize()
    tiempo = cp.cuda.get_elapsed_time(start_time, end_time)
    
    # Estado final
    masa_final = cp.sum(I_new)
    print(f"\nEstado final:")
    print(f"  Masa total I: {masa_final:.8f}")
    print(f"  Max I: {cp.max(I_new):.8f}")
    print(f"  Puntos con I>0: {cp.sum(I_new > 0)}")
    
    # Conservación
    conservacion = abs(masa_inicial - masa_final)
    print(f"\nConservación de masa:")
    print(f"  Diferencia absoluta: {conservacion:.12f}")
    print(f"  Diferencia relativa: {conservacion/masa_inicial:.12f}")
    print(f"  Tiempo: {tiempo:.4f} ms")
    
    # Verificar valores negativos
    negativos = cp.sum(I_new < 0)
    if negativos > 0:
        print(f"\n⚠️  PROBLEMA: {negativos} valores negativos!")
        print(f"    Min I: {cp.min(I_new):.12f}")
    else:
        print(f"\n✓ No hay valores negativos")
    
    # Verificar simetría (debería ser simétrica para condición inicial gaussiana)
    center_val = float(I_new[center_y, center_x])
    test_points = [
        (center_y-1, center_x), (center_y+1, center_x),
        (center_y, center_x-1), (center_y, center_x+1)
    ]
    print(f"\nVerificación de simetría:")
    print(f"  Centro: {center_val:.8f}")
    for i, (y, x) in enumerate(test_points):
        val = float(I_new[y, x])
        print(f"  Vecino {i+1}: {val:.8f}")
    
    return I, I_new, conservacion

if __name__ == "__main__":
    # Ejemplo de uso
    try:
        print("=== Prueba del Método ADI - Solo Difusión ===")
        test_diffusion_only()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()