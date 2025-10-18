import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F
import numpy as np # type: ignore
import copy
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## DEFINICIÓN DE LA PINN ###############################################

temporal_domain = 10
domain_size = 2
MAX_BLOCK_POINTS = 4096  # límite de puntos por bloque temporal para estabilidad de memoria

class FireSpread_PINN(nn.Module):
    def __init__(self, modo='forward', beta = 1.0, gamma = 0.3, D_I = 0.005,
                layers=[3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 3]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

        # inicialización de pesos como tensores en el device
        self.w_ic  = torch.tensor(1.0, device=device)
        self.w_bc  = torch.tensor(1.0, device=device)
        self.w_pde = torch.tensor(1.0, device=device)

        # Parámetros del modelo
        self.beta = beta
        self.gamma = gamma
        self.mode = modo

        # Parámetro D_I puede ser fijo o entrenable
        if self.mode == 'forward':
            self.D_I = D_I
        elif self.mode == 'inverse':
            self.D_I = nn.Parameter(torch.tensor(D_I, device=device))

    def forward(self, x, y, t):
        inputs = torch.cat((x, y, t), dim=1)
        x_scaled = 2 * (inputs[:, 0:1] / domain_size) - 1
        y_scaled = 2 * (inputs[:, 1:2] / domain_size) - 1
        t_scaled = 2 * (inputs[:, 2:3] / temporal_domain) - 1
        out = torch.cat([x_scaled, y_scaled, t_scaled], dim=1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        return self.layers[-1](out)

    #---------------- PÉRDIDA POR CONDICIONES INICIALES ----------------#
    def loss_initial_condition(self, x_ic, y_ic, t_ic, S0, I0, R0):
        pred = self.forward(x_ic, y_ic, t_ic)
        S_pred, I_pred, R_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        return F.mse_loss(S_pred, S0) + F.mse_loss(I_pred, I0) + F.mse_loss(R_pred, R0)

    #---------------- PÉRDIDA POR CONDICIONES DE BORDE ----------------#
    def loss_boundary_condition(self, y_top, y_bottom, x_left, x_right, x_bc, y_bc, t_bc):
        top_pred = self.forward(x_bc, y_top, t_bc) # Borde de arriba (x,y)=(x,1)
        S_top_pred, I_top_pred, R_top_pred = top_pred[:, 0:1], top_pred[:, 1:2], top_pred[:, 2:3]

        bottom_pred = self.forward(x_bc, y_bottom, t_bc) # Borde de abajo (x,y)=(x,0)
        S_bottom_pred, I_bottom_pred, R_bottom_pred = bottom_pred[:, 0:1], bottom_pred[:, 1:2], bottom_pred[:, 2:3]

        left_pred = self.forward(x_left, y_bc, t_bc) # Borde de la izquierda (x,y)=(0,y)
        S_left_pred, I_left_pred, R_left_pred = left_pred[:, 0:1], left_pred[:, 1:2], left_pred[:, 2:3]

        right_pred = self.forward(x_right, y_bc, t_bc) # Borde de la derecha (x,y)=(1,y)
        S_right_pred, I_right_pred, R_right_pred = right_pred[:, 0:1], right_pred[:, 1:2], right_pred[:, 2:3]

        # Pérdida por condiciones de borde
        loss_top_bc = F.mse_loss(S_top_pred, S_bottom_pred) + F.mse_loss(I_top_pred, I_bottom_pred) + F.mse_loss(R_top_pred, R_bottom_pred)
        loss_left_bc = F.mse_loss(S_left_pred, S_right_pred) + F.mse_loss(I_left_pred, I_right_pred) + F.mse_loss(R_left_pred, R_right_pred)

        return loss_top_bc + loss_left_bc

    #---------------- PÉRDIDA POR LA FÍSICA ----------------#
    def loss_pde(self, x_phys, y_phys, t_phys, temporal_weights, N_blocks):
        # Crear bloques temporales
        T_final = temporal_domain
        t_blocks = torch.linspace(0, T_final, N_blocks + 1, device=device)

        # Lista para pérdidas por bloque
        block_losses = []

        for i in range(N_blocks):
            t_start, t_end = t_blocks[i].item(), t_blocks[i + 1].item()

            # Máscara de puntos en el bloque temporal
            t1 = t_phys.squeeze(-1)
            mask = (t1 >= t_start) & ((t1 < t_end) | (i == N_blocks - 1))  # incluir último punto
            idx_block = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if idx_block.numel() == 0:
                print("No hay puntos en el bloque temporal")
                block_loss = torch.tensor(0.0, device=device)
            else:
                # Submuestreo para limitar uso de memoria
                if idx_block.numel() > MAX_BLOCK_POINTS:
                    perm = torch.randperm(idx_block.numel(), device=device)
                    idx_block = idx_block[perm[:MAX_BLOCK_POINTS]]

                x_block = x_phys[idx_block]
                y_block = y_phys[idx_block]
                t_block = t_phys[idx_block]

                # Predicción de la red
                SIR_pred = self.forward(x_block, y_block, t_block)
                S_pred, I_pred, R_pred = SIR_pred[:, 0:1], SIR_pred[:, 1:2], SIR_pred[:, 2:3]

                # Derivadas temporales
                dS_dt = torch.autograd.grad(S_pred, t_block, torch.ones_like(S_pred), create_graph=True)[0]
                dI_dt = torch.autograd.grad(I_pred, t_block, torch.ones_like(I_pred), create_graph=True)[0]
                dR_dt = torch.autograd.grad(R_pred, t_block, torch.ones_like(R_pred), create_graph=True)[0]

                # Derivadas espaciales
                dI_dx = torch.autograd.grad(I_pred, x_block, torch.ones_like(I_pred), create_graph=True)[0]
                dI_dy = torch.autograd.grad(I_pred, y_block, torch.ones_like(I_pred), create_graph=True)[0]
                d2I_dx2 = torch.autograd.grad(dI_dx, x_block, torch.ones_like(dI_dx), create_graph=True)[0]
                d2I_dy2 = torch.autograd.grad(dI_dy, y_block, torch.ones_like(dI_dy), create_graph=True)[0]

                # Residuales PDE
                loss_S = dS_dt + self.beta * S_pred * I_pred
                loss_I = dI_dt - (self.beta * S_pred * I_pred - self.gamma * I_pred) - self.D_I * (d2I_dx2 + d2I_dy2)
                loss_R = dR_dt - self.gamma * I_pred

                block_loss = (loss_S**2 + loss_I**2 + loss_R**2).mean()

                # Chequeo NaN/Inf
                if torch.isnan(block_loss) or torch.isinf(block_loss):
                    print(f"Warning: NaN/Inf en bloque {i}, asignando 0")
                    block_loss = torch.tensor(0.0, device=device)

            block_losses.append(block_loss)

        # Tensor con pérdidas por bloque
        block_losses_tensor = torch.stack(block_losses)  # GPU
        pde_loss = torch.sum(temporal_weights * block_losses_tensor) / N_blocks
        temporal_loss = block_losses_tensor.detach()  # detach para usar en re-pesado temporal sin grafo

        return pde_loss, temporal_loss

    # -------------------- PÉRDIDA POR LOS DATOS --------------------#
    def loss_data(self, t_data, S_data, I_data, R_data):
        nx, ny = S_data.shape

        # construimos la grilla espacial según los límites
        x = torch.linspace(0, domain_size, nx, device=device)
        y = torch.linspace(0, domain_size, ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # aplanamos
        x_flat = X.flatten().unsqueeze(1)
        y_flat = Y.flatten().unsqueeze(1)

        # aseguramos que t tenga la misma longitud
        if torch.is_tensor(t_data):
            t_flat = torch.full_like(x_flat, t_data.item())  # si es tensor escalar
        else:
            t_flat = torch.full_like(x_flat, t_data)

        # flatten de los datos
        S_flat = S_data.flatten().unsqueeze(1)
        I_flat = I_data.flatten().unsqueeze(1)
        R_flat = R_data.flatten().unsqueeze(1)

        # predicción del modelo
        S_pred, I_pred, R_pred = self.forward(x_flat, y_flat, t_flat)

        # pérdida MSE sobre los datos
        loss = (
            F.mse_loss(S_pred, S_flat) +
            F.mse_loss(I_pred, I_flat) +
            F.mse_loss(R_pred, R_flat)
        )

        return loss

    # -------------------- PÉRDIDA POR NO NEGATIVIDAD --------------------#
    def non_negative_loss(self, x, y, t):
        S, I, R = self.forward(x, y, t).split(1, dim=1)
        loss = torch.mean(torch.relu(-S)) + torch.mean(torch.relu(-I)) + torch.mean(torch.relu(-R))
        return loss
    
    # -------------------- ACTUALIZACIÓN DE PESOS --------------------
    def update_loss_weights(self, loss_ic, loss_bc, loss_phys):
        """
        Recalcula pesos adaptativos para IC, BC y PDE cada 'every' épocas.
        """
        eps = 1e-8
        alpha = 0.9

        # Asignamos los pesos viejos a una variable intermedia
        w_ic_old, w_bc_old, w_pde_old = self.w_ic, self.w_bc, self.w_pde

        params_for_grad = [p for p in self.parameters() if p.requires_grad]

        grads_ic = torch.autograd.grad(loss_ic, params_for_grad, retain_graph=True, create_graph=False, allow_unused=True)
        grads_bc = torch.autograd.grad(loss_bc, params_for_grad, retain_graph=True, create_graph=False, allow_unused=True)
        grads_pde = torch.autograd.grad(loss_phys, params_for_grad, retain_graph=True, create_graph=False, allow_unused=True)

        def grad_norm(grads):
            total = torch.tensor(0.0, device=device)
            for g in grads:
                if g is not None:
                    total = total + (g.detach()**2).sum()
            return total.sqrt()

        g_ic, g_bc, g_pde = grad_norm(grads_ic), grad_norm(grads_bc), grad_norm(grads_pde)
        norms = torch.stack([g_ic, g_bc, g_pde]).clamp(min=eps)

        lambdas = (norms.sum() / norms)  

        # actualizar atributos internos
        self.w_ic, self.w_bc, self.w_pde = [l.item() for l in lambdas]

        # Asignamos nuevos valores
        self.w_ic = alpha * w_ic_old + (1-alpha) * self.w_ic
        self.w_bc = alpha * w_bc_old + (1-alpha) * self.w_bc
        self.w_pde = alpha * w_pde_old + (1-alpha) * self.w_pde

        return self.w_ic, self.w_bc, self.w_pde

    # -------------------- SAMPLEO POR NO LINEALIDAD --------------------
    def sample_by_nonlinearity(self, N_samples, N_candidates=100000, initial_points=False):
        x = domain_size*torch.rand(N_candidates, 1, device=device)
        y = domain_size*torch.rand(N_candidates, 1, device=device)
        if initial_points:
            t = torch.zeros(N_candidates, 1, device=device)
        else:
            t = temporal_domain * torch.rand(N_candidates, 1, device=device)

        with torch.no_grad():
            S_pred, I_pred, _ = self.forward(x, y, t).split(1, dim=1)
            nonlin_strength = torch.abs(S_pred * I_pred).squeeze()
            denom = nonlin_strength.sum()
            eps = 1e-12
            if torch.isnan(denom) or denom <= eps:
                # Caer a muestreo uniforme si no hay señal de no linealidad
                weights = torch.full((N_candidates,), 1.0 / N_candidates, device=device)
            else:
                weights = nonlin_strength / (denom + eps)
            idx = torch.multinomial(weights, N_samples, replacement=False)
        return x[idx], y[idx], t[idx]

    # -------------------- CLOSURE PARA OPTIMIZACIÓN --------------------
    def closure(self, optimizer, data, params):
        # calcular pérdidas
        loss_ic = self.loss_initial_condition(*data['ic'])
        loss_bc = self.loss_boundary_condition(*data['bc'])
        loss_phys, temporal_losses = self.loss_pde(*data['phys'], *params)

        if self.mode == 'inverse':
            loss_data = self.loss_data(*data['data'])
            # combinación lineal con los pesos
            total_loss = self.w_ic * loss_ic + self.w_bc * loss_bc + self.w_pde * loss_phys + loss_data
        else:
            # combinación lineal con los pesos
            total_loss = self.w_ic * loss_ic + self.w_bc * loss_bc + self.w_pde * loss_phys

        optimizer.zero_grad()
        # Retener grafo para permitir autograd.grad en update_loss_weights
        total_loss.backward(retain_graph=True)

        return (
            total_loss.detach(),
            loss_phys,  # no detach: se necesita el grafo para autograd.grad
            loss_ic,    # no detach
            loss_bc,    # no detach
            loss_data if self.mode == 'inverse' else torch.tensor(0.0, device=device),
            temporal_losses.detach(),
        )

############################## ENTRENAMIENTO ###############################################

# Función para entrenar la PINN con ecuaciones de propagación de fuego
def train_pinn(modo='forward',
               beta_val=1.0,
               gamma_val=0.3,
               D_I=0.005,
               mean_x=None, mean_y=None, sigma_x=None, sigma_y=None,
               epochs_adam=1000, N_blocks=10,
               checkpoint_path=None,
               t_data=None, S_data=None, I_data=None, R_data=None):

    model = FireSpread_PINN(
        modo=modo,
        beta=beta_val,
        gamma=gamma_val,
        D_I=D_I
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Cargando checkpoint
    best_loss = float("inf")
    start_epoch = 0

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint.get('best_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"🔄 Reanudando entrenamiento desde {checkpoint_path}, epoch {start_epoch}")

    # Genera datos de entrenamiento
    N_interior = 40000  # Puntos adentro del dominio
    N_boundary = 2000    # Puntos para condiciones de borde
    N_initial = 10000     # Puntos para condiciones iniciales

    # Sampleo (x, y, t) en el dominio interior (0,1)x(0,1)x(0,1)
    x_interior, y_interior, t_interior = model.sample_by_nonlinearity(N_interior)
    x_interior.requires_grad, y_interior.requires_grad, t_interior.requires_grad = True, True, True

    # Puntos de condiciones iniciales (t=0)
    x_init, y_init, t_init = model.sample_by_nonlinearity(N_initial-1, initial_points=True)

    # Agregar el punto de ignición manualmente
    x_ignition = torch.tensor([[mean_x]], device=device)
    y_ignition = torch.tensor([[mean_y]], device=device)

    # Concatenar el punto de ignición con los demás puntos
    x_init = torch.cat([x_init, x_ignition], dim=0)
    y_init = torch.cat([y_init, y_ignition], dim=0)
    t_init = torch.cat([t_init, torch.zeros_like(x_ignition)], dim=0)

    I_init = torch.exp(-0.5 * (((x_init - x_ignition) / sigma_x) ** 2 + ((y_init - y_ignition) / sigma_y) ** 2))
    S_init = 1 - I_init
    R_init = torch.zeros_like(I_init)

    # Sampleo puntos del borde
    x_left = torch.zeros(N_boundary, 1, device=device) # x=0
    x_right = domain_size*torch.ones(N_boundary, 1, device=device) # x=1
    y_top = domain_size*torch.ones(N_boundary, 1, device=device) # y=1
    y_bottom = torch.zeros(N_boundary, 1, device=device) # y=0
    x_boundary = domain_size*torch.rand(N_boundary, 1, device=device) # x en (0, 1)
    y_boundary = domain_size*torch.rand(N_boundary, 1, device=device) # y en (0, 1)
    t_boundary = temporal_domain*torch.rand(N_boundary, 1, device=device) # t en (0, 1)

    loss_phys_list, loss_bc_list, loss_ic_list, loss_data_list = [], [], [], []

    best_loss = float('inf')
    best_model_state = None

    temporal_weights = torch.ones(N_blocks, device=device)

    print(f"Entrenando PINNs con D = {self.D_I}, beta = {self.beta}, gamma = {self.gamma}")

    for epoch in range(start_epoch, epochs_adam):
        if epoch % 500 == 0 and epoch > 0: # Sampleo adaptativo cada 500 épocas
            x_interior, y_interior, t_interior = model.sample_by_nonlinearity(N_interior)
            x_interior.requires_grad, y_interior.requires_grad, t_interior.requires_grad = True, True, True

            x_init, y_init, t_init = model.sample_by_nonlinearity(N_initial-1, initial_points=True)

            x_init = torch.cat([x_init, x_ignition], dim=0)
            y_init = torch.cat([y_init, y_ignition], dim=0)
            t_init = torch.cat([t_init, torch.zeros_like(x_ignition)], dim=0)

            I_init = torch.exp(-0.5 * (((x_init - x_ignition) / sigma_x) ** 2 + ((y_init - y_ignition) / sigma_y) ** 2))
            S_init = 1 - I_init
            R_init = torch.zeros_like(I_init)

            print(f"[Epoch {epoch}] Sampleo adaptativo realizado.")
            # Liberar caché de CUDA para reducir fragmentación
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        data = {
            'ic': (x_init, y_init, t_init, S_init, I_init, R_init),
            'bc': (y_top, y_bottom, x_left, x_right, x_boundary, y_boundary, t_boundary),
            'phys': (x_interior, y_interior, t_interior),
            'data': (t_data, S_data, I_data, R_data) if model.mode == 'inverse' else None,
        }

        params = (self.D_I, self.beta, self.gamma, temporal_weights, N_blocks)

        total_loss, loss_phys, loss_ic, loss_bc, loss_data, temporal_losses = model.closure(optimizer, data, params)

        # Actualización de pesos temporales (con normalización para estabilidad)
        partial_sums = torch.cumsum(temporal_losses.detach(), dim=0)  # sumatoria acumulada por bloques
        temporal_weights = torch.exp(-partial_sums)                   # exp(-sum) por bloque
        temporal_weights[0] = 1.0
        temporal_weights = temporal_weights / (temporal_weights.mean() + 1e-12)

        # Actualización de pesos (usar el grafo antes del step)
        # if epoch % 1000 == 0 and epoch > 0:
        #     # Computamos los pesos nuevos
        #     model.w_ic, model.w_bc, model.w_pde = model.update_loss_weights(loss_ic, loss_bc, loss_phys)
        #     print(f"[Epoch {epoch}] Pesos actualizados: w_pde = {model.w_pde}, w_ic = {model.w_ic}, w_bc = {model.w_bc}")

        optimizer.step()

        # Guardar cada pérdida individual
        loss_phys_list.append(loss_phys.item())
        loss_ic_list.append(loss_ic.item())
        loss_bc_list.append(loss_bc.item())
        loss_data_list.append(loss_data.item())

        # 📌 Guardar el mejor modelo
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}) # Transfiere el modelo a la CPU

        if epoch % 100 == 0 or epoch == epochs_adam - 1:
            print(f"Adam Época {epoch} | Loss: {total_loss.item()} | PDE Loss: {loss_phys.item()} | IC Loss: {loss_ic.item()} | BC Loss: {loss_bc.item()}")

        last_epoch = epoch

    np.save("loss_phys.npy", np.array(loss_phys_list))
    np.save("loss_ic.npy", np.array(loss_ic_list))
    np.save("loss_bc.npy", np.array(loss_bc_list))
    np.save("loss_data.npy", np.array(loss_data_list))

    # Restaurar el mejor modelo en memoria
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.eval()

    return model, optimizer, best_loss, last_epoch