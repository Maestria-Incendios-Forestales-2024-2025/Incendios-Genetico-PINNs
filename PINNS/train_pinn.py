import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F
import numpy as np # type: ignore
import copy

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## DEFINICIÓN DE LA PINN ###############################################

temporal_domain = 10
domain_size = 2

# # Definir la red neuronal PINN
# class FireSpread_PINN(nn.Module):
#     def __init__(self, d=3, m=50, layers=[3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 3], mu = 1.0, sigma = 0.1):
#         super().__init__()

#         # d = dimensión de entrada original (x,y,t)
#         # m = número de frecuencias para el embedding

#         # matriz B ~ N(0, sigma^2)
#         self.B = nn.Parameter(torch.randn(m, d) * 10.0, requires_grad = False)

#         # arquitectura de la red (entrada = 2m, por sin/cos)
#         input_dim = 2 * m
#         full_layers = [input_dim] + layers

#         self.s_list = nn.ParameterList()  # escalares
#         self.V_list = nn.ParameterList()  # pesos fijos
#         self.bias_list = nn.ParameterList()

#         for i in range(len(full_layers) - 1):
#             out_dim = full_layers[i+1]
#             in_dim = full_layers[i]

#             # s^(l) ~ N(mu, sigma)
#             s = nn.Parameter(torch.randn(out_dim) * sigma + mu)
#             self.s_list.append(s)

#             # V^(l) inicializado Glorot
#             V = nn.Parameter(torch.empty(out_dim, in_dim))
#             nn.init.xavier_uniform_(V)
#             V.requires_grad = False  # fijo
#             self.V_list.append(V)

#             # bias entrenable
#             b = nn.Parameter(torch.zeros(out_dim))
#             self.bias_list.append(b)

#         self.activation = nn.Tanh()

#     def fourier_features(self, x):
#         # x: (batch_size, d)
#         # Bx -> (batch_size, m)
#         proj = 2 * torch.pi * x @ self.B.T
#         return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

#     def forward(self, x, y, t):
#         inputs = torch.cat((x, y, t), dim=1)
#         out = self.fourier_features(inputs)

#         for s, V, b in zip(self.s_list[:-1], self.V_list[:-1], self.bias_list[:-1]):
#             W = torch.diag(torch.exp(s)) @ V  # factoriza pesos
#             out = self.activation(F.linear(out, W, b))

#         # última capa
#         W = torch.diag(torch.exp(self.s_list[-1])) @ self.V_list[-1]
#         out = F.linear(out, W, self.bias_list[-1])
#         return out

# Definir la red neuronal PINN
class FireSpread_PINN(nn.Module):
    def __init__(self, d=3, m=250, hidden=[100]*11, out_dim=3):
        super().__init__()

        # d = dimensión de entrada original (x,y,t)
        # m = número de frecuencias para el embedding

        # matriz B ~ N(0, sigma^2)
        self.B = nn.Parameter(torch.randn(m, d) * 10.0, requires_grad = False)

        # arquitectura de la red (entrada = 2m, por sin/cos)
        input_dim = 2 * m
        full_layers = [input_dim] + hidden + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(full_layers[i], full_layers[i+1]) for i in range(len(full_layers)-1)]
        )
        self.activation = nn.Tanh()

    def fourier_features(self, x):
        # x: (batch_size, d)
        # Bx -> (batch_size, m)
        proj = 2 * torch.pi * x @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x, y, t):
        inputs = torch.cat((x, y, t), dim=1) # (batch, 3)
        ff = self.fourier_features(inputs) # (batch, 2m)
        out = ff
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        return self.layers[-1](out)

############################## PÉRDIDA POR CONDICIONES INICIALES ###############################################

def loss_initial_condition(model, x_ic, y_ic, t_ic, S0, I0, R0):
    pred = model(x_ic, y_ic, t_ic)
    S_pred, I_pred, R_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    return nn.MSELoss()(S_pred, S0) + nn.MSELoss()(I_pred, I0) + nn.MSELoss()(R_pred, R0)

############################## PÉRDIDA POR CONDICIONES DE BORDE ###############################################

def loss_boundary_condition(model, y_top, y_bottom, x_left, x_right, x_bc, y_bc, t_bc):
    top_pred = model(x_bc, y_top, t_bc) # Borde de arriba (x,y)=(x,1)
    S_top_pred, I_top_pred, R_top_pred = top_pred[:, 0:1], top_pred[:, 1:2], top_pred[:, 2:3]

    bottom_pred = model(x_bc, y_bottom, t_bc) # Borde de abajo (x,y)=(x,0)
    S_bottom_pred, I_bottom_pred, R_bottom_pred = bottom_pred[:, 0:1], bottom_pred[:, 1:2], bottom_pred[:, 2:3]

    left_pred = model(x_left, y_bc, t_bc) # Borde de la izquierda (x,y)=(0,y)
    S_left_pred, I_left_pred, R_left_pred = left_pred[:, 0:1], left_pred[:, 1:2], left_pred[:, 2:3]

    right_pred = model(x_right, y_bc, t_bc) # Borde de la derecha (x,y)=(1,y)
    S_right_pred, I_right_pred, R_right_pred = right_pred[:, 0:1], right_pred[:, 1:2], right_pred[:, 2:3]

    # Pérdida por condiciones de borde
    loss_top_bc = nn.MSELoss()(S_top_pred, S_bottom_pred) + nn.MSELoss()(I_top_pred, I_bottom_pred) + nn.MSELoss()(R_top_pred, R_bottom_pred)
    loss_left_bc = nn.MSELoss()(S_left_pred, S_right_pred) + nn.MSELoss()(I_left_pred, I_right_pred) + nn.MSELoss()(R_left_pred, R_right_pred)

    return loss_top_bc + loss_left_bc

############################## PÉRDIDA POR LA FÍSICA ###############################################

def loss_pde(model, x_phys, y_phys, t_phys, D_I, beta_val, gamma_val):
    x_phys.requires_grad = True
    y_phys.requires_grad = True
    t_phys.requires_grad = True

    SIR_pred = model(x_phys, y_phys, t_phys)
    S_pred, I_pred, R_pred = SIR_pred[:, 0:1], SIR_pred[:, 1:2], SIR_pred[:, 2:3]

    dS_dt = torch.autograd.grad(S_pred, t_phys, torch.ones_like(S_pred), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I_pred, t_phys, torch.ones_like(I_pred), create_graph=True)[0]
    dR_dt = torch.autograd.grad(R_pred, t_phys, torch.ones_like(R_pred), create_graph=True)[0]

    dI_dx = torch.autograd.grad(I_pred, x_phys, torch.ones_like(I_pred), create_graph=True)[0]
    dI_dy = torch.autograd.grad(I_pred, y_phys, torch.ones_like(I_pred), create_graph=True)[0]

    d2I_dx2 = torch.autograd.grad(dI_dx, x_phys, torch.ones_like(dI_dx), create_graph=True)[0]
    d2I_dy2 = torch.autograd.grad(dI_dy, y_phys, torch.ones_like(dI_dy), create_graph=True)[0]

    loss_S = dS_dt + beta_val * S_pred * I_pred
    loss_I = dI_dt - (beta_val * S_pred * I_pred - gamma_val * I_pred) - D_I * (d2I_dx2 + d2I_dy2)
    loss_R = dR_dt - gamma_val * I_pred

    return (loss_S**2 + loss_I**2 + loss_R**2).mean()
    
############################## PÉRDIDA POR INCONSISTENCIAS ###############################################

def non_negative_loss(S, I, R):
    loss = torch.mean(torch.relu(-S)) + torch.mean(torch.relu(-I)) + torch.mean(torch.relu(-R))
    return loss

############################## CIERRE DE OPTIMIZACIÓN ###############################################

def closure(model, optimizer, data, params):
    loss_ic = loss_initial_condition(model, *data['ic'])
    loss_bc = loss_boundary_condition(model, *data['bc'])
    loss_phys = loss_pde(model, *data['phys'], *params)
    return loss_phys, loss_ic, loss_bc

############################## SAMPLEO POR NO LINEALIDAD ###############################################

def sample_by_nonlinearity(model, N_samples, N_candidates=100000, initial_points=False):
    # Step 1: Generate candidates
    x = domain_size*torch.rand(N_candidates, 1, device=device)
    y = domain_size*torch.rand(N_candidates, 1, device=device)
    if initial_points:
        t = torch.zeros(N_candidates, 1, device=device)
    else:
        t = temporal_domain*torch.rand(N_candidates, 1, device=device)

    with torch.no_grad():
        pred = model(x, y, t)
        S_pred, I_pred = pred[:, 0:1], pred[:, 1:2]
        nonlin_strength = torch.abs(S_pred * I_pred).squeeze()  # Shape: (N_candidates,)

    # Step 2: Normalize weights to sum to 1
    weights = nonlin_strength / nonlin_strength.sum()

    # Step 3: Sample indices using importance
    idx = torch.multinomial(weights, N_samples, replacement=False)

    # Step 4: Return the selected points
    return x[idx], y[idx], t[idx]

############################## ENTRENAMIENTO ###############################################

# Función para entrenar la PINN con ecuaciones de propagación de fuego
def train_pinn(D_I, beta_val, gamma_val, mean_x, mean_y, sigma_x, sigma_y, epochs_adam=1000, N_blocks=10):
    model = FireSpread_PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Genera datos de entrenamiento
    N_interior = 40000  # Puntos adentro del dominio
    N_boundary = 2000    # Puntos para condiciones de borde
    N_initial = 10000     # Puntos para condiciones iniciales

    # Sampleo (x, y, t) en el dominio interior (0,1)x(0,1)x(0,1)
    x_interior, y_interior, t_interior = sample_by_nonlinearity(model, N_interior)
    x_interior.requires_grad, y_interior.requires_grad, t_interior.requires_grad = True, True, True

    # Puntos de condiciones iniciales (t=0)
    x_init, y_init, t_init = sample_by_nonlinearity(model, N_initial-1, initial_points=True)

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

    # loss_phys_list, loss_bc_list, loss_ic_list, loss_nonneg_list = [], [], [], []
    loss_phys_list, loss_bc_list, loss_ic_list = [], [], []

    best_loss = float('inf')
    best_model_state = None

    print(f"Entrenando PINNs con D = {D_I}, beta = {beta_val}, gamma = {gamma_val}")

    # --------- Primera etapa: Adam ---------
    for epoch in range(epochs_adam):
        if epoch % 200 == 0 and epoch > 0: # Sampleo adaptativo cada 200 épocas
            x_interior, y_interior, t_interior = sample_by_nonlinearity(model, N_interior)
            x_interior.requires_grad = y_interior.requires_grad = t_interior.requires_grad = True

            x_init, y_init, t_init = sample_by_nonlinearity(model, N_initial-1, initial_points=True)

            x_init = torch.cat([x_init, x_ignition], dim=0)
            y_init = torch.cat([y_init, y_ignition], dim=0)
            t_init = torch.cat([t_init, torch.zeros_like(x_ignition)], dim=0)

            I_init = torch.exp(-0.5 * (((x_init - x_ignition) / sigma_x) ** 2 + ((y_init - y_ignition) / sigma_y) ** 2))
            S_init = 1 - I_init
            R_init = torch.zeros_like(I_init)

            print(f"[Epoch {epoch}] Sampleo adaptativo realizado.")

        data = {
            'ic': (x_init, y_init, t_init, S_init, I_init, R_init),
            'bc': (y_top, y_bottom, x_left, x_right, x_boundary, y_boundary, t_boundary),
            'phys': (x_interior, y_interior, t_interior)
        }

        params = (D_I, beta_val, gamma_val)

        optimizer.zero_grad()
        loss_phys, loss_ic, loss_bc = closure(model, optimizer, data, params)

        # Penalizamos los valores negativos
        # x_interior.requires_grad = True
        # y_interior.requires_grad = True
        # t_interior.requires_grad = True
        # SIR_pred = model(x_interior, y_interior, t_interior)
        # S_pred, I_pred, R_pred = SIR_pred[:, 0:1], SIR_pred[:, 1:2], SIR_pred[:, 2:3]
        # loss_nonneg = non_negative_loss(S_pred, I_pred, R_pred)

        loss = loss_phys + loss_ic + loss_bc #+ loss_nonneg
        loss.backward()
        optimizer.step()

        # Guardar cada pérdida individual
        loss_phys_list.append(loss_phys.item())
        loss_ic_list.append(loss_ic.item())
        loss_bc_list.append(loss_bc.item())
        # loss_nonneg_list.append(loss_nonneg.item())

        # 📌 Guardar el mejor modelo
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
            # print(f"✅ Mejor modelo guardado en epoch {epoch} con pérdida total {best_loss:.2e}")

        if epoch % 100 == 0 or epoch == epochs_adam - 1:
            print(f"Adam Época {epoch} | Loss: {loss.item()}")

    np.save("loss_phys.npy", np.array(loss_phys_list))
    np.save("loss_ic.npy", np.array(loss_ic_list))
    np.save("loss_bc.npy", np.array(loss_bc_list))
    # np.save("loss_nonneg.npy", np.array(loss_nonneg_list))

    # Restaurar el mejor modelo en memoria
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.eval()

    return model, best_loss

if __name__ == "__main__":
    model = FireSpread_PINN().to(device)
    print(model)