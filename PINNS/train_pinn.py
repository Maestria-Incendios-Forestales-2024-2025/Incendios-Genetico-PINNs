import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import cupy as cp # type: ignore
import torch.nn.functional as F # type: ignore

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## DEFINICIÓN DE LA PINN ###############################################

# Definir la red neuronal PINN
class FireSpread_PINN(nn.Module):
    def __init__(self):
        super(FireSpread_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # Entrada (t, x, y)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),  # Salida (S, I, R)
            nn.Sigmoid()  # Asegurar valores entre 0 y 1
        )

    def forward(self, x, y, t):
        inputs = torch.cat((x, y, t), dim=1)
        return self.net(inputs)

############################## ENTRENAMIENTO DE LA PINN ###############################################

# Función para entrenar la PINN con ecuaciones de propagación de fuego
def train_pinn(beta, gamma, D_I, A, wx, h_dx, B, wy, h_dy, epochs=1):
    model = FireSpread_PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Genera datos de entrenamiento
    N_interior = 10000  # Puntos adentro del dominio
    N_boundary = 2000    # Puntos para condiciones de borde
    N_initial = 2000     # Puntos para condiciones iniciales

    dt = 1/6  # Paso temporal en horas (10 minutos)
    num_T = 1000  # Cantidad de pasos de tiempo
    T_max = num_T * dt  # Tiempo total simulado en horas (~6 días)
    Nx, Ny = 1768, 1060

    # Sampleo (x, y, t) en el dominio interior (0,1768)x(0,1060)x(0,1000)
    x_interior = torch.rand(N_interior, 1, device=device) * Nx
    y_interior = torch.rand(N_interior, 1, device=device) * Ny
    t_interior = torch.rand(N_interior, 1, device=device) * T_max

    # Puntos de condiciones iniciales (t=0)
    x_init = torch.rand(N_initial-1, 1, device=device) * Nx
    y_init = torch.rand(N_initial-1, 1, device=device) * Ny
    t_init = torch.zeros(N_initial, 1, device=device)

    # Agregar el punto de ignición manualmente
    x_ignition = torch.tensor([[700.0]], device=device)
    y_ignition = torch.tensor([[700.0]], device=device)

    # Concatenar el punto de ignición con los demás puntos
    x_init = torch.cat([x_init, x_ignition], dim=0)
    y_init = torch.cat([y_init, y_ignition], dim=0)

    # Mapas iniciales (condiciones iniciales en t=0)
    S_init = torch.ones_like(x_init) # Todo combustible
    I_init = torch.zeros_like(x_init) # Nada incendiado
    R_init = torch.zeros_like(x_init) # Nada "recuperado" aún

    # Asignar la ignición al último punto (el que agregamos)
    S_init[-1] = 0 # Todo combustible menos ignición
    I_init[-1] = 1 # Todo vacío excepto punto de ignición

    # Sampleo puntos del borde
    x_left = torch.zeros(N_boundary, 1, device=device) # x=0
    x_right = torch.ones(N_boundary, 1, device=device) * Nx # x=1768
    y_top = torch.ones(N_boundary, 1, device=device) * Ny # y=1060
    y_bottom = torch.zeros(N_boundary, 1, device=device) # y=0
    x_boundary = torch.rand(N_boundary, 1, device=device) * Nx # x en (0, 1768)
    y_boundary = torch.rand(N_boundary, 1, device=device) * Ny # y en (0, 1060)
    t_boundary = torch.rand(N_boundary, 1, device=device) * T_max # t en (0, ~6 días)

    # Condiciones de borde S = I = R = 0 para todos los bordes
    S_boundary = torch.zeros_like(x_boundary)
    I_boundary = torch.zeros_like(x_boundary)
    R_boundary = torch.zeros_like(x_boundary)

    # Indexo directamente los valores de beta y gamma en los puntos de muestreo
    x_idx = x_interior.long().clamp(0, Nx-1)
    y_idx = y_interior.long().clamp(0, Ny-1)

    beta_sampled = beta[y_idx, x_idx].view(-1, 1).to(device)
    gamma_sampled = gamma[y_idx, x_idx].view(-1, 1).to(device)

    # Suponiendo que wx, wy, h_dx, h_dy están en una grilla de tamaño (1768, 1060)
    wx_tensor = wx.unsqueeze(0).unsqueeze(0)  # (1,1,1768,1060)
    wy_tensor = wy.unsqueeze(0).unsqueeze(0)
    h_dx_tensor = h_dx.unsqueeze(0).unsqueeze(0)
    h_dy_tensor = h_dy.unsqueeze(0).unsqueeze(0)

    # Normalizar coordenadas a [-1,1] para grid_sample
    x_norm = (x_interior / (Nx - 1)) * 2 - 1  # Nx=1768
    y_norm = (y_interior / (Ny - 1)) * 2 - 1  # Ny=1060
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)  # (1, N, 1, 2)

    # Interpolación bilineal
    wx_interp = F.grid_sample(wx_tensor, grid, mode='bilinear', align_corners=True).squeeze()
    wy_interp = F.grid_sample(wy_tensor, grid, mode='bilinear', align_corners=True).squeeze()
    h_dx_interp = F.grid_sample(h_dx_tensor, grid, mode='bilinear', align_corners=True).squeeze()
    h_dy_interp = F.grid_sample(h_dy_tensor, grid, mode='bilinear', align_corners=True).squeeze()

    for epoch in range(epochs):

        x_interior.requires_grad, y_interior.requires_grad, t_interior.requires_grad = True, True, True

        # Calculamos la perdida de la PDE
        SIR_pred = model(x_interior, y_interior, t_interior)
        S_pred, I_pred, R_pred = SIR_pred[:, 0:1], SIR_pred[:, 1:2], SIR_pred[:, 2:3]

        dS_dt = torch.autograd.grad(S_pred, t_interior, torch.ones_like(S_pred).to(device), create_graph=True)[0]
        dI_dt = torch.autograd.grad(I_pred, t_interior, torch.ones_like(I_pred).to(device), create_graph=True)[0]
        dR_dt = torch.autograd.grad(R_pred, t_interior, torch.ones_like(R_pred).to(device), create_graph=True)[0]

        dI_dx = torch.autograd.grad(I_pred, x_interior, torch.ones_like(I_pred).to(device), create_graph=True)[0]
        dI_dy = torch.autograd.grad(I_pred, y_interior, torch.ones_like(I_pred).to(device), create_graph=True)[0]

        d2I_dx2 = torch.autograd.grad(dI_dx, x_interior, torch.ones_like(dI_dx).to(device), create_graph=True)[0]
        d2I_dy2 = torch.autograd.grad(dI_dy, y_interior, torch.ones_like(dI_dy).to(device), create_graph=True)[0]

        # Definir pérdidas basadas en las ecuaciones diferenciales
        loss_S = dS_dt + beta_sampled * S_pred * I_pred
        loss_I = dI_dt - (beta_sampled * S_pred * I_pred - gamma_sampled * I_pred) - D_I * (d2I_dx2 + d2I_dy2) + (A * wx_interp + B * h_dx_interp) * dI_dx + (A * wy_interp + B * h_dy_interp) * dI_dy
        loss_R = dR_dt - gamma_sampled * I_pred

        loss_pde = loss_S.pow(2).mean() + loss_I.pow(2).mean() + loss_R.pow(2).mean()

        # Calcular salida de la red en t=0 para condición inicial
        SIR_init_pred = model(x_init, y_init, t_init)
        S_init_pred, I_init_pred, R_init_pred = SIR_init_pred[:, 0:1], SIR_init_pred[:, 1:2], SIR_init_pred[:, 2:3]

        # Pérdida por condición inicial (forzar que la red prediga bien los valores iniciales)
        loss_ic = (S_init_pred - S_init).pow(2).mean() + (I_init_pred - I_init).pow(2).mean() + (R_init_pred - R_init).pow(2).mean()

        # Calcular salida de la red para los bordes
        SIR_top_pred = model(x_boundary, y_top, t_boundary) # Borde de arriba (x,y)=(x,1060)
        S_top_pred, I_top_pred, R_top_pred = SIR_top_pred[:, 0:1], SIR_top_pred[:, 1:2], SIR_top_pred[:, 2:3]

        SIR_bottom_pred = model(x_boundary, y_bottom, t_boundary) # Borde de abajo (x,y)=(x,0)
        S_bottom_pred, I_bottom_pred, R_bottom_pred = SIR_bottom_pred[:, 0:1], SIR_bottom_pred[:, 1:2], SIR_bottom_pred[:, 2:3]

        SIR_left_pred = model(x_left, y_boundary, t_boundary) # Borde de la izquierda (x,y)=(0,y)
        S_left_pred, I_left_pred, R_left_pred = SIR_left_pred[:, 0:1], SIR_left_pred[:, 1:2], SIR_left_pred[:, 2:3]

        SIR_right_pred = model(x_right, y_boundary, t_boundary) # Borde de la derecha (x,y)=(1768,y)
        S_right_pred, I_right_pred, R_right_pred = SIR_right_pred[:, 0:1], SIR_right_pred[:, 1:2], SIR_right_pred[:, 2:3]

        # Pérdida por condiciones de borde
        loss_top_bc = (S_top_pred - S_boundary).pow(2).mean() + (I_top_pred - I_boundary).pow(2).mean() + (R_top_pred - R_boundary).pow(2).mean()
        loss_bottom_bc = (S_bottom_pred - S_boundary).pow(2).mean() + (I_bottom_pred - I_boundary).pow(2).mean() + (R_bottom_pred - R_boundary).pow(2).mean()
        loss_left_bc = (S_left_pred - S_boundary).pow(2).mean() + (I_left_pred - I_boundary).pow(2).mean() + (R_left_pred - R_boundary).pow(2).mean()
        loss_right_bc = (S_right_pred - S_boundary).pow(2).mean() + (I_right_pred - I_boundary).pow(2).mean() + (R_right_pred - R_boundary).pow(2).mean()

        loss_bc = loss_top_bc + loss_bottom_bc + loss_left_bc + loss_right_bc

        # Pérdida total
        loss = loss_pde + loss_ic + loss_bc

        optimizer.zero_grad()

        try:
          loss.backward()
        except RuntimeError as e:
          print(f"Error en backward: {e}")

        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model