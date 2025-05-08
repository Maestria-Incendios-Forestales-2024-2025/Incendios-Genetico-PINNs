import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## DEFINICIÓN DE LA PINN ###############################################

# Definir la red neuronal PINN
class FireSpread_PINN(nn.Module):
    def __init__(self):
        super(FireSpread_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, x, y, t):
        inputs = torch.cat((x, y, t), dim=1)
        return self.net(inputs)

############################## FUNCIÓN DE ENTRENAMIENTO ###############################################

# Función para entrenar la PINN con ecuaciones de propagación de fuego
def train_pinn(beta_val, gamma_val, D_I, mean_x, mean_y, sigma_x, sigma_y, epochs_adam=1000, N_blocks=10):
    model = FireSpread_PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Genera datos de entrenamiento
    N_interior = 20000  # Puntos adentro del dominio
    N_boundary = 2000    # Puntos para condiciones de borde
    N_initial = 4000     # Puntos para condiciones iniciales

    # Sampleo (x, y, t) en el dominio interior (0,1)x(0,1)x(0,1)
    # x_interior = torch.rand(N_interior, 1, device=device)
    # y_interior = torch.rand(N_interior, 1, device=device)
    # t_interior = torch.rand(N_interior, 1, device=device)
    # x_interior.requires_grad, y_interior.requires_grad, t_interior.requires_grad = True, True, True

    # Puntos de condiciones iniciales (t=0)
    x_init = torch.rand(N_initial-1, 1, device=device)
    y_init = torch.rand(N_initial-1, 1, device=device)
    t_init = torch.zeros(N_initial, 1, device=device)

    # Agregar el punto de ignición manualmente
    x_ignition = torch.tensor([[mean_x]], device=device)
    y_ignition = torch.tensor([[mean_y]], device=device)

    # Concatenar el punto de ignición con los demás puntos
    x_init = torch.cat([x_init, x_ignition], dim=0)
    y_init = torch.cat([y_init, y_ignition], dim=0)

    I_init = torch.exp(-0.5 * (((x_init - x_ignition) / sigma_x) ** 2 + ((y_init - y_ignition) / sigma_y) ** 2))
    S_init = 1 - I_init
    R_init = torch.zeros_like(I_init)

    # Sampleo puntos del borde
    # x_left = torch.zeros(N_boundary, 1, device=device) # x=0
    # x_right = torch.ones(N_boundary, 1, device=device) # x=1
    # y_top = torch.ones(N_boundary, 1, device=device) # y=1
    # y_bottom = torch.zeros(N_boundary, 1, device=device) # y=0
    # x_boundary = torch.rand(N_boundary, 1, device=device) # x en (0, 1)
    # y_boundary = torch.rand(N_boundary, 1, device=device) # y en (0, 1)
    # t_boundary = torch.rand(N_boundary, 1, device=device) # t en (0, 1)

    T_final = 1.0
    t_blocks = torch.linspace(0, T_final, N_blocks + 1, device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(N_blocks):
        t0, t1 = t_blocks[i].item(), t_blocks[i+1].item()

        # Puntos temporales para este bloque
        t_interior = t0 + (t1 - t0) * torch.rand(N_interior, 1, device=device)
        x_interior = torch.rand(N_interior, 1, device=device)
        y_interior = torch.rand(N_interior, 1, device=device)
        x_interior.requires_grad, y_interior.requires_grad, t_interior.requires_grad = True, True, True

        t_init = torch.full((N_initial, 1), t0, device=device)
    
        # --------- Cierre de optimización ---------
        last_loss = None
        def closure():
            nonlocal last_loss

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
            loss_S = dS_dt + beta_val * S_pred * I_pred
            loss_I = dI_dt - (beta_val * S_pred * I_pred - gamma_val * I_pred) - D_I * (d2I_dx2 + d2I_dy2) 
            loss_R = dR_dt - gamma_val * I_pred

            loss_pde = loss_S.pow(2).mean() + loss_I.pow(2).mean() + loss_R.pow(2).mean()

            # Calcular salida de la red en t=0 para condición inicial
            SIR_init_pred = model(x_init, y_init, t_init)
            S_init_pred, I_init_pred, R_init_pred = SIR_init_pred[:, 0:1], SIR_init_pred[:, 1:2], SIR_init_pred[:, 2:3]

            # Pérdida por condición inicial (forzar que la red prediga bien los valores iniciales)
            loss_ic = (S_init_pred - S_init).pow(2).mean() + (I_init_pred - I_init).pow(2).mean() + (R_init_pred - R_init).pow(2).mean()

            # # Calcular salida de la red para los bordes
            # SIR_top_pred = model(x_boundary, y_top, t_boundary) # Borde de arriba (x,y)=(x,1)
            # S_top_pred, I_top_pred, R_top_pred = SIR_top_pred[:, 0:1], SIR_top_pred[:, 1:2], SIR_top_pred[:, 2:3]

            # SIR_bottom_pred = model(x_boundary, y_bottom, t_boundary) # Borde de abajo (x,y)=(x,0)
            # S_bottom_pred, I_bottom_pred, R_bottom_pred = SIR_bottom_pred[:, 0:1], SIR_bottom_pred[:, 1:2], SIR_bottom_pred[:, 2:3]

            # SIR_left_pred = model(x_left, y_boundary, t_boundary) # Borde de la izquierda (x,y)=(0,y)
            # S_left_pred, I_left_pred, R_left_pred = SIR_left_pred[:, 0:1], SIR_left_pred[:, 1:2], SIR_left_pred[:, 2:3]

            # SIR_right_pred = model(x_right, y_boundary, t_boundary) # Borde de la derecha (x,y)=(1,y)
            # S_right_pred, I_right_pred, R_right_pred = SIR_right_pred[:, 0:1], SIR_right_pred[:, 1:2], SIR_right_pred[:, 2:3]

            # # Pérdida por condiciones de borde
            # loss_top_bc = (S_top_pred).pow(2).mean() + (I_top_pred).pow(2).mean() + (R_top_pred).pow(2).mean()
            # loss_bottom_bc = (S_bottom_pred).pow(2).mean() + (I_bottom_pred).pow(2).mean() + (R_bottom_pred).pow(2).mean()
            # loss_left_bc = (S_left_pred).pow(2).mean() + (I_left_pred).pow(2).mean() + (R_left_pred).pow(2).mean()
            # loss_right_bc = (S_right_pred).pow(2).mean() + (I_right_pred).pow(2).mean() + (R_right_pred).pow(2).mean()

            # loss_bc = loss_top_bc + loss_bottom_bc + loss_left_bc + loss_right_bc

            # Pérdida total
            loss = loss_pde + loss_ic #+ loss_bc
            loss.backward()
            last_loss = loss.item()
            return loss
    
        # --------- Primera etapa: Adam ---------
        for epoch in range(epochs_adam):
            optimizer.zero_grad()
            loss = closure()
            optimizer.step()
            if epoch % 100 == 0 or epoch == epochs_adam - 1:
                print(f"[Bloque {i+1}/{N_blocks}] Adam Época {epoch} | Loss: {loss.item():.6f}")

        # Predicción al final del bloque como nueva condición inicial
        t_next = torch.full((N_initial, 1), t1, device=device)
        x_init = torch.rand(N_initial, 1, device=device)
        y_init = torch.rand(N_initial, 1, device=device)
        with torch.no_grad():
            SIR_next = model(x_init, y_init, t_next)
        S_init = SIR_next[:, 0:1].detach()
        I_init = SIR_next[:, 1:2].detach()
        R_init = SIR_next[:, 2:3].detach()

    return model, loss