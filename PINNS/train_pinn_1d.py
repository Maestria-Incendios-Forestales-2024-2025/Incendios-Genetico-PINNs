import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import cupy as cp # type: ignore

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## DEFINICIÓN DE LA PINN ###############################################

# Definir la red neuronal PINN
class FireSpread_PINN(nn.Module):
    def __init__(self):
        super(FireSpread_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  # Entrada (x, t)
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # Salida (S, I)
            nn.Sigmoid()  # Asegurar valores entre 0 y 1
        )

    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        return self.net(inputs)

############################## FUNCIÓN DE ENTRENAMIENTO ###############################################

# Función para entrenar la PINN con ecuaciones de propagación de fuego
def train_pinn(D_I, epochs_adam=1000, epochs_lbfgs=20):
    model = FireSpread_PINN().to(device)

    # Configuración del dominio
    N_interior, N_boundary, N_initial = 20000, 2000, 4000

    x_ignition = torch.tensor([[0.4]], device=device)

    x_init = torch.rand(N_initial - 1, 1, device=device)
    t_init = torch.zeros(N_initial, 1, device=device)
    x_init = torch.cat([x_init, x_ignition], dim=0)

    sigma_x = 0.05
    I_init = torch.exp(-0.5 * ((x_init - x_ignition) / sigma_x) ** 2)
    S_init = 1 - I_init

    beta_val = 0.3
    gamma_val = 0.1

    x_interior = torch.rand(N_interior, 1, device=device)
    t_interior = torch.rand(N_interior, 1, device=device)
    x_interior.requires_grad, t_interior.requires_grad = True, True

    beta_sampled = torch.full((N_interior, 1), beta_val, device=device)
    gamma_sampled = torch.full((N_interior, 1), gamma_val, device=device)

    x_left = torch.zeros(N_boundary, 1, device=device)
    x_right = torch.ones(N_boundary, 1, device=device)
    t_boundary = torch.rand(N_boundary, 1, device=device)

    # --------- Cierre de optimización ---------
    last_loss = None
    def closure():
        nonlocal last_loss

        SIR_pred = model(x_interior, t_interior)
        S_pred, I_pred = SIR_pred[:, 0:1], SIR_pred[:, 1:2]

        dS_dt = torch.autograd.grad(S_pred, t_interior, torch.ones_like(S_pred), create_graph=True)[0]
        dI_dt = torch.autograd.grad(I_pred, t_interior, torch.ones_like(I_pred), create_graph=True)[0]
        dI_dx = torch.autograd.grad(I_pred, x_interior, torch.ones_like(I_pred), create_graph=True)[0]
        d2I_dx2 = torch.autograd.grad(dI_dx, x_interior, torch.ones_like(dI_dx), create_graph=True)[0]

        loss_S = dS_dt + beta_sampled * S_pred * I_pred
        loss_I = dI_dt - (beta_sampled * S_pred * I_pred - gamma_sampled * I_pred) - D_I * d2I_dx2
        loss_pde = loss_S.pow(2).mean() + loss_I.pow(2).mean()

        SIR_init_pred = model(x_init, t_init)
        S_init_pred, I_init_pred = SIR_init_pred[:, 0:1], SIR_init_pred[:, 1:2]
        loss_ic = (S_init_pred - S_init).pow(2).mean() + (I_init_pred - I_init).pow(2).mean()

        S_left_pred, I_left_pred = model(x_left, t_boundary)[:, 0:1], model(x_left, t_boundary)[:, 1:2]
        S_right_pred, I_right_pred = model(x_right, t_boundary)[:, 0:1], model(x_right, t_boundary)[:, 1:2]
        loss_bc = (S_left_pred**2).mean() + (I_left_pred**2).mean() + (S_right_pred**2).mean() + (I_right_pred**2).mean()

        loss = loss_pde + loss_ic + loss_bc
        loss.backward()
        last_loss = loss.item()
        return loss

    # --------- Primera etapa: Adam ---------
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs_adam):
        optimizer.zero_grad()
        loss = closure()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs_adam - 1:
            print(f"[Adam] Época {epoch} | Loss: {loss.item():.6f}")

    #--------- Segunda etapa: LBFGS ---------
    # optimizer = optim.LBFGS(
    #     model.parameters(),
    #     lr=2.0,
    #     max_iter=500,
    #     history_size=50,
    #     tolerance_grad=1e-10,
    #     tolerance_change=1e-9,
    #     line_search_fn="strong_wolfe"
    # )

    # for epoch in range(epochs_lbfgs):
    #     optimizer.step(closure)
    #     print(f"[LBFGS] Época {epoch} | Loss final: {last_loss:.6f}")

    return model