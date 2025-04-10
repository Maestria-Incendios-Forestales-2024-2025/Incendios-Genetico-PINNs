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
def train_pinn(D_I, epochs_adam=500, epochs_lbfgs=500):
    model = FireSpread_PINN().to(device)
    model.double()

    # Genera datos de entrenamiento
    N_interior = 20000  # Puntos adentro del dominio
    N_boundary = 2000    # Puntos para condiciones de borde
    N_initial = 2000     # Puntos para condiciones iniciales

    Nx = 1768

    # Agregar el punto de ignición manualmente
    x_ignition = torch.tensor([[700.0]], device=device).double() / Nx

    # Puntos de condiciones iniciales (t=0)
    x_init = torch.rand(N_initial-1, 1, device=device).double() #* Nx
    t_init = torch.zeros(N_initial, 1, device=device).double()

    # Concatenar el punto de ignición con los demás puntos (normalizado)
    x_init = torch.cat([x_init, x_ignition], dim=0).double()

    # Condiciones iniciales (en t=0)
    sigma_x = 1 / Nx # Tamaño de la celda normalizada

    I_init = torch.exp(-1/2*(((x_init - x_ignition) / sigma_x)**2))
    S_init = 1 - I_init

    beta_val = 0.3
    gamma_val = 0.1

    num_blocks = 10
    T_block = 1/10

    for block in range(num_blocks):
        t_start = block * T_block
        t_end = (block + 1) * T_block

        # Sampleo (x, t) en el dominio interior (0,1768)x(0,1000) normalizadas al (0,1)x(0,1)
        x_interior = torch.rand(N_interior, 1, device=device).double() #* Nx
        t_interior = t_start + (t_end - t_start) * torch.rand(N_interior, 1, device=device).double() #* T_max

        x_interior.requires_grad, t_interior.requires_grad = True, True

        # Genero un tensor para beta y gamma con valor constante en todo el mapa
        beta_sampled = torch.full((N_interior,1), beta_val, device=device).double()
        gamma_sampled = torch.full((N_interior,1), gamma_val, device=device).double()

        # Sampleo condiciones de borde
        x_left = torch.zeros(N_boundary, 1, device=device).double() # x=0
        x_right = torch.ones(N_boundary, 1, device=device).double() # x=1768
        t_boundary = t_start + (t_end - t_start) * torch.rand(N_boundary, 1, device=device).double()

        def closure():
            optimizer.zero_grad()

            SIR_pred = model(x_interior, t_interior)
            S_pred, I_pred = SIR_pred[:, 0:1], SIR_pred[:, 1:2]

            dS_dt = torch.autograd.grad(S_pred, t_interior, torch.ones_like(S_pred), create_graph=True)[0]
            dI_dt = torch.autograd.grad(I_pred, t_interior, torch.ones_like(I_pred), create_graph=True)[0]
            dI_dx = torch.autograd.grad(I_pred, x_interior, torch.ones_like(I_pred), create_graph=True)[0]
            d2I_dx2 = torch.autograd.grad(dI_dx, x_interior, torch.ones_like(dI_dx), create_graph=True)[0]

            loss_S = dS_dt + beta_sampled * S_pred * I_pred
            loss_I = dI_dt - (beta_sampled * S_pred * I_pred - gamma_sampled * I_pred) - D_I * d2I_dx2
            loss_pde = loss_S.pow(2).mean() + loss_I.pow(2).mean()

            t_init_block = torch.full_like(x_init, t_start).double()
            SIR_init_pred = model(x_init, t_init_block)
            S_init_pred, I_init_pred = SIR_init_pred[:, 0:1], SIR_init_pred[:, 1:2]
            loss_ic = (S_init_pred - S_init).pow(2).mean() + (I_init_pred - I_init).pow(2).mean()

            S_left_pred, I_left_pred = model(x_left, t_boundary)[:, 0:1], model(x_left, t_boundary)[:, 1:2]
            S_right_pred, I_right_pred = model(x_right, t_boundary)[:, 0:1], model(x_right, t_boundary)[:, 1:2]
            loss_bc = (S_left_pred**2).mean() + (I_left_pred**2).mean() + (S_right_pred**2).mean() + (I_right_pred**2).mean()

            loss = loss_pde + loss_ic + loss_bc
            loss.backward()
            return loss

        # Etapa 1: Adam
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs_adam):
            loss = closure()
            optimizer.step()
            optimizer.zero_grad()

        # Etapa 2: LBFGS
        optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=epochs_lbfgs, history_size=50, line_search_fn='strong_wolfe')
        optimizer.step(closure)

        print(f"Bloque {block + 1} terminado | Loss final: {closure().item():.6f}")

        t_next = torch.full_like(x_init, t_end)
        with torch.no_grad():
            SIR_next = model(x_init, t_next)
            S_init = SIR_next[:, 0:1].detach()
            I_init = SIR_next[:, 1:2].detach()

    return model