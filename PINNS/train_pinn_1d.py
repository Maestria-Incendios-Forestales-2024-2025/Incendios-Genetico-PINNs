import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import cupy as cp # type: ignore

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

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
def train_pinn(D_I, epochs_per_block=1000):
    model = FireSpread_PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Genera datos de entrenamiento
    N_interior = 20000  # Puntos adentro del dominio
    N_boundary = 2000    # Puntos para condiciones de borde
    N_initial = 2000     # Puntos para condiciones iniciales

    Nx = 1768

    # Agregar el punto de ignición manualmente
    x_ignition = torch.tensor([[700.0]], device=device) / Nx

    # Puntos de condiciones iniciales (t=0)
    x_init = torch.rand(N_initial-1, 1, device=device) #* Nx
    t_init = torch.zeros(N_initial, 1, device=device)

    # Concatenar el punto de ignición con los demás puntos (normalizado)
    x_init = torch.cat([x_init, x_ignition], dim=0)

    # Condiciones iniciales (en t=0)
    sigma_x = 1 / Nx # Tamaño de la celda normalizada

    I_init = torch.exp(-1/2*(((x_init - x_ignition) / sigma_x)**2))
    S_init = 1 - I_init

    beta_val = 0.3
    gamma_val = 0.1

    num_blocks = 10
    T_block = 1/10

    print("print para ver si lee la función nueva")

    for block in range(num_blocks):
        t_start = block * T_block
        t_end = (block + 1) * T_block

        # Sampleo (x, t) en el dominio interior (0,1768)x(0,1000) normalizadas al (0,1)x(0,1)
        x_interior = torch.rand(N_interior, 1, device=device) #* Nx
        t_interior = t_start + (t_end - t_start) * torch.rand(N_interior, 1, device=device) #* T_max

        x_interior.requires_grad, t_interior.requires_grad = True, True

        # Genero un tensor para beta y gamma con valor constante en todo el mapa
        beta_sampled = torch.full((N_interior,1), beta_val, device=device)
        gamma_sampled = torch.full((N_interior,1), gamma_val, device=device)

        # Sampleo condiciones de borde
        x_left = torch.zeros(N_boundary, 1, device=device) # x=0
        x_right = torch.ones(N_boundary, 1, device=device) # x=1768
        t_boundary = t_start + (t_end - t_start) * torch.rand(N_boundary, 1, device=device)

        for epoch in range(epochs_per_block):
            # Calculamos la perdida de la PDE
            SIR_pred = model(x_interior, t_interior)
            S_pred, I_pred = SIR_pred[:, 0:1], SIR_pred[:, 1:2]

            dS_dt = torch.autograd.grad(S_pred, t_interior, torch.ones_like(S_pred).to(device), create_graph=True)[0]
            dI_dt = torch.autograd.grad(I_pred, t_interior, torch.ones_like(I_pred).to(device), create_graph=True)[0]
            dI_dx = torch.autograd.grad(I_pred, x_interior, torch.ones_like(I_pred).to(device), create_graph=True)[0]
            d2I_dx2 = torch.autograd.grad(dI_dx, x_interior, torch.ones_like(dI_dx).to(device), create_graph=True)[0]

            # Definir pérdidas basadas en las ecuaciones diferenciales
            loss_S = dS_dt + beta_sampled * S_pred * I_pred
            loss_I = dI_dt - (beta_sampled * S_pred * I_pred - gamma_sampled * I_pred) - D_I * d2I_dx2
            loss_pde = loss_S.pow(2).mean() + loss_I.pow(2).mean()

            # Condición inicial en t_start
            #if block == 0:
            t_init = torch.full_like(x_init, t_start)
            SIR_init_pred = model(x_init, t_init)
            S_init_pred, I_init_pred = SIR_init_pred[:, 0:1], SIR_init_pred[:, 1:2]
            loss_ic = (S_init_pred - S_init).pow(2).mean() + (I_init_pred - I_init).pow(2).mean()
            #else:
            #    loss_ic = torch.tensor(0.0, device=device)

            # Calcular salida de la red para los bordes
            S_left_pred, I_left_pred = model(x_left, t_boundary)[:,0:1], model(x_left, t_boundary)[:,1:2] # Borde de la izquierda (x,t)=(0,t)
            S_right_pred, I_right_pred = model(x_right, t_boundary)[:,0:1], model(x_right, t_boundary)[:,1:2] # Borde de la derecha (x,t)=(1768,t)
            loss_bc = (S_left_pred**2).mean() + (I_left_pred**2).mean() + (S_right_pred**2).mean() + (I_right_pred**2).mean()

            # Pérdida total
            loss = loss_pde + 10*loss_ic + 0.1*loss_bc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Bloque {block} terminado")
        print(f"Loss final en el bloque {block}: {loss.item()}")

        # Actualizar condición inicial: evalúa en t = t_end
        t_next = torch.full_like(x_init, t_end)
        with torch.no_grad():
            SIR_next = model(x_init, t_next)
            S_init = SIR_next[:, 0:1].detach()
            I_init = SIR_next[:, 1:2].detach()

    return model