import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## DEFINICIÓN DE LA PINN ###############################################

# Definir la red neuronal PINN
class FireSpread_PINN(nn.Module):
    def __init__(self):
        super(FireSpread_PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
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

    def forward(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        return self.net(inputs)

############################## FUNCIÓN DE ENTRENAMIENTO ###############################################

dominio_total = [0, 10.0]
subdominios = [(i, i + 1.0) for i in range(10)]
# Función para entrenar la PINN con ecuaciones de propagación de fuego
def train_pinn(D_I, beta_val, gamma_val, epochs_adam=1000, epochs_lbfgs=20):
    pinns = [FireSpread_PINN().to(device) for _ in subdominios]

    # Configuración del dominio
    N_interior, N_boundary, N_initial = 20000, 2000, 4000
    N_interface = 1000
    t_interface = torch.rand(N_interface, 1, device=device)

    x_ignition = torch.tensor([[4.5]], device=device)

    for i, ((x_min, x_max), model) in enumerate(zip(subdominios, pinns)):
        x_interior = torch.rand(N_interior, 1, device=device)
        t_interior = torch.rand(N_interior, 1, device=device)
        x_interior.requires_grad, t_interior.requires_grad = True, True

        if x_min < x_ignition.item() and x_max > x_ignition.item():
            x_init = torch.rand(N_initial - 1, 1, device=device)
            x_ignition /= 10
            x_init = torch.cat([x_init, x_ignition], dim=0)
            
            sigma_x = 0.05
            I_init = torch.exp(-0.5 * ((x_init - x_ignition) / sigma_x) ** 2)
            S_init = 1 - I_init
        else:
            x_init = torch.rand(N_initial, 1, device=device)
            I_init = torch.zeros_like(x_init)
            S_init = torch.ones_like(x_init)

        t_init = torch.zeros(N_initial, 1, device=device)
        R_init = torch.zeros_like(I_init)

        x_left = torch.zeros(N_boundary, 1, device=device)
        x_right = torch.ones(N_boundary, 1, device=device)
        t_boundary = torch.rand(N_boundary, 1, device=device)

        beta_sampled = torch.full((N_interior, 1), beta_val, device=device)
        gamma_sampled = torch.full((N_interior, 1), gamma_val, device=device)

        # Modelos vecinos
        left_model = pinns[i - 1] if i > 0 else None
        right_model = pinns[i + 1] if i < len(subdominios) - 1 else None

        # Interfaces
        x_interface_left = torch.zeros(N_interface, 1, device=device) if i > 0 else None
        x_interface_right = torch.ones(N_interface, 1, device=device) if i < len(subdominios) - 1 else None

        # --------- Cierre de optimización ---------
        def make_closure():
            def closure():
                model.train()
                SIR_pred = model(x_interior, t_interior)
                S_pred, I_pred, R_pred = SIR_pred[:, 0:1], SIR_pred[:, 1:2], SIR_pred[:, 2:3]

                dS_dt = torch.autograd.grad(S_pred, t_interior, torch.ones_like(S_pred), create_graph=True)[0]
                dI_dt = torch.autograd.grad(I_pred, t_interior, torch.ones_like(I_pred), create_graph=True)[0]
                dR_dt = torch.autograd.grad(R_pred, t_interior, torch.ones_like(R_pred), create_graph=True)[0]

                dI_dx = torch.autograd.grad(I_pred, x_interior, torch.ones_like(I_pred), create_graph=True)[0]
                d2I_dx2 = torch.autograd.grad(dI_dx, x_interior, torch.ones_like(dI_dx), create_graph=True)[0]

                loss_S = (dS_dt + beta_sampled).pow(2).mean()
                loss_I = (dI_dt - beta_sampled*S_pred*I_pred + gamma_sampled*I_pred - D_I*d2I_dx2).pow(2).mean()
                loss_R = (dR_dt - gamma_sampled*I_pred).pow(2).mean()

                loss_pde = loss_S + loss_I + loss_R

                # Condición inicial
                SIR_init_pred = model(x_init, t_init)
                S_init_pred, I_init_pred, R_init_pred = SIR_init_pred[:, 0:1], SIR_init_pred[:, 1:2], SIR_init_pred[:, 2:3]
                loss_ic = (S_init_pred - S_init).pow(2).mean() + (I_init_pred - I_init).pow(2).mean() + (R_init_pred - R_init).pow(2).mean()

                # Acomplamiento con subdominios vecinos
                loss_interface = 0.0
                if left_model is not None and x_interface_left is not None:
                    pred_left = left_model(x_interface_left, t_interface)
                    S_left, I_left, R_left = pred_left[:, 0:1], pred_left[:, 1:2], pred_left[:, 2:3]
                    pred_curr = model(x_interface_left, t_interface)
                    S_curr, I_curr, R_curr = pred_curr[:, 0:1], pred_curr[:, 1:2], pred_curr[:, 2:3]
                    loss_interface += (S_left - S_curr).pow(2).mean() + (I_left - I_curr).pow(2).mean() + (R_left - R_curr).pow(2).mean()

                if right_model is not None and x_interface_right is not None:
                    pred_right = right_model(x_interface_right, t_interface)
                    S_right, I_right, R_right = pred_right[:, 0:1], pred_right[:, 1:2], pred_right[:, 2:3]
                    pred_curr = model(x_interface_right, t_interface)
                    S_curr, I_curr, R_curr = pred_curr[:, 0:1], pred_curr[:, 1:2], pred_curr[:, 2:3]
                    loss_interface += (S_right - S_curr).pow(2).mean() + (I_right - I_curr).pow(2).mean() + (R_right - R_curr).pow(2).mean()

                loss_phys = (S_pred + I_pred + R_pred - 1).pow(2).mean()

                loss = loss_pde + loss_ic + loss_interface + loss_phys
                loss.backward()
                return loss
            
            return closure
        
        closure = make_closure()
                                                                                                                      
        # --------- Primera etapa: Adam ---------
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(epochs_adam):
            optimizer.zero_grad()
            loss = closure()
            optimizer.step()
            if epoch % 200 == 0 or epoch == epochs_adam - 1:
                print(f"[Adam] Subdominio {i} | Época {epoch} | Loss: {loss.item():.6f}")

        # --------- Optimización: LBFGS ---------
        # optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=epochs_lbfgs, history_size=50)
        # optimizer_lbfgs.step(closure)

    return pinns