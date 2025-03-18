import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import cupy as cp # type: ignore

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
def train_pinn(beta, gamma, D_I, A, wx, h_dx, B, wy, h_dy, S0, I0, R0, epochs=10000):
    model = FireSpread_PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dt = 1/6  # Paso temporal en horas (10 minutos)
    num_T = 1000  # Cantidad de pasos de tiempo
    T_max = num_T * dt  # Tiempo total simulado en horas (~6 días)

    t = torch.linspace(0, T_max, num_T, device=device).view(-1, 1)  # Discretización del tiempo
    x = torch.linspace(0, 1768, 1768, device=device).view(-1, 1)  # Eje X con 1768 puntos
    y = torch.linspace(0, 1060, 1060, device=device).view(-1, 1)  # Eje Y con 1060 puntos
    
    T, X, Y = torch.meshgrid(t.squeeze(), x.squeeze(), y.squeeze(), indexing='ij')
    TXY = torch.cat([T.reshape(-1,1), X.reshape(-1,1), Y.reshape(-1,1)], dim=1)
    TXY.requires_grad = True

    # Condiciones iniciales (t = 0)
    T0, X0, Y0 = torch.meshgrid(torch.tensor([0.0], device="cuda"), x.squeeze(), y.squeeze(), indexing='ij')
    TXY0 = torch.cat([T0.reshape(-1,1), X0.reshape(-1,1), Y0.reshape(-1,1)], dim=1)

    S0_flat = S0.reshape(-1, 1)
    I0_flat = I0.reshape(-1, 1)
    R0_flat = R0.reshape(-1, 1)

    for epoch in range(epochs):
        SIR = model(TXY)
        S, I, R = SIR[:, 0:1], SIR[:, 1:2], SIR[:, 2:3]

        # Cálculo de derivadas
        dS_dt = torch.autograd.grad(S, TXY, torch.ones_like(S).to(device), create_graph=True)[0][:, 0:1]
        dI_dt = torch.autograd.grad(I, TXY, torch.ones_like(I).to(device), create_graph=True)[0][:, 0:1]
        dR_dt = torch.autograd.grad(R, TXY, torch.ones_like(R).to(device), create_graph=True)[0][:, 0:1]

        dI_dx = torch.autograd.grad(I, TXY, torch.ones_like(I).to(device), create_graph=True)[0][:, 1:2]
        dI_dy = torch.autograd.grad(I, TXY, torch.ones_like(I).to(device), create_graph=True)[0][:, 2:3]

        d2I_dx2 = torch.autograd.grad(dI_dx, TXY, torch.ones_like(dI_dx).to(device), create_graph=True)[0][:, 1:2]
        d2I_dy2 = torch.autograd.grad(dI_dy, TXY, torch.ones_like(dI_dy).to(device), create_graph=True)[0][:, 2:3]

        # Definir pérdidas basadas en las ecuaciones diferenciales
        loss_S = dS_dt + beta * S * I 
        loss_I = dI_dt - (beta * S * I - gamma * I) - D_I * (d2I_dx2 + d2I_dy2) + (A * wx + B * h_dx) * dI_dx + (A * wy + B * h_dy) * dI_dy
        loss_R = dR_dt - gamma * I

        loss_pde = loss_S.pow(2).mean() + loss_I.pow(2).mean() + loss_R.pow(2).mean()

        # Calcular salida de la red en t=0 para condición inicial
        SIR0 = model(TXY0)
        S0_pred, I0_pred, R0_pred = SIR0[:, 0:1], SIR0[:, 1:2], SIR0[:, 2:3]

        # Pérdida por condición inicial (forzar que la red prediga bien los valores iniciales)
        loss_ic = (S0_pred - S0_flat).pow(2).mean() + (I0_pred - I0_flat).pow(2).mean() + (R0_pred - R0_flat).pow(2).mean()

        # Pérdida total
        loss = loss_pde + loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model