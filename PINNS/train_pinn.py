import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import cupy as cp # type: ignore

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
def train_pinn(beta, gamma, D_I, A, wx, h_dx, B, wy, h_dy, epochs=10000):
    model = FireSpread_PINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dt = 1/6  # Paso temporal en horas (10 minutos)
    num_T = 1000  # Cantidad de pasos de tiempo
    T_max = num_T * dt  # Tiempo total simulado en horas (~6 días)

    t = torch.linspace(0, T_max, num_T).view(-1, 1)  # Discretización del tiempo
    x = torch.linspace(0, 1768, 1768).view(-1, 1)  # Eje X con 1768 puntos
    y = torch.linspace(0, 1060, 1060).view(-1, 1)  # Eje Y con 1060 puntos
    
    T, X, Y = torch.meshgrid(t.squeeze(), x.squeeze(), y.squeeze(), indexing='ij')
    TXY = torch.cat([T.reshape(-1,1), X.reshape(-1,1), Y.reshape(-1,1)], dim=1)
    TXY.requires_grad = True

    for epoch in range(epochs):
        SIR = model(TXY)
        S, I, R = SIR[:, 0:1], SIR[:, 1:2], SIR[:, 2:3]

        # Cálculo de derivadas
        dS_dt = torch.autograd.grad(S, TXY, torch.ones_like(S), create_graph=True)[0][:, 0:1]
        dI_dt = torch.autograd.grad(I, TXY, torch.ones_like(I), create_graph=True)[0][:, 0:1]
        dR_dt = torch.autograd.grad(R, TXY, torch.ones_like(R), create_graph=True)[0][:, 0:1]

        dI_dx = torch.autograd.grad(I, TXY, torch.ones_like(I), create_graph=True)[0][:, 1:2]
        dI_dy = torch.autograd.grad(I, TXY, torch.ones_like(I), create_graph=True)[0][:, 2:3]

        d2I_dx2 = torch.autograd.grad(dI_dx, TXY, torch.ones_like(dI_dx), create_graph=True)[0][:, 1:2]
        d2I_dy2 = torch.autograd.grad(dI_dy, TXY, torch.ones_like(dI_dy), create_graph=True)[0][:, 2:3]

        # Definir pérdidas basadas en las ecuaciones diferenciales
        loss_S = dS_dt + beta * S * I 
        loss_I = dI_dt - (beta * S * I - gamma * I) - D_I * (d2I_dx2 + d2I_dy2) + (A * wx + B * h_dx) * dI_dx + (A * wy + B * h_dy) * dI_dy
        loss_R = dR_dt - gamma * I

        loss = loss_S.pow(2).mean() + loss_I.pow(2).mean() + loss_R.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model