import torch # type: ignore
from train_pinn import train_pinn, domain_size
import sys

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## PARÁMETROS DE LOS MAPAS ###############################################

# Parámetros fijos
gamma_val = 0.3
domain_size = 2
mean_x, mean_y = domain_size / 2, domain_size / 2
sigma_x, sigma_y = 0.05, 0.05
epochs_adam = 10000

# Parámetros variables (desde línea de comandos)
# D_I = float(sys.argv[1])
# beta_val = float(sys.argv[2])
beta_val = 1.0
D_I = 0.05

############################## ENTRENAMIENTO DEL MODELO ###############################################

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
model, best_loss = train_pinn(D_I, beta_val, gamma_val, mean_x, mean_y, sigma_x, sigma_y, epochs_adam=epochs_adam)
end_time.record()

# Sincronizar para asegurar medición correcta
torch.cuda.synchronize()
training_time = start_time.elapsed_time(end_time) / 1000  # Convertir de ms a s

print(f"Mejor modelo guardado. Best Loss: {best_loss}")
print(f"Training time: {training_time:.2f} seconds")

############################## GUARDADO DEL MODELO ENTRENADO ###############################################

torch.save(model.state_dict(), f"adaptive_pinns_DI{D_I}_beta{beta_val}.pth")


