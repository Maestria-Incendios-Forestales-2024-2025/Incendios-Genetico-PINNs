import numpy as np
import cupy as cp # type: ignore
import torch # type: ignore
from train_pinn_1d import train_pinn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## PARÁMETROS DE LOS MAPAS ###############################################
d = 30
Nx = 1768
# Coeficiente de difusión
D_I = 1 

############################## ENTRENAMIENTO DEL MODELO ###############################################

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
model = train_pinn(D_I, epochs_per_block=1000)
end_time.record()

# Sincronizar para asegurar medición correcta
torch.cuda.synchronize()
training_time = start_time.elapsed_time(end_time) / 1000  # Convertir de ms a s

print(f"Training time: {training_time:.2f} seconds")

############################## GUARDADO DEL MODELO ENTRENADO ###############################################

torch.save(model.state_dict(), "modelo_entrenado.pth")