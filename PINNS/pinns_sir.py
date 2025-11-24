import torch # type: ignore
from train_pinn import train_pinn, domain_size
import sys
import numpy as np
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## PARÁMETROS DE LOS MAPAS ###############################################

# Parámetros fijos
gamma_val = 0.3
domain_size = 1
mean_x, mean_y = domain_size / 2, domain_size / 2
sigma_x, sigma_y = 0.05, 0.05
epochs_adam = 50000

# Parámetros variables (desde línea de comandos)
# D_I = float(sys.argv[1])
# beta_val = float(sys.argv[2])
beta_val = 1.0
# D_I = -10.0 # valor inicial para el problema inverso
D_I = 5e-4
############################## CARGADO DE LOS DATOS PARA EL PROBLEMA INVERSO ###############################################

# Función para cargar S, I, R en varios tiempos
def load_SIR_data(time_points, data_dir='_numpy_D0.0005_beta1.0_gamma0.3_t'):
    S_data_list, I_data_list, R_data_list, t_data_list = [], [], [], []
    
    for t in time_points:
        S = np.load(f"S{data_dir}{t}.npy")
        I = np.load(f"I{data_dir}{t}.npy")
        R = np.load(f"R{data_dir}{t}.npy")
        
        # Convertir a tensores
        S = torch.tensor(S, dtype=torch.float32, device=device)
        I = torch.tensor(I, dtype=torch.float32, device=device)
        R = torch.tensor(R, dtype=torch.float32, device=device)
        
        S_data_list.append(S)
        I_data_list.append(I)
        R_data_list.append(R)

    return S_data_list, I_data_list, R_data_list

# --- Cargar datos para los tiempos que te interesen ---
time_points = [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# S_data_list, I_data_list, R_data_list = load_SIR_data(time_points)

############################## ENTRENAMIENTO DEL MODELO ###############################################

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
# model, optimizer, best_loss, last_epoch, best_D_I = train_pinn(
#     modo = 'inverse',
#     beta_val = beta_val,
#     gamma_val = gamma_val,
#     D_I = D_I,
#     mean_x=mean_x, mean_y=mean_y,
#     sigma_x=sigma_x, sigma_y=sigma_y,
#     epochs_adam=epochs_adam,
#     t_data=time_points,
#     S_data=S_data_list,
#     I_data=I_data_list,
#     R_data=R_data_list
# )
model, optimizer, best_loss, last_epoch, best_D_I = train_pinn(
    modo = 'forward',
    beta_val = beta_val,
    gamma_val = gamma_val,
    D_I = D_I,
    mean_x=mean_x, mean_y=mean_y,
    sigma_x=sigma_x, sigma_y=sigma_y,
    epochs_adam=epochs_adam,
    t_data=None,
    S_data=None,
    I_data=None,
    R_data=None
)
end_time.record()

# Sincronizar para asegurar medición correcta
torch.cuda.synchronize()
training_time = start_time.elapsed_time(end_time) / 1000  # Convertir de ms a s

print(f"Mejor modelo guardado. Best Loss: {best_loss}")
print(f"Training time: {training_time:.2f} seconds")

job_id = os.getenv('SLURM_JOB_ID', 'local')
print(f"Job ID: {job_id}")

############################## GUARDADO DEL MODELO ENTRENADO ###############################################

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_loss': best_loss,
    'epoch': last_epoch,
    'D_I': best_D_I
}, f"adaptive_pinns_DI{D_I}_beta{beta_val}_job{job_id}.pth")

print("Mejor D_I encontrado:", best_D_I)

