import numpy as np

def solve_radiative_transfer(tau, ssa, g, n_layers):
    # Initialize variables
    layer_thickness = 1.0   # Thickness of each atmospheric layer
    solar_flux = 1.0        # Incident solar flux

    # Initialize arrays to store results
    upward_flux = np.zeros(n_layers)
    downward_flux = np.zeros(n_layers)
    
    # Loop through atmospheric layers
    for i in range(n_layers):
        # Calculate extinction coefficient
        extinction = tau[i] / layer_thickness

        # Calculate single scattering albedo
        omega = ssa[i]

        # Calculate asymmetry parameter
        asymmetry = g[i]

        # Calculate optical properties
        absorption = extinction * (1 - omega)
        scattering = extinction - absorption

        # Calculate source term
        source_term = solar_flux * extinction

        # Calculate fluxes
        if i == 0:
            downward_flux[i] = source_term / (extinction + scattering)
        else:
            downward_flux[i] = (source_term + scattering * upward_flux[i-1]) / (extinction + scattering)
        
        upward_flux[i] = downward_flux[i] * omega + asymmetry * scattering * upward_flux[i-1]
    
    return upward_flux, downward_flux

# Example usage
tau = np.array([0.2, 0.3, 0.4])   # Aerosol optical depth in each layer
ssa = np.array([0.9, 0.85, 0.8])  # Single scattering albedo in each layer
g = np.array([0.7, 0.75, 0.8])    # Asymmetry parameter in each layer
n_layers = len(tau)

up_flux, down_flux = solve_radiative_transfer(tau, ssa, g, n_layers)

print("Upward Flux:", up_flux)
print("Downward Flux:", down_flux)
