import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import os
import json

# Constants
focal = 6  # mm
zc = 234 - 8
h = 9
delta = 19
l = delta + h
thou_pixels_eq_mm = 4.8  # 1000 pixels are equal to this number of mm

# Projection function
def project(point, focal):
    """Project a point onto the camera plane."""
    return (point / point[..., 2, np.newaxis]) * focal

# Initial guess function
def initial_guess(proj_apex, proj_cm, focal, zc, l, h):
    """Find approximate solution for nutation and precession."""
    A_ = proj_apex.T / focal
    CM_ = proj_cm.T / focal
    X = (A_[0] - CM_[0]) * zc - (A_[0] * l - CM_[0] * h)
    Y = (A_[1] - CM_[1]) * zc - (A_[1] * l - CM_[0] * h)
    nu = 1 / (l - h) * np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan(Y / X) + np.pi * (X < 0)
    return np.vstack((nu, phi))

# Non-linear solver for Euler angles
def solve_euler_angles(proj_apex, proj_cm, focal, zc, l, h):
    initial_angles = initial_guess(proj_apex, proj_cm, focal, zc, l, h)

    def residual(x, A_, CM_):
        nu, phi = x
        Rx = (A_[0] * l - CM_[0] * h) * np.cos(nu) + delta * np.sin(nu) * np.cos(phi) - (A_[0] - CM_[0]) * zc
        Ry = (A_[1] * l - CM_[1] * h) * np.cos(nu) + delta * np.sin(nu) * np.sin(phi) - (A_[1] - CM_[1]) * zc
        return np.array([Rx, Ry])

    solutions = np.zeros_like(initial_angles)
    for i in range(initial_angles.shape[1]):
        A_ = proj_apex[i] / focal
        CM_ = proj_cm[i] / focal
        res = root(residual, initial_angles[:, i], args=(A_, CM_))
        solutions[:, i] = res.x
    return solutions

# Load coordinates from a JSON file
def load_apex_coordinates(file_path):
    with open(file_path, 'r') as f:
        apex_coords_matrix = json.load(f)
    return apex_coords_matrix

# Save angles to a JSON file
def save_angles_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    base_folder = 'C:/Users/maxdi/OneDrive/Desktop/Bachelorthesis/Pycharm/ST2/Photos'
    sub_folder = '13-12-24/8'
    input_folder = os.path.join(base_folder, sub_folder)

    # Load the com coordinates from JSON
    file_path = os.path.join(input_folder, 'p1_com_coords.json')
    com_coords_matrix = load_apex_coordinates(file_path)

    # Load the apex coordinates from JSON
    file_path = os.path.join(input_folder, 'p1_apex_coords.json')
    apex_coords_matrix = load_apex_coordinates(file_path)

    # Use only the first coordinate for a single spinning top
    proj_cm = np.array(com_coords_matrix)  # Convert to numpy array
    proj_apex = np.array(apex_coords_matrix)  # Convert to numpy array

    # Modify coordinates
    proj_cm_modified = (proj_cm - [400, 300]) * thou_pixels_eq_mm / 1000
    proj_apex_modified = (proj_apex - [400, 300]) * thou_pixels_eq_mm / 1000

    # Solve for Euler angles
    final_solution = np.rad2deg(solve_euler_angles(proj_apex_modified, proj_cm_modified, focal, zc, l, h))
    print("Final solution (in degrees) from solver:")
    print(final_solution)

    # Save `nu` and `phi` from the final solution to JSON files
    nu_final = final_solution[0].tolist()  # Extract `nu` values and convert to list
    phi_final = final_solution[1].tolist()  # Extract `phi` values and convert to list

    nu_filename_final = os.path.join(input_folder, 'final_nu_angles.json')
    phi_filename_final = os.path.join(input_folder, 'final_phi_angles.json')

    save_angles_to_json(nu_final, nu_filename_final)  # Save `nu` final solution
    save_angles_to_json(phi_final, phi_filename_final)  # Save `phi` final solution

    print("Saving angles in degrees...")
    save_angles_to_json(nu_final, nu_filename_final)
    save_angles_to_json(phi_final, phi_filename_final)

    # Plotting
    vecs = proj_apex_modified.T - proj_cm_modified.T
    plt.scatter(*proj_cm_modified.T, c='k', label='Modified CM')
    plt.scatter(*proj_apex_modified.T, c='r', label='Modified A')
    plt.quiver(*proj_cm_modified.T, *vecs, scale=1, scale_units='xy', angles='xy')
    plt.grid()
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
