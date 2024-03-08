import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
dt = 0.01  # Time step (s)

# Initial conditions
m = 1.0  # Mass of the rocket (kg)
thrust = np.array([0, 0, 10.0])  # Thrust of the rocket (N) in the z-direction
initial_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation (no rotation)
initial_velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (m/s)

# Function to calculate the derivative of state variables
def derivatives(pos, vel, orientation, t):
    F = thrust  # Force vector in body frame
    acc = np.array([0, 0, thrust[2] / m - g])  # Acceleration including gravity
    
    dposdt = vel
    dveldt = acc
    return dposdt, dveldt

# Forward differencing for orientation
def update_orientation(orientation, dt):
    # Angular velocity in body frame (assumed zero for simplicity)
    omega = np.array([0, 0, 0])

    # Update quaternion using forward differencing
    dqdt = 0.5 * np.array([0.0, *omega]) * orientation
    orientation += dqdt * dt
    orientation /= np.linalg.norm(orientation)  # Normalize quaternion
    return orientation

# Runge-Kutta 4th Order integration
def rk4_step(pos, vel, orientation, t, dt):
    k1pos, k1vel = derivatives(pos, vel, orientation, t)
    k1_orientation = orientation

    k2pos, k2vel = derivatives(pos + k1pos * dt/2, vel + k1vel * dt/2, update_orientation(k1_orientation, dt/2), t + dt/2)
    k2_orientation = update_orientation(k1_orientation, dt/2)

    k3pos, k3vel = derivatives(pos + k2pos * dt/2, vel + k2vel * dt/2, update_orientation(k2_orientation, dt/2), t + dt/2)
    k3_orientation = update_orientation(k2_orientation, dt/2)

    k4pos, k4vel = derivatives(pos + k3pos * dt, vel + k3vel * dt, update_orientation(k3_orientation, dt), t + dt)
    k4_orientation = update_orientation(k3_orientation, dt)

    pos += (k1pos + 2*k2pos + 2*k3pos + k4pos) * dt / 6
    vel += (k1vel + 2*k2vel + 2*k3vel + k4vel) * dt / 6
    
    return pos, vel, (k1_orientation + 2*k2_orientation + 2*k3_orientation + k4_orientation) / 6

# Initial position
pos = np.array([0.0, 0.0, 0.0])

# Initial velocities
vel = initial_velocity

# Initial orientation
orientation = initial_orientation

# Simulation loop
trajectory = [pos]
t = 0.0
while pos[2] >= 0.0:
    pos, vel, orientation = rk4_step(pos, vel, orientation, t, dt)
    t += dt
    trajectory.append(pos)

# Plot the trajectory
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.title("Rocket Trajectory (Straight Up)")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Altitude (m)")
plt.grid(True)
plt.show()