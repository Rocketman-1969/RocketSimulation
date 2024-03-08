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
initial_angular_velocity = np.array([0.0, 0.0, 0.0])  # Initial angular velocity (rad/s)

# Function to calculate the derivative of state variables
def derivatives(pos, vel, orientation, ang_vel, t):
    F = thrust  # Force vector in body frame
    acc = np.array([0, 0, thrust[2] / m - g])  # Acceleration including gravity
    
    dposdt = vel
    dveldt = acc
    
    # Moment of force
    moment = np.cross(np.array([0, 0, 1]), F)
    # Moment of inertia tensor (assuming diagonal for simplicity)
    I = np.diag([1, 1, 1])
    # Angular acceleration
    davel = np.linalg.inv(I) @ moment
    
    return dposdt, dveldt, davel

# Runge-Kutta 4th Order integration
def rk4_step(pos, vel, orientation, ang_vel, t, dt):
    k1pos, k1vel, k1avel = derivatives(pos, vel, orientation, ang_vel, t)
    k1_orientation = orientation

    k2pos, k2vel, k2avel = derivatives(pos + k1pos * dt/2, vel + k1vel * dt/2, update_orientation(k1_orientation, ang_vel, dt/2), ang_vel + k1avel * dt/2, t + dt/2)
    k2_orientation = update_orientation(k1_orientation, ang_vel + k1avel * dt/2, dt/2)

    k3pos, k3vel, k3avel = derivatives(pos + k2pos * dt/2, vel + k2vel * dt/2, update_orientation(k2_orientation, ang_vel, dt/2), ang_vel + k2avel * dt/2, t + dt/2)
    k3_orientation = update_orientation(k2_orientation, ang_vel + k2avel * dt/2, dt/2)

    k4pos, k4vel, k4avel = derivatives(pos + k3pos * dt, vel + k3vel * dt, update_orientation(k3_orientation, ang_vel, dt), ang_vel + k3avel * dt, t + dt)
    k4_orientation = update_orientation(k3_orientation, ang_vel + k3avel * dt, dt)

    pos += (k1pos + 2*k2pos + 2*k3pos + k4pos) * dt / 6
    vel += (k1vel + 2*k2vel + 2*k3vel + k4vel) * dt / 6
    ang_vel += (k1avel + 2*k2avel + 2*k3avel + k4avel) * dt / 6
    
    return pos, vel, update_orientation((k1_orientation + 2*k2_orientation + 2*k3_orientation + k4_orientation) / 6, ang_vel, dt), ang_vel

# Function to update orientation using quaternion
def update_orientation(orientation, ang_vel, dt):
    omega = 0.5 * ang_vel
    delta_q = np.array([0, *omega]) * orientation
    orientation += delta_q * dt
    orientation /= np.linalg.norm(orientation)
    return orientation

# Initial position
pos = np.array([0.0, 0.0, 0.0])

# Initial velocities
vel = initial_velocity
ang_vel = initial_angular_velocity

# Initial orientation
orientation = initial_orientation

# Simulation loop
trajectory = [pos]
t = 0.0
while pos[2] >= 0.0:
    pos, vel, orientation, ang_vel = rk4_step(pos, vel, orientation, ang_vel, t, dt)
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
