import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
dt = 0.01  # Time step (s)

# Quaternion class implementation
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def rotate(self, vector):
        q_vec = Quaternion(0, *vector)
        q_conj = self.conjugate()
        rotated_vec = (self * q_vec * q_conj).xyz()
        return rotated_vec

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def xyz(self):
        return np.array([self.x, self.y, self.z])

# Initial conditions
m = 1.0  # Mass of the rocket (kg)
thrust = np.array([0, 0, 10.0])  # Thrust of the rocket (N) in the z-direction
initial_orientation = Quaternion(1, 0, 0, 0)  # Initial orientation (no rotation)
initial_velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (m/s)

# Function to calculate the derivative of state variables
def derivatives(pos, vel, orientation, t):
    F = Quaternion(0, *thrust)  # Force vector in body frame
    F = orientation.rotate(F.xyz())  # Transform force to global frame
    
    acc = F / m - np.array([0, 0, g])  # Acceleration including gravity
    
    dposdt = vel
    dveldt = acc
    return dposdt, dveldt

# Runge-Kutta 4th Order integration
def rk4_step(pos, vel, orientation, t, dt):
    k1pos, k1vel = derivatives(pos, vel, orientation, t)
    k2pos, k2vel = derivatives(pos + k1pos * dt/2, vel + k1vel * dt/2, orientation, t + dt/2)
    k3pos, k3vel = derivatives(pos + k2pos * dt/2, vel + k2vel * dt/2, orientation, t + dt/2)
    k4pos, k4vel = derivatives(pos + k3pos * dt, vel + k3vel * dt, orientation, t + dt)
    
    pos += (k1pos + 2*k2pos + 2*k3pos + k4pos) * dt / 6
    vel += (k1vel + 2*k2vel + 2*k3vel + k4vel) * dt / 6
    
    return pos, vel

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
    pos, vel = rk4_step(pos, vel, orientation, t, dt)
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







