#include <iostream>
#include <vector>
#include <cmath>

// Constants
const double g = 9.81; // Acceleration due to gravity (m/s^2)
const double dt = 0.01; // Time step (s)

// Quaternion class
class Quaternion {
public:
    double w, x, y, z;

    Quaternion(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}

    Quaternion operator*(const Quaternion &other) const {
        return Quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        );
    }

    Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }

    Quaternion rotate(const Quaternion &v) const {
        Quaternion qv(0, v.x, v.y, v.z);
        Quaternion result = (*this) * qv * conjugate();
        return Quaternion(result.w, result.x, result.y, result.z);
    }
};

// Function to calculate the derivative of state variables
std::pair<double, double> derivatives(double pos, double vel, Quaternion orientation) {
    Quaternion thrust(0, 0, 0, 10.0); // Thrust of the rocket (N) in the z-direction
    Quaternion F = orientation.rotate(thrust); // Transform force to global frame

    double acc = F.z / 1.0 - g; // Acceleration including gravity

    return std::make_pair(vel, acc);
}

// Runge-Kutta 4th Order integration
std::pair<double, double> rk4_step(double pos, double vel, Quaternion orientation, double t) {
    auto k1 = derivatives(pos, vel, orientation);
    auto k2 = derivatives(pos + k1.first * dt/2, vel + k1.second * dt/2, orientation);
    auto k3 = derivatives(pos + k2.first * dt/2, vel + k2.second * dt/2, orientation);
    auto k4 = derivatives(pos + k3.first * dt, vel + k3.second * dt, orientation);

    pos += (k1.first + 2*k2.first + 2*k3.first + k4.first) * dt / 6;
    vel += (k1.second + 2*k2.second + 2*k3.second + k4.second) * dt / 6;

    return std::make_pair(pos, vel);
}

int main() {
    // Initial conditions
    double initial_velocity = 0.0; // Initial velocity (m/s)
    Quaternion initial_orientation(1, 0, 0, 0); // Initial orientation (no rotation)

    // Initial position
    double pos = 0.0;

    // Initial velocity
    double vel = initial_velocity;

    // Initial orientation
    Quaternion orientation = initial_orientation;

    // Simulation loop
    std::vector<double> trajectory;
    double t = 0.0;
    while (pos >= 0.0) {
        auto result = rk4_step(pos, vel, orientation, t);
        pos = result.first;
        vel = result.second;
        t += dt;
        trajectory.push_back(pos);
    }

    // Output the trajectory
    for (double p : trajectory) {
        std::cout << p << std::endl;
    }

    return 0;
}
