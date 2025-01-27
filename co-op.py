import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
M1 = 100     # Mass of the first cart (robot base) (kg)
M2 = 10000   # Mass of the second cart (ship) (kg)
m = 10       # Mass of the pendulum (robot arm) (kg)
l = 1        # Length of the pendulum (m)
g = 9.81     # Gravitational acceleration (m/s^2)

# Desired states
theta_desired = 0    # Desired pendulum angle (upright)
X1_desired = 0       # Desired position of Cart 1

# PD controller gains
Kp_theta = 100       # Proportional gain for pendulum angle
Kd_theta = 10        # Derivative gain for pendulum angle
Kp_X1 = 100          # Proportional gain for Cart 1 position
Kd_X1 = 10           # Derivative gain for Cart 1 velocity

# Initial conditions: [X1, X2, theta, v1, v2, omega]
x0 = [5, 0, np.pi/4, 0, 0, 0]  # Initial conditions

# Time span for simulation
tspan = (0, 5)  # Simulate from t = 0 to 5 seconds

# Ship dimensions
L_ship = 100    # Length of the ship (meters)
W_ship = 20     # Width of the ship (meters)

# Noise standard deviation for ship velocity
sigma_v2 = 1005


# Define the nonlinear dynamics with bounds
def pendulum_cart_cart_nonlinear_dynamics_with_bounds(t, x):
    # Extract state variables
    X1, X2, theta, v1, _, omega = x

    # Noisy velocity of Cart 2 (ship)
    v2 = 2 * np.sin(0.1 * t) + sigma_v2 * np.random.randn()

    # PD control laws
    tau_pendulum = Kp_theta * (theta_desired - theta) + Kd_theta * (0 - omega)
    F1 = Kp_X1 * (X1_desired - X1) + Kd_X1 * (0 - v1)

    # Mass matrix M(q)
    M = np.array([
        [M1 + m, m, -m * l * np.cos(theta)],
        [m, M2 + m, -m * l * np.cos(theta)],
        [-m * l * np.cos(theta), -m * l * np.cos(theta), m * l**2]
    ])

    # Coriolis and centrifugal terms C(q, q_dot)
    C = np.array([
        -m * l * omega**2 * np.sin(theta),
        -m * l * omega**2 * np.sin(theta),
        2 * m * l * (v1 + v2) * omega * np.sin(theta)
    ])

    # Gravitational forces G(q)
    G = np.array([0, 0, m * g * l * np.sin(theta)])

    # Input forces (F1 and tau_pendulum)
    tau = np.array([F1, 0, tau_pendulum])

    # Solve for accelerations: M(q) * q_ddot = tau - C - G
    q_ddot = np.linalg.solve(M, tau - C - G)

    # State derivatives
    dx = np.zeros(6)
    dx[0] = v1                   # d(X1)/dt = v1
    dx[1] = v2                   # d(X2)/dt = v2
    dx[2] = omega                # d(theta)/dt = omega
    dx[3] = q_ddot[0]            # d(v1)/dt = ddot_X1
    dx[4] = q_ddot[1]            # d(v2)/dt = ddot_X2
    dx[5] = q_ddot[2]            # d(omega)/dt = ddot_theta

    return dx


# Solve the system using solve_ivp
sol = solve_ivp(
    pendulum_cart_cart_nonlinear_dynamics_with_bounds, tspan, x0, 
    method='RK45', t_eval=np.linspace(tspan[0], tspan[1], 1000)
)

# Extract results
t = sol.t
x = sol.y
robot_X_absolute = x[0, :] + x[1, :]  # Robot's absolute X position
robot_Y_absolute = np.zeros_like(t)  # Assuming no Y motion

# Final position of the robot
X1_final = x[0, -1]
X2_final = x[1, -1]
robot_X_final = X1_final + X2_final
robot_Y_final = 0

# Display the final coordinates
print(f"Robot Final Position (Relative to Ship): X = {robot_X_final - X2_final:.2f} m, Y = {robot_Y_final:.2f} m")
print(f"Robot Final Position (Absolute): X = {robot_X_final:.2f} m, Y = {robot_Y_final:.2f} m")

# Plot the results
plt.figure(figsize=(10, 6))

# Position of Cart 1
plt.subplot(3, 1, 1)
plt.plot(t, x[0, :], label='Cart 1 (X1)', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('X1 (m)')
plt.title('Position of Cart 1 (X1)')
plt.grid()

# Position of Cart 2
plt.subplot(3, 1, 2)
plt.plot(t, x[1, :], label='Cart 2 (X2)', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('X2 (m)')
plt.title('Position of Cart 2 (X2)')
plt.grid()

# Pendulum angle
plt.subplot(3, 1, 3)
plt.plot(t, x[2, :], label='Pendulum Angle (θ)', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('θ (rad)')
plt.title('Pendulum Angle (θ)')
plt.grid()

plt.tight_layout()
plt.show()

# Plot robot trajectory relative to ship
plt.figure(figsize=(8, 8))
plt.plot(robot_X_absolute, robot_Y_absolute, label='Robot Trajectory', linewidth=1.5)
plt.gca().add_patch(plt.Rectangle((-L_ship/2, -W_ship/2), L_ship, W_ship,
                                   edgecolor='red', fill=False, linewidth=2))
plt.plot(robot_X_final, robot_Y_final, 'go', markersize=10, label='Final Position')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Robot Movement on Ship')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
