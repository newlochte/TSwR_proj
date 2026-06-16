import matplotlib.pyplot as plt
import numpy as np
from controllers.adrc_flc_controller import ADRFLController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.001
end = 3

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
# traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
traj_gen = Poly3(np.array([0., 0.]), np.array([np.pi/4, np.pi/6]), end)

p1 = 200.0
p2 = 200.0

omega_1 = 0.2 * p1
omega_2 = 0.2 * p2
ksi = 1.0

kp_est_1 = omega_1 ** 2
kp_est_2 = omega_2 ** 2
kd_est_1 = 2 * ksi * omega_1
kd_est_2 = 2 * ksi * omega_2

q0, qdot0, _ = traj_gen.generate(0.)
q1_0 = np.array([q0[0], qdot0[0]])
q2_0 = np.array([q0[1], qdot0[1]])

Kp = np.diag([kp_est_1, kp_est_2])
Kd = np.diag([kd_est_1, kd_est_2])
p = np.array([p1, p2])

controller = ADRFLController(Tp, np.concatenate([q0, qdot0]), Kp, Kd, p)


Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso = np.array(controller.eso.states)

plt.figure('ESO estimates vs. measured state')
plt.subplot(221)
plt.plot(T, eso[:, 0], label='q_1 (ESO)')
plt.plot(T, Q[:, 0], 'r', label='q_1 (true)')
plt.title('Joint 1 position')
plt.xlabel('t [s]')
plt.ylabel('q_1 [rad]')
plt.legend()
plt.subplot(222)
plt.plot(T, eso[:, 2], label='q_1_dot (ESO)')
plt.plot(T, Q[:, 2], 'r', label='q_1_dot (true)')
plt.title('Joint 1 velocity')
plt.xlabel('t [s]')
plt.ylabel('q_1_dot [rad/s]')
plt.legend()
plt.subplot(223)
plt.plot(T, eso[:, 1], label='q_2 (ESO)')
plt.plot(T, Q[:, 1], 'r', label='q_2 (true)')
plt.title('Joint 2 position')
plt.xlabel('t [s]')
plt.ylabel('q_2 [rad]')
plt.legend()
plt.subplot(224)
plt.plot(T, eso[:, 3], label='q_2_dot (ESO)')
plt.plot(T, Q[:, 3], 'r', label='q_2_dot (true)')
plt.title('Joint 2 velocity')
plt.xlabel('t [s]')
plt.ylabel('q_2_dot [rad/s]')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure('Tracking and control')
plt.subplot(221)
plt.plot(T, Q[:, 0], 'r', label='q_1')
plt.plot(T, Q_d[:, 0], 'b', label='q_1 desired')
plt.title('Joint 1 tracking')
plt.xlabel('t [s]')
plt.ylabel('q_1 [rad]')
plt.legend()
plt.subplot(222)
plt.plot(T, Q[:, 1], 'r', label='q_2')
plt.plot(T, Q_d[:, 1], 'b', label='q_2 desired')
plt.title('Joint 2 tracking')
plt.xlabel('t [s]')
plt.ylabel('q_2 [rad]')
plt.legend()
plt.subplot(223)
plt.plot(T, u[:, 0], 'r', label='u_1')
plt.plot(T, u[:, 1], 'b', label='u_2')
plt.title('Control signals')
plt.xlabel('t [s]')
plt.ylabel('u [Nm]')
plt.legend()
plt.tight_layout()
plt.show()