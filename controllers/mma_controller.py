import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.Tp = Tp
        self.models = [
            ManiuplatorModel(Tp, m3=0.1, r3=0.05),
            ManiuplatorModel(Tp, m3=0.01, r3=0.01),
            ManiuplatorModel(Tp, m3=1.0, r3=0.3),
        ]
        self.i = 0

        self.u_prev = np.zeros((2, 1))
        self.x_prev = np.zeros(4)

    def choose_model(self, x):
        x = np.array(x)

        errors = []
        for model in self.models:
            x_pred = self.x_prev.reshape(4, 1) + self.Tp * model.x_dot(self.x_prev, self.u_prev)
            errors.append(np.linalg.norm(x_pred.flatten() - x))

        self.i = int(np.argmin(errors))

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)

        q1, q2, q1_dot, q2_dot = x
        q = np.array([[q1], [q2]])
        q_dot = np.array([[q1_dot], [q2_dot]])

        omega = 20.0
        Kp = omega ** 2
        Kd = 2 * omega

        v = (
            q_r_ddot.reshape(2, 1)
            + Kd * (q_r_dot.reshape(2, 1) - q_dot)
            + Kp * (q_r.reshape(2, 1) - q)
        )

        model = self.models[self.i]
        u = (model.M(x) @ v + model.C(x) @ q_dot).flatten()

        self.x_prev = np.array(x)
        self.u_prev = u
        return u
