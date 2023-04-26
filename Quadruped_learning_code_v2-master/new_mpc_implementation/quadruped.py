import numpy as np
DTYPE = np.float16

class Quadruped:
    def __init__(self):
        self._abadLinkLength = 0.0838
        self._hipLinkLength = 0.2
        self._kneeLinkLength = 0.2
        self._kneeLinkY_offset = 0.0
        self._bodyMass = 12.0
        self._bodyInertia = np.array([0.0168, 0, 0, 
                                      0, 0.0565, 0, 
                                      0, 0, 0.064], dtype=DTYPE)
        self._bodyHeight = 0.26
        self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            # self._mpc_weights = [1., 1., 0, 0, 0, 20, 0., 0., .1, .1, .1, .0, 0]
        self._mpc_weights = [0.25, 0.25, 10, 2.5, 2.5, 20, 0, 0, 0.3, 0.5, 0.5, 0.5, 0]