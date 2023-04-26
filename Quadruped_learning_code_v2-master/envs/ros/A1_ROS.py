"""This file implements the functionalities of a quadruped using pybullet.
getstate
send control
"""
import time

import numpy as np
import pybullet
from ros_python_interface.api import A1Controller


class A1ROS(object):
    def __init__(self):
        self._num_motors = 12
        self._num_legs = 4

        self.Kp_Joint = np.zeros(12)
        self.Kd_Joint = np.zeros(12)
        self.Kp_Cartesian = np.diag([0, 0, 0])
        self.Kd_Cartesian = np.diag([0, 0, 0])
        self.torque_cmds = np.zeros(12)

        self.A1 = A1Controller()
        self.Reset()

    # ================================= load urdf and setup ====================================#

    # ================================= State feedback ====================================#
    def GetBasePosition(self):
        # in world frame
        position = self.A1.get_base_position()
        return position

    def GetBaseOrientation(self):
        orientation = self.A1.get_base_orientation()
        return orientation

    def GetBaseRPY(self):
        ori = self.GetBaseOrientation()
        rpy = pybullet.getEulerFromQuaternion(ori)
        return np.asarray(rpy)

    def GetBaseOrientationMatrix(self):
        """ Get the base orientation matrix, as numpy array """
        baseOrn = self.GetBaseOrientation()
        return np.asarray(pybullet.getMatrixFromQuaternion(baseOrn)).reshape((3, 3))

    def GetBaseLinearVelocity(self):
        """ Get base linear velocities (dx, dy, dz) in world frame"""
        linVel = self.A1.get_base_linear_velocity()
        return np.asarray(linVel)

    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        """Transform the angular velocity from world frame to robot's frame.

    Args:
      angular_velocity: Angular velocity of the robot in world frame.
      orientation: Orientation of the robot represented as a quaternion.

    Returns:
      angular velocity of based on the given orientation.
    """
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.

        _, orientation_inversed = pybullet.invertTransform([0, 0, 0], orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = pybullet.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            pybullet.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    def GetBaseAngularVelocity(self):
        """ Get base angular velocities (droll, dpitch, dyaw) in world frame"""
        angVel = self.A1.get_base_angular_velocity()
        return np.asarray(angVel)

    def GetBaseAngularVelocityLocalFrame(self):
        angVel = self.A1.get_base_angular_velocity()
        orientation = self.GetBaseOrientation()
        return self.TransformAngularVelocityToLocalFrame(angVel, orientation)

    def GetMotorAngles(self):
        """Get all motor angles """
        motor_angles = self.A1.get_motor_angles()
        return motor_angles

    def GetMotorVelocities(self):
        """Get the velocity of all motors."""
        motor_velocities = self.A1.get_motor_velocities()
        return motor_velocities

    def GetMotorTorqueCmds(self):
        return self.torque_cmds

    def GetFootContacts(self):
        contacts = self.A1.get_foot_contacts()
        return contacts

    def GetInvalidContacts(self):
        return False

    def get_cam_view(self):
        pass

    # ================================== Send Cmd =======================================#

    def ApplyAction(self, kpJoint, kdJoint, qDes, qdotDes, tauDes):
        self.A1.send_motor_command(kpJoint, kdJoint, qDes, qdotDes, tauDes)

    def SetCartesianPD(self, kpCartesian, kdCartesian):
        self.Kp_Cartesian = kpCartesian
        self.Kd_Cartesian = kdCartesian

    def ComputeImpedanceControl(self, pDes, vDes):
        q = self.GetMotorAngles()
        qdot = self.GetMotorVelocities()
        torque = np.zeros(12)
        for i in range(4):
            pFoot = self.ComputeFootPosHipFrame(q[i * 3: i * 3 + 3], i)
            J = self.ComputeLegJacobian(q[i * 3: i * 3 + 3], i)
            vFoot = self.ComputeFootVelHipFrame(q[i * 3: i * 3 + 3], qdot[i * 3: i * 3 + 3], i)

            torque[i * 3:i * 3 + 3] = J.T @ (self.Kp_Cartesian @ (pDes[i * 3:i * 3 + 3] - pFoot)
                                             + self.Kd_Cartesian @ (vDes[i * 3:i * 3 + 3] - vFoot))

        return torque

    def ComputeForceControl(self, force_cmd):
        q = self.GetMotorAngles()
        torque = np.zeros(12)
        for i in range(4):
            J = self.ComputeLegJacobian(q[i * 3:i * 3 + 3], i)
            torque[i * 3:i * 3 + 3] = J.T @ force_cmd[i * 3:i * 3 + 3]

        return torque

    # =================================== Kinematics ========================================#
    def ComputeLegIK(self, foot_position, leg_id):
        side_sign = 1

        if leg_id == 0 or leg_id == 2:
            side_sign = -1

        l_up = 0.2
        l_low = 0.2
        l_hip = 0.0838 * side_sign
        x, y, z = foot_position[0], foot_position[1], foot_position[2]
        theta_knee = -np.arccos(
            (x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) / (2 * l_low * l_up))
        l = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
        theta_hip = np.arcsin(-x / l) - theta_knee / 2
        c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
        s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
        theta_ab = np.arctan2(s1, c1)
        qDes = np.array([theta_ab, theta_hip, theta_knee])
        if np.isnan(qDes).any():
            qDes = np.array([0, 0.5, -1.4])
        return qDes

    def ComputeFootPosHipFrame(self, q, leg_id):

        side_sign = 1
        if leg_id == 0 or leg_id == 2:
            side_sign = -1

        pos = np.zeros(3)
        l1 = 0.0838
        l2 = 0.2
        l3 = 0.2

        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s3 = np.sin(q[2])

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        c3 = np.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        pos[0] = -l3 * s23 - l2 * s2
        pos[1] = l1 * side_sign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
        pos[2] = l1 * side_sign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

        return pos

    def ComputeLegJacobian(self, q, leg_id):
        side_sign = 1
        if leg_id == 0 or leg_id == 2:
            side_sign = -1

        pos = np.zeros(3)
        l1 = 0.0838
        l2 = 0.2
        l3 = 0.2

        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s3 = np.sin(q[2])

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        c3 = np.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        J = np.zeros((3, 3))
        J[1, 0] = -side_sign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
        J[2, 0] = side_sign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
        J[0, 1] = -l3 * c23 - l2 * c2
        J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
        J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
        J[0, 2] = -l3 * c23
        J[1, 2] = -l3 * s23 * s1
        J[2, 2] = l3 * s23 * c1

        return J

    def ComputeFootVelHipFrame(self, q, qdot, leg_id):
        J = self.ComputeLegJacobian(q, leg_id)
        foot_vel = J @ qdot
        return foot_vel

    # ==================================== Reset =======================================#
    def Reset(self):
        self.A1.settle_robot()
        time.sleep(0.2)
