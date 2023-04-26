import time
import sys

import numpy as np
from new_mpc_implementation.gait import gait
from new_mpc_implementation.footSwingController import footSwingController
from new_mpc_implementation.quadruped import Quadruped
from enum import Enum, auto

try:
    import mpc_osqp as mpc
except:
    print("No MPC module found")
    sys.exit()

DTYPE = np.float16
NUM_LEGS = 4
SIDE_SIGN = np.array([-1, 1, -1, 1])

class CoordinateAxis(Enum):
    X = auto()
    Y = auto()
    Z = auto()


def coordinateRotation(axis:CoordinateAxis, theta:float) -> np.ndarray:
    s = np.sin(float(theta))
    c = np.cos(float(theta))
    R:np.ndarray = None
    if axis is CoordinateAxis.X:
        R = np.array([1, 0, 0, 0, c, s, 0, -s, c], dtype=DTYPE).reshape((3,3))
    elif axis is CoordinateAxis.Y:
        R = np.array([c, 0, -s, 0, 1, 0, s, 0, c], dtype=DTYPE).reshape((3,3))
    elif axis is CoordinateAxis.Z:
        R = np.array([c, s, 0, -s, c, 0, 0, 0, 1], dtype=DTYPE).reshape((3,3))
    return R

class MPCLocomotion:
    def __init__(self, _dt, _iterationsBetweenMPC:int):
        self.iterationsBetweenMPC = int(_iterationsBetweenMPC)
        self.horizonLength = 10
        self.dt = _dt

        self.trotting = gait(self.horizonLength, np.array([0, 5, 5, 0]),  np.array([5, 5, 5, 5]), "Trotting")
        self.standing = gait(self.horizonLength, np.array([0, 0, 0, 0]),  np.array([10, 10, 10, 10]), "Standing")
        self.bounding = gait(self.horizonLength, np.array([5, 5, 0, 0]),  np.array([5, 5, 5, 5]), "Bounding")
        self.standing_three = gait(self.horizonLength, np.array([0, 0, 0, 0]),  np.array([0, 10, 10, 10]), "Bounding")

        self.dtMPC = self.dt * self.iterationsBetweenMPC
        self.default_iterations_between_mpc = self.iterationsBetweenMPC

        self.firstRun = True
        self.iterationCounter = 0
        self.pFoot = np.zeros((4,3), dtype=DTYPE) # world frame
        self.rFoot = np.zeros((4,3), dtype=DTYPE)
        self.f_ff = np.zeros((4,3), dtype=DTYPE)
        self.p_des_leg = np.zeros((4,3), dtype=DTYPE)
        self.v_des_leg = np.zeros((4,3), dtype=DTYPE)

        self.p_com_world = np.zeros((3,1), dtype=DTYPE)
        self.v_com_world = np.zeros((3,1), dtype=DTYPE)
        self.R_body = np.zeros((3,1), dtype=DTYPE)

        self.foot_positions = np.zeros((4,3), dtype=DTYPE) #body frame

        self.gait = self.trotting

        self.current_gait = 0
        self._x_vel_des = 0.0
        self._y_vel_des = 0.0
        self._yaw_turn_rate = 0.0

        self.xStart = 0.0
        self.yStart = 0.0

        self._roll_des = 0.0
        self._pitch_des = 0.0

        self._robot_parameter = Quadruped()

        self.footSwingTrajectories = [footSwingController() for _ in range(4)]
        self.swingTimes = np.zeros(4)
        self.contactState = np.zeros(4)
        self.swingTimeRemaining = np.zeros(4)
        self.cmpc_alpha = 1e-5
    
    def initialize(self):
        self.iterationCounter = 0

        # velocity commands in body frame
        self.roll_int = 0.0
        self.pitch_int = 0.0
        self.roll_comp = 0.0
        self.pitch_comp = 0.0

        self._x_vel_des = 0.0 
        self._y_vel_des = 0.0
        self._des_turn_rate = 0.0

        self.xStart = 0.0
        self.yStart = 0.0
        self.yawStart = 0.0

        self.v_des_robot = np.array([self._x_vel_des, self._y_vel_des, 0]).reshape((3,1))
        self.v_des_world = np.array([0, 0, 0]).reshape((3,1))
        self.firstSwing = [True, True, True, True]
        self.firstRun = True

        self._cpp_mpc = mpc.ConvexMpc(self._robot_parameter._bodyMass,
                                     list(self._robot_parameter._bodyInertia),
                                     NUM_LEGS,
                                     self.horizonLength,
                                     self.dtMPC,
                                     self.cmpc_alpha,
                                     mpc.QPOASES)

    def setupCmd(self, vx, vy, turn_rate, bodyHeight):
        self._body_height = bodyHeight
        self._x_vel_des = vx
        self._y_vel_des = vy
        self._yaw_turn_rate = turn_rate
        self.v_des_robot = np.array([self._x_vel_des, self._y_vel_des, 0])
        
    def updateMPCIfNeeded(self, robot, mpcTable):
        if(self.iterationCounter % self.iterationsBetweenMPC)==0:
            mpc_weight = self._robot_parameter._mpc_weights
            #flat ground
            gravity_projection_vec = np.array([0, 0, 1])
            com_rpy = robot.GetBaseRPY().flatten()
            # print(com_rpy)
            com_ang_vel_world = robot.GetBaseAngularVelocity().flatten()
            self.xStart += self.dt * self.iterationsBetweenMPC * self.v_des_world[0]
            self.yStart += self.dt * self.iterationsBetweenMPC * self.v_des_world[1]
            self.yawStart += self.dt * self.iterationsBetweenMPC * self._yaw_turn_rate

            if self.yawStart > 3.141 and com_rpy[2] < 0:
                self.yawStart -= 6.28
            if self.yawStart < -3.141 and com_rpy[2] > 0:
                self.yawStart += 6.28
            
            if abs(self.xStart - robot.GetBasePosition()[0]) > 0.1:
                self.xStart = robot.GetBasePosition()[0]
            if abs(self.yStart - robot.GetBasePosition()[1]) > 0.1:
                self.yStart = robot.GetBasePosition()[1]

            # change to xStart, yStart similar to C++ 
            
            desired_com_pos = np.array([0, 0, self._body_height])
            desired_com_pos[0] = self.xStart 
            desired_com_pos[1] = self.yStart

            # print("desired com pos ", desired_com_pos)
            desired_com_vel = self.v_des_world
            desired_com_rpy = np.array([self.roll_comp, self.pitch_comp, self.yawStart], dtype = DTYPE)
            # desired_com_rpy[1] = 0.3
            # desired_com_rpy[2] = self.yawStart
            desired_ang_vel_body = np.array([0.0, 0.0, self._yaw_turn_rate], dtype=DTYPE)
            desired_ang_vel = self.R_body.transpose() @ desired_ang_vel_body.reshape((3,1))
            # starttime =time.time()
            predicted_contact_forces = self._cpp_mpc.compute_contact_forces(
            mpc_weight, # mpc weights list(12,)
            list(self.p_com_world), # com_position (set x y to 0.0)
            list(self.v_com_world), # com_velocity
            list(com_rpy), # com_roll_pitch_yaw (set yaw to 0.0)
            list(gravity_projection_vec),  # Normal Vector of ground
            list(com_ang_vel_world), # com_angular_velocity
            np.asarray(mpcTable, dtype=DTYPE),  # Foot contact states
            np.array(self.rFoot.flatten(), dtype=DTYPE),  # foot_positions_base_frame
            # np.array(self.foot_positions.flatten(), dtype=DTYPE),
            np.ones(4, dtype=DTYPE) * 0.4,  # foot_friction_coeffs
            list(desired_com_pos),  # desired_com_position
            list(desired_com_vel),  # desired_com_velocity
            list(desired_com_rpy),  # desired_com_roll_pitch_yaw
            desired_ang_vel # desired_com_angular_velocity
            )
            
            # end = time.time()
            # print("mpc solve time ", end-starttime)
            for leg in range(4):
                self.f_ff[leg] = -self.R_body @ np.array(predicted_contact_forces[leg*3: (leg+1)*3],dtype=DTYPE)
            # print("force ", predicted_contact_forces)
            # print("f_ff ", self.f_ff)

    def run(self, robot):
        # self.gait = self.trotting
        self.gait = self.standing
        # self.gait = self.standing_three
        
        self.gait.setIterations(self.iterationsBetweenMPC, self.iterationCounter)
        
        self.p_com_world = np.asarray(robot.GetBasePosition())
        self.v_com_world = np.asarray(robot.GetBaseLinearVelocity())
        self.R_body = robot.GetBaseOrientationMatrix().transpose()
        rpy = robot.GetBaseRPY()

        self.v_des_world = self.R_body.T @ self.v_des_robot.reshape((3,1))
        self.foot_positions = robot.GetFootPositionsInBaseFrame() # body frame

        if abs(self.v_com_world[0]) > 0.2:
            self.roll_int += self.dt * (0.0 - rpy[0])/self.v_com_world[1]
        if abs(self.v_com_world[1]) > 0.1:
            self.pitch_int += self.dt * (0.0 - rpy[1])/self.v_com_world[0]
        

        self.roll_int = min(max(self.roll_int, -0.25), 0.25)
        self.pitch_int = min(max(self.pitch_int, -0.25), 0.25)

        self.roll_comp = self.v_com_world[1] * self.roll_int
        self.pitch_comp = self.v_com_world[0] * self.pitch_int

        for i in range(4):
            self.rFoot[i] = (self.R_body.T @ self.foot_positions[i].reshape((3,1))).reshape((1,3))
            self.pFoot[i] = (self.p_com_world.reshape((3,1)) + np.transpose(self.R_body) @ self.foot_positions[i].reshape((3,1))).reshape((1,3))
        # print(" r foot: ", self.rFoot)
        if self.firstRun:
            self.xStart = robot.GetBasePosition()[0]
            self.yStart = robot.GetBasePosition()[1]
            self.firstRun = False
            self.swingTimes[0] = self.dtMPC * self.gait._swing_duration
            self.swingTimes[1] = self.dtMPC * self.gait._swing_duration
            self.swingTimes[2] = self.dtMPC * self.gait._swing_duration
            self.swingTimes[3] = self.dtMPC * self.gait._swing_duration

            for i in range(4):
                self.footSwingTrajectories[i].set_swing_height(0.05)
                self.footSwingTrajectories[i].set_init_position(self.pFoot[i].reshape((3,1)))
                self.footSwingTrajectories[i].set_final_position(self.pFoot[i].reshape((3,1)))

        for i in range(4):
            if self.firstSwing[i]:
                self.swingTimeRemaining[i] = self.swingTimes[i]
            else:
                self.swingTimeRemaining[i] -= self.dt
        
            self.footSwingTrajectories[i].set_swing_height(0.05)
            # add flag for RL interface later
            # heuristic
            if True:
                offset = np.array([0, SIDE_SIGN[i] * 0.0838, 0]).reshape((3,1))
                pRobotFrame = robot.GetHipPositionsInBaseFrame()[i].reshape((3,1)) + offset 
                stance_time = self.dtMPC * self.gait._stance_duration       
                    
                pYawCorrected = coordinateRotation(CoordinateAxis.Z, -self._yaw_turn_rate * stance_time/2.0) @ pRobotFrame
                Pf = self.p_com_world.reshape((3,1)) + self.R_body.T @ (pYawCorrected + self.swingTimeRemaining[i] * self.v_des_robot.reshape((3,1)) )
                
                p_rel_max = 0.3
                pfx_rel = self.v_com_world[0] * 0.5 * stance_time + \
                                        0.03 * (self.v_com_world[0] - self.v_des_world[0]) + \
                                        (0.5 * self.p_com_world[2] / 9.81) * (-self.v_com_world[1] * self._des_turn_rate)
                                
                pfy_rel  = self.v_com_world[1] * 0.5 * stance_time + \
                                        0.03 * (self.v_com_world[1] - self.v_des_world[1]) + \
                                        (0.5 * self.p_com_world[2] / 9.81) * (-self.v_com_world[0] * self._des_turn_rate)
                                
                pfx_rel = min(max(pfx_rel, -p_rel_max), p_rel_max)
                pfy_rel = min(max(pfy_rel, -p_rel_max), p_rel_max)

                Pf[0] += pfx_rel
                Pf[1] += pfy_rel 

                Pf[2] = -0.003
                self.footSwingTrajectories[i].set_final_position(Pf)  
        
        swingPhases = self.gait.getSwingSubPhase()
        mpcTable = self.gait.getMPCtable()

        self.updateMPCIfNeeded(robot, mpcTable)

        self.iterationCounter += 1
        for foot in range(4):
            if swingPhases[foot] > 0.:
                if self.firstSwing[foot]:
                    self.firstSwing[foot] = False
                    self.footSwingTrajectories[foot].set_init_position(self.pFoot[foot].reshape((3,1)))

                self.footSwingTrajectories[foot].computeSwingTrajectory(swingPhases[foot])
                pDesFoot = self.footSwingTrajectories[foot]._p
                vDesFoot = self.footSwingTrajectories[foot]._v
                self.p_des_leg[foot] = (self.R_body @ (pDesFoot - self.p_com_world.reshape((3,1))) - robot.GetHipPositionsInBaseFrame()[foot].reshape((3,1))).T
                    
                self.v_des_leg[foot] = (self.R_body @ (vDesFoot - self.v_com_world.reshape((3,1)))).T
                self.contactState[foot] = 0
            else:
                self.firstSwing[foot] = True
                vDesFoot = np.zeros((3,1))
                self.v_des_leg[foot] = (self.R_body @ (vDesFoot - self.v_com_world.reshape((3,1)))).T
                self.contactState[foot] = 1
            