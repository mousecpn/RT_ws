import random
from scipy.spatial.transform import Rotation
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import qpsolvers as qp
import pickle

from spatialmath import SE3, base
import math
from typing import Union
from sensor_msgs.msg import JointState as jointstateMsg
import numpy as np
import tf.transformations as tft



def rand(size=None):
    if size==None:
        return 2*np.random.rand()-1
    else:
        return 2*np.random.rand(size)-1

def setCameraOnRobotWrist(p, robot_id, link_id, physicsClientId=0):
    distance = 1
    position, orientation = p.getLinkState(robot_id,link_id, physicsClientId)[4:6]
    position = np.array(position)
    orientation = list(orientation)
    R_mat = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3,3)
    z_direction = R_mat[:,2]
    y_direction = R_mat[:,1]
    x_direction = R_mat[:,0]

    camera_pose = position + 0.05*x_direction - z_direction*0.08
    tar_p = camera_pose+z_direction*distance
    # print(orientation)

    # p.removeAllUserDebugItems()
    # x_end_p = (np.array(camera_pose) + np.array(x_direction*2)).tolist()
    # x_line_id = p.addUserDebugLine(camera_pose,x_end_p,[1,0,0])# y 轴
    # y_end_p = (np.array(camera_pose) + np.array(y_direction*2)).tolist()
    # y_line_id = p.addUserDebugLine(camera_pose,y_end_p,[0,1,0])# z轴
    # z_end_p = (np.array(camera_pose) + np.array(z_direction*2)).tolist()
    # z_line_id = p.addUserDebugLine(camera_pose,z_end_p,[0,0,1])


    viewMatrix = p.computeViewMatrix(position, tar_p, -z_direction, physicsClientId=physicsClientId)
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=42.5,               # 摄像头的视线夹角
        # fov=80,               # 摄像头的视线夹角
        aspect=1,
        nearVal=0.01,            # 摄像头焦距下限
        farVal=10,               # 摄像头能看上限
        physicsClientId=physicsClientId
    )

    p.getCameraImage(
        width=320, height=200,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )
    return

def angle_axis(T, Td):
    e = np.empty(6)
    e[:3] = Td[:3, -1] - T[:3, -1]
    R = Td[:3, :3] @ T[:3, :3].T
    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if base.iszerovec(li):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        ln = base.norm(li)
        a = math.atan2(ln, np.trace(R) - 1) * li / ln

    e[3:] = a

    return e

def calculate_velocity(p, panda, cur_joint, tar_position, tar_orientation):
    # The pose of the Panda's end-effector
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)

    # R_Mat = np.array(p.getMatrixFromQuaternion(tar_orientation)).reshape(3,3)
    R_Mat = tft.quaternion_matrix(tar_orientation)[0:3,0:3]
    Tep = np.c_[R_Mat, tar_position.reshape(3,1)]
    Tep = np.r_[Tep, np.array([0,0,0,1]).reshape(1,4)]
    Tep = sm.SE3(Tep)


    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    # v, arrived = rtb.p_servo(Te, Tep, 5, 0.001)
    v, arrived = rtb.p_servo(Te, Tep, 1, 0.001)

    # Gain term (lambda) for control minimisation
    Y = 0.1

    # v += rand(v.shape[0]) * v * 0.5 # * np.array([1,1,1,0,0,0])

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp')

    # Apply the joint velocities to the Panda
    joint_velocity = qd[:n]

    return joint_velocity, arrived


def velocity_based_control(p, panda, cur_joint, tar_vel, ang_vel):
    # The pose of the Panda's end-effector
    n = 7
    panda.q = cur_joint
    Te = panda.fkine(cur_joint)
    
    tar_vel = Te.A[:3,:3] @ tar_vel

    # Spatial error
    e = np.sum(np.abs(np.r_[tar_vel, ang_vel]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v = np.r_[tar_vel, ang_vel]

    # Gain term (lambda) for control minimisation
    Y = 0.1

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6) * 8

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp')

    # Apply the joint velocities to the Panda
    joint_velocity = qd[:n]

    return joint_velocity


def pseudo_etasl(p, panda, data_dict):
    # The pose of the Panda's end-effector
    cur_joint = data_dict['cur_joint']
    tar_vel = data_dict['tar_vel']
    ang_vel = data_dict['ang_vel']
    tar_traj = data_dict['tar_traj']
    traj_weight = data_dict['traj_weight']
    goal = np.array(data_dict['goal'])
    goal_weight = data_dict['goal_weight']

    n = 7
    panda.q = cur_joint
    Aeq = []
    beq = []


    ################## user vel constraint ####################
    Te = panda.fkine(cur_joint)
    tar_vel = Te.A[:3,:3] @ tar_vel

    ################## goal constraint #########################
    Tep = np.c_[Te.A[:3,:3], goal.reshape(3,1)]
    Tep = np.r_[Tep, np.array([0,0,0,1]).reshape(1,4)]
    Tep = sm.SE3(Tep)
    eTep = Te.inv() * Tep

    # Spatial error
    e_1 = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v_goal, _ = rtb.p_servo(Te, Tep, 1, 0.01)
    Aeq.append(np.c_[panda.jacobe(panda.q), np.eye(6)] * goal_weight)
    beq.append(v_goal.reshape((6,))*goal_weight)


    ################## trajectory following constraint #########################
    if tar_traj is not None:
        mask_z = tar_traj[:,-1] < Te.A[2,-1]
        traj_pos = tar_traj[mask_z][0]
        Tep = np.c_[Te.A[:3,:3], traj_pos.reshape(3,1)]
        Tep = np.r_[Tep, np.array([0,0,0,1]).reshape(1,4)]
        Tep = sm.SE3(Tep)
        eTep = Te.inv() * Tep

        # Spatial error
        e_2 = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        v_traj, _ = rtb.p_servo(Te, Tep, 1, 0.01)
        Aeq.append(np.c_[panda.jacobe(panda.q), np.eye(6)] * traj_weight)
        beq.append(v_traj.reshape((6,)) * traj_weight)

    # Spatial error
    e = np.sum(np.abs(np.r_[tar_vel, ang_vel]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v = np.r_[tar_vel, ang_vel]

    # Gain term (lambda) for control minimisation
    Y = 0.1

    # v += rand(v.shape[0]) * v * 0.5 # * np.array([1,1,1,0,0,0])

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)

    # The equality contraints
    Aeq.append(np.c_[panda.jacobe(panda.q), np.eye(6)])
    beq.append(v.reshape((6,)))
    Aeq = np.concatenate(Aeq, axis=0)
    beq = np.concatenate(beq, axis=0)
    # Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    # beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='daqp',verbose=True)

    # Apply the joint velocities to the Panda
    joint_velocity = qd[:n]

    return joint_velocity