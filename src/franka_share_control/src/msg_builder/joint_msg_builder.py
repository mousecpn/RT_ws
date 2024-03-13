from sensor_msgs.msg import JointState as jointstateMsg
import rospy

PANDA_JOINT_NAME = ["panda_joint{}".format(i+1) for i in range(7)]
PANDA_JOINT_W_GRIPPER_NAME = ["panda_joint{}".format(i+1) for i in range(7)]

def joint_state_builder(name, jointstates):
    joint_state_msg = jointstateMsg()
    # joint_state_msg.header.stamp = rospy.Time.now()
    joint_state_msg.name = name
    joint_state_msg.position = jointstates
    return joint_state_msg