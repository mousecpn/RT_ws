import pyrealsense2 as rs
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import sys
from sensor_msgs.msg import CameraInfo



def realsense_test():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)


    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    depth_image = np.asanyarray(depth.get_data())
    color_image = np.asanyarray(color.get_data())


    depth_image[depth_image>1200] = 1200
    depth_image[depth_image<400] = 400

    depth_image = ((depth_image-np.min(depth_image)).astype("float32")/np.max(depth_image).astype("float32")*255)
    depth_image = depth_image.astype('uint16')

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.03),cv2.COLORMAP_JET)
    # depth_colormap_dim = depth_colormap.shape
    # color_colormap_dim = color_image.shape

    # if depth_colormap_dim != color_colormap_dim:
    #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1],depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
    #     images = np.hstack((resized_color_image, depth_colormap))
    # else:
    #     images = np.hstack((color_image, depth_colormap))
    cv2.imwrite("depth.jpg", depth_image)


    # depth_profile = depth.get_profile()
    # color_profile = color.get_profile()

    # cvsprofile = rs.video_stream_profile(color_profile)
    # dvsprofile = rs.video_stream_profile(depth_profile)

    # color_intrin = cvsprofile.get_intrinsics()
    # print(color_intrin)

    # depth_intrin = dvsprofile.get_intrinsics()
    # print(depth_intrin)

    # extrin = depth_profile.get_extrinsics_to(color_profile)
    # print(extrin)




class ImageListener(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.depth = None
        self.color = None
        self.intr = None
        self.extr = None
        # rospy.wait_for_service("/gqcnn/grasp_planner")
        # self.grasp_planner = rospy.ServiceProxy("/gqcnn/grasp_planner", GQCNNGraspPlanner)
        # rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.imageDepthCallback)
        # rospy.Subscriber("/camera/rgb/image_raw", Image, self.imageColorCallback)
        rospy.Subscriber("tag_detections_image", Image, self.imageColorCallback)
        # rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.intrinsics)
        # rospy.Subscriber("/camera/extrinsics/depth_to_color", CameraInfo, self.extrinsics)


    def imageDepthCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            cv_image = cv_image/1000.0
            # print(np.max(cv_image))
            self.depth = self.bridge.cv2_to_imgmsg(cv_image, encoding='64FC1')
        # pix = (data.width/2, data.height/2)
        # self.depth = data
            return
        except CvBridgeError as e:
            print(e)
            return
        
    def imageColorCallback(self, data):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self.color = data
        except CvBridgeError as e:
            print(e)
            return
        
    def intrinsics(self, data):
        self.intr = data
        return
    
    # def extrinsics(self, data):
    #     self.extr = data
    #     return
    
    
    def service_call_test(self):
        while not rospy.is_shutdown():
        # try:
            color_im = self.bridge.imgmsg_to_cv2(self.color, self.color.encoding)
            cv2.imwrite("rbg.jpg", color_im)
        

if __name__=='__main__':
    rospy.init_node("depth_image_processor")
    listener = ImageListener()
    rospy.sleep(1)
    # listener.prepare_planner_request()
    listener.service_call_test()
    # rospy.spin()