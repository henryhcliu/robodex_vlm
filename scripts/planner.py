#!/usr/bin/env python3
import rospy
import numpy as np
from PIL import Image
import cv2
from grasp.srv import MaskGenerate, GraspPose, set_angle
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import pyrealsense2 as rs
import socket
import pickle
import os
import json
import time
import threading
from scipy.spatial.transform import Rotation as R

IP_ADDRESS = 'xx.xx.xx.xx' # replace with your server IP address

class GraspPlanner:
    def __init__(self):
        rospy.init_node('grasp_planner')
        self.bridge = CvBridge()

        self.pipeline = None
        self.align = None
        
        self.color_image = None
        self.depth_image = None
        self.box_info = None
        self.target_pose = None

        # the home pose for object detection every time
        home_translation = [-0.45, 0, 0.7]
        rpy_deg = [-176, 0, 90]  # RPY angle value

        rpy_rad = np.array(rpy_deg) * np.pi / 180.0  # convert to radian
        rot_matrix = R.from_euler('xyz', rpy_rad).as_matrix()
        # flatten 3x3 matrix to list
        rot_list = rot_matrix.flatten().tolist()
        
        self.home_pose = {
            'translation': home_translation,  # unit: meter
            'rotation_matrix': rot_list,
            'score': 1.0  # set a fixed confidence score
        }
        
        # add execution status variables
        self.execution_completed = False
        self.is_executing = False
        self.wait_thread = None  # add waiting thread variable
        
        # wait for mask generation service available
        rospy.loginfo("Waiting for mask generation service...")
        rospy.wait_for_service('generate_mask')
        self.mask_service = rospy.ServiceProxy('generate_mask', MaskGenerate)
        
        rospy.loginfo("Waiting for grasp pose service...")
        rospy.wait_for_service('detect_grasp_pose')
        self.grasp_service = rospy.ServiceProxy('detect_grasp_pose', GraspPose)
        
        # wait for hand control service available
        rospy.loginfo("Waiting for hand control service...")
        rospy.wait_for_service('/right_inspire_hand/set_angle')
        self.hand_service = rospy.ServiceProxy('/right_inspire_hand/set_angle', set_angle)
        
        # initialize socket server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(1.0)
        self.server_socket.bind((IP_ADDRESS, 12345))
        self.server_socket.listen(1)
        rospy.loginfo("Planner server started, waiting for connection...")

    def capture_images(self):
        """capture images and automatically handle camera initialization and shutdown"""
        try:
            # initialize camera
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            
            # start camera
            pipeline.start(config)
            align = rs.align(rs.stream.color)
            
            # wait for camera to warm up
            rospy.sleep(1.0)
            
            # capture images
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            self.depth_image = np.asanyarray(depth_frame.get_data())
            self.color_image = np.asanyarray(color_frame.get_data())
            
            # close camera
            pipeline.stop()
            
            return self.color_image, self.depth_image
            
        except Exception as e:
            rospy.logerr(f"Error capturing images: {str(e)}")
            # ensure camera is closed in case of error
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass
            return None, None

    def generate_mask(self, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """generate mask and save box information"""
        try:
            # capture images
            color_image, _ = self.capture_images()
            if color_image is None:
                return False
            
            # call mask generation service
            rgb_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
            response = self.mask_service(rgb_msg, text_prompt, box_threshold, text_threshold)
            
            if not response.masks:
                rospy.logwarn("No masks detected")
                return False
            
            # get the mask with the highest score
            best_mask = max(response.masks, key=lambda x: x.score)
            self.box_info = {
                'box': [int(x) for x in best_mask.box],
                'label': best_mask.label,
                'score': float(best_mask.score)
            }
            
            rospy.loginfo(f"Successfully generated mask and saved box info: {self.box_info}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error in generate_mask: {str(e)}")
            return False

    def detect_grasp_pose(self):
        """detect grasp pose"""
        try:
            # check if necessary data is available
            if self.box_info is None:
                rospy.logerr("No box information found. Please run generate_mask first.")
                return False
            
            if self.color_image is None or self.depth_image is None:
                rospy.logerr("No images found. Please run generate_mask first.")
                return False
            
            # process images
            x1, y1, x2, y2 = self.box_info['box']
            mask = np.zeros_like(self.depth_image, dtype=bool)
            mask[y1:y2, x1:x2] = True
            
            depth_masked = self.depth_image.copy()
            depth_masked[~mask] = 2000
            
            rgb_masked = self.color_image.copy()
            rgb_masked[~mask] = 0
            
            # convert to ROS message
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_masked, "bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_masked)
            
            # call grasp service
            response = self.grasp_service(rgb_msg, depth_msg)
            
            if response.success:
                # save target pose
                self.target_pose = {
                    'translation': response.translation,
                    'rotation_matrix': response.rotation_matrix,
                    'score': response.score
                }
                # print("grasp pose:", self.target_pose)
                rospy.loginfo("Successfully detected grasp pose")
                return True
            else:
                rospy.logwarn("Failed to get grasp pose")
                return False
                
        except Exception as e:
            rospy.logerr(f"Error in detect_grasp_pose: {str(e)}")
            return False

    def detect_release_pose(self):
        """detect release pose"""
        try:
            # check if necessary data is available
            if self.box_info is None:
                rospy.logerr("No box information found. Please run generate_mask first.")
                return False
            
            if self.color_image is None or self.depth_image is None:
                rospy.logerr("No images found. Please run generate_mask first.")
                return False
            
            # get the center point of the box
            x1, y1, x2, y2 = self.box_info['box']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # get the depth information (unit: millimeter)
            depth = self.depth_image[center_y, center_x]
            
            # use fixed camera intrinsic parameters
            depth_intrin = rs.intrinsics()
            depth_intrin.width = 640
            depth_intrin.height = 480
            depth_intrin.ppx = 321.285  # principal point x
            depth_intrin.ppy = 237.486  # principal point y
            depth_intrin.fx = 383.970   # focal length x
            depth_intrin.fy = 383.970   # focal length y
            depth_intrin.model = rs.distortion.brown_conrady
            depth_intrin.coeffs = [0, 0, 0, 0, 0]  # distortion coefficients
            
            # convert pixel coordinates to camera coordinates (unit: meter)
            camera_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth)
            
            x = camera_point[0] / 1000.0  # convert to meter
            y = camera_point[1] / 1000.0
            z = camera_point[2] / 1000.0 
            
            
            rotation_matrix = [0, 1, 0,
                             0, 0, 1,
                             1, 0, 0]
            
            # save target pose
            self.target_pose = {
                'translation': tuple([x, y, z]),  # convert to tuple
                'rotation_matrix': rotation_matrix,
                'score': 1.0  # set a fixed confidence score
            }
            
            # add flag after successfully detecting release pose
            self.is_release_pose = True
            
            rospy.loginfo("Successfully detected release pose")
            return True
                
        except Exception as e:
            rospy.logerr(f"Error in detect_release_pose: {str(e)}")
            return False

    def move_to_target_pose(self):
        """send target pose to tf_convert"""
        try:
            if self.target_pose is None:
                rospy.logerr("No target pose available. Please detect grasp pose first.")
                return False
            
            # connect to tf_convert and send data
            client_socket = None
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(3.0)
                client_socket.connect(('10.7.145.140', 12347))
                
                # create command data
                command_data = {
                    'pose': self.target_pose
                }
                
                # if release pose, add special flag
                if hasattr(self, 'is_release_pose') and self.is_release_pose:
                    command_data['is_release'] = True
                    command_data['release_height'] = 0.15 # release height (meter)
                
                data = pickle.dumps(command_data)
                client_socket.send(len(data).to_bytes(4, 'big'))
                client_socket.sendall(data)
                
                response_size = int.from_bytes(client_socket.recv(4), 'big')
                server_response = b''
                while len(server_response) < response_size:
                    packet = client_socket.recv(response_size - len(server_response))
                    if not packet:
                        raise Exception("Connection closed by server")
                    server_response += packet
                
                # reset release pose flag
                if hasattr(self, 'is_release_pose'):
                    self.is_release_pose = False
                    rospy.loginfo("Reset release pose flag")
                
                rospy.loginfo("Successfully sent pose to tf_convert")
                return True
                
            except Exception as e:
                rospy.logerr(f"Failed to send pose to tf_convert: {e}")
                return False
            finally:
                if client_socket:
                    client_socket.close()
                
        except Exception as e:
            rospy.logerr(f"Error in move_to_target_pose: {str(e)}")
            return False

    def move_home(self):
        """move to initial pose"""
        try:
            # connect to tf_convert and send data
            client_socket = None
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(3.0)
                client_socket.connect(('10.7.145.140', 12347))
                
                # add flag indicating this is home pose
                home_command = {
                    'is_home': True,
                    'pose': self.home_pose
                }
                
                data = pickle.dumps(home_command)
                client_socket.send(len(data).to_bytes(4, 'big'))
                client_socket.sendall(data)
                
                response_size = int.from_bytes(client_socket.recv(4), 'big')
                server_response = b''
                while len(server_response) < response_size:
                    packet = client_socket.recv(response_size - len(server_response))
                    if not packet:
                        raise Exception("Connection closed by server")
                    server_response += packet
                
                rospy.loginfo("Successfully sent home pose to tf_convert")
                return True
                
            except Exception as e:
                rospy.logerr(f"Failed to send home pose to tf_convert: {e}")
                return False
            finally:
                if client_socket:
                    client_socket.close()
                
        except Exception as e:
            rospy.logerr(f"Error in move_home: {str(e)}")
            return False

    def control_hand(self, command):
        """control hand"""
        try:
            if command == 'grasp':
                # close hand
                response = self.hand_service(0, 0, 0, 0, 0, 0)
                success = response.angle_accepted
            elif command == 'release':
                # open hand
                response = self.hand_service(1000, 1000, 1000, 1000, 1000, 0)
                success = response.angle_accepted
            
            if success:
                rospy.loginfo(f"Successfully executed hand {command}")
                return True
            else:
                rospy.logerr(f"Failed to execute hand {command}")
                return False
                
        except Exception as e:
            rospy.logerr(f"Error controlling hand: {str(e)}")
            return False
        
    def pull(self):
        """execute pull action"""
        try:
            # connect to tf_convert and send data
            client_socket = None
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(3.0)
                client_socket.connect(('10.7.145.140', 12347))
                
                # create pull command
                pull_command = {
                    'is_pull': True,  # mark this is a pull command
                    'distance': -0.25   # move distance (meter)
                }
                
                data = pickle.dumps(pull_command)
                client_socket.send(len(data).to_bytes(4, 'big'))
                client_socket.sendall(data)
                
                response_size = int.from_bytes(client_socket.recv(4), 'big')
                server_response = b''
                while len(server_response) < response_size:
                    packet = client_socket.recv(response_size - len(server_response))
                    if not packet:
                        raise Exception("Connection closed by server")
                    server_response += packet
                
                rospy.loginfo("Successfully sent pull command to tf_convert")
                return True
                
            except Exception as e:
                rospy.logerr(f"Failed to send pull command to tf_convert: {e}")
                return False
            finally:
                if client_socket:
                    client_socket.close()
                
        except Exception as e:
            rospy.logerr(f"Error in pull: {str(e)}")
            return False

    def run(self):
        """run server loop"""
        while not rospy.is_shutdown():
            try:
                client_socket, address = self.server_socket.accept()
                rospy.loginfo(f"Connected to {address}")
                
                try:
                    # receive data
                    data_size = int.from_bytes(client_socket.recv(4), 'big')
                    data = b''
                    while len(data) < data_size:
                        packet = client_socket.recv(data_size - len(data))
                        if not packet:
                            raise Exception("Connection closed by client")
                        data += packet
                    
                    # parse command
                    command = pickle.loads(data)
                    success = False
                    
                    if isinstance(command, dict):
                        if command['type'] == 'execution_complete':
                            self.execution_completed = command['status']
                            success = True
                        elif self.is_executing:
                            rospy.logwarn("Cannot execute new command while robot is moving")
                            success = False
                        else:
                            # process command
                            if command['type'] == 'generate_mask':
                                success = self.generate_mask(command['prompt'])
                            elif command['type'] == 'detect_grasp_pose':
                                success = self.detect_grasp_pose()
                            elif command['type'] == 'detect_release_pose':
                                success = self.detect_release_pose()
                            elif command['type'] == 'move_to_target_pose':
                                success = self.move_to_target_pose()
                            elif command['type'] == 'grasp':
                                success = self.control_hand('grasp')
                            elif command['type'] == 'release':
                                success = self.control_hand('release')
                            elif command['type'] == 'move_home':
                                success = self.move_home()
                            elif command['type'] == 'pull':
                                success = self.pull()
                            else:
                                rospy.logwarn(f"Unknown command type: {command['type']}")
                    
                    # send response
                    response = pickle.dumps(success)
                    client_socket.send(len(response).to_bytes(4, 'big'))
                    client_socket.sendall(response)
                    
                except Exception as e:
                    rospy.logerr(f"Error handling client request: {e}")
                finally:
                    client_socket.close()
                    
            except socket.timeout:
                continue
            except Exception as e:
                rospy.logerr(f"Socket error: {e}")
                continue

    def __del__(self):
        """clean up resources"""
        try:
            self.server_socket.close()
        except:
            pass

if __name__ == '__main__':
    try:
        planner = GraspPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass 