#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from grasp.srv import GraspPose, GraspPoseResponse
from demo import AnyGrasp
import argparse
import torch
import open3d as o3d

class GraspPoseServer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('grasp_pose_server')
        
        # Create parameter parser
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path', default="log/checkpoint_detection.tar")
        parser.add_argument('--max_gripper_width', type=float, default=0.1)
        parser.add_argument('--gripper_height', type=float, default=0.08)
        parser.add_argument('--top_down_grasp', default=True, action='store_true')
        parser.add_argument('--debug', default=True, action='store_true')    # Whether to visualize grasp point cloud
        self.cfgs = parser.parse_args([])
        
        # Initialize AnyGrasp model
        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()
        rospy.loginfo("AnyGrasp model loaded successfully")
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Create ROS service
        self.service = rospy.Service('detect_grasp_pose', GraspPose, self.handle_grasp_pose)
        rospy.loginfo("Grasp pose server is ready")

    def handle_grasp_pose(self, req):
        try:
            # Convert ROS image message to OpenCV format
            rgb_image = self.bridge.imgmsg_to_cv2(req.rgb_image, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(req.depth_image)
            print("rgb_image: ", rgb_image.shape)
            print("depth_image: ", depth_image.shape)
            
            # Camera parameters
            fx, fy = 608.013, 608.161
            cx, cy = 318.828, 241.382
            scale = 1000.0
            
            # Workspace limits, unit: meter
            lims = [-1, 1, -1, 1, 0, 1.0]
            
            # Process image
            colors = rgb_image.astype(np.float32) / 255.0
            depths = depth_image
            
            # Generate point cloud
            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depths / scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z
            
            mask = (points_z > 0) & (points_z < 1)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors = colors[mask].astype(np.float32)
            
            # Use AnyGrasp to get grasp pose
            rospy.loginfo("grasp pose calculation started")
            with torch.no_grad():
                gg, _ = self.anygrasp.get_grasp(
                    points, colors, lims=lims,
                    apply_object_mask=True,
                    dense_grasp=False,
                    collision_detection=True
                )
            rospy.loginfo("pose calculation completed")
            
            response = GraspPoseResponse()
            
            if len(gg) == 0:
                rospy.logwarn('No grasp detected after collision detection!')
                response.success = False
                return response
            
            # Get best grasp
            gg = gg.nms().sort_by_score()
            best_grasp = gg[0]
            
            # Visualize best grasp
            if self.cfgs.debug:
                # Create point cloud object
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(points)
                cloud.colors = o3d.utility.Vector3dVector(colors)
                
                # Apply coordinate system transformation
                trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
                cloud.transform(trans_mat)
                
                # Get gripper geometry
                best_gripper = gg[0:10].to_open3d_geometry_list()
                for gripper in best_gripper:
                    gripper.transform(trans_mat)
                
                # Display point cloud and gripper
                o3d.visualization.draw_geometries([*best_gripper, cloud])
                o3d.visualization.draw_geometries([best_gripper[0], cloud])
            
            # Fill response
            response.translation = best_grasp.translation.tolist()
            response.rotation_matrix = best_grasp.rotation_matrix.flatten().tolist()
            response.score = float(best_grasp.score)
            response.success = True
            
            # Clean GPU cache
            torch.cuda.empty_cache()
            
            rospy.loginfo("request processed")
            print("\n")
            return response
            
        except Exception as e:
            rospy.logerr(f"Error in grasp pose server: {str(e)}")
            response = GraspPoseResponse()
            response.success = False
            return response

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        server = GraspPoseServer()
        server.run()
    except rospy.ROSInterruptException:
        pass 