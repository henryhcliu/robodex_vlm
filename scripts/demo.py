"""
    This script is used to test the graspnet API.
    It will load the model and detect the grasp pose of the object in the image.
    To run this script, you need to run demo.sh.
"""

import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import time
from gsnet import AnyGrasp

from graspnetAPI import GraspGroup

import pyrealsense2 as rs
import cv2

import ultralytics
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path',default="log/checkpoint_detection.tar", help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.08, help='Gripper height')
parser.add_argument('--top_down_grasp', default=True, action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', default=True, action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.2, cfgs.max_gripper_width))

def show_point_cloud(color_image, depth_image, fx, fy, cx, cy, scale):
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Generate point cloud coordinates
    xmap, ymap = np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    
    points_z = depth_image / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    
    # Filter invalid points
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask]
    
    # Get corresponding colors
    colors = color_image.astype(np.float32) / 255.0
    colors = colors[mask]
    
    # Set the points and colors of the point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Display the point cloud
    o3d.visualization.draw_geometries([pcd, coordinate_frame])

def grasp_box():
    try:
        
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
        print("anygrasp loaded")

        # Read RGB image from local
        colors = cv2.imread('../assets/color_image.jpg')
        
        # Read depth data from local
        depths = np.load('../assets/depth_image_data.npy')

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depths, alpha=0.4), cv2.COLORMAP_JET)
        colors = colors.astype(np.float32) / 255.0

        # Use original camera intrinsic parameters
        fx, fy = 608.013, 608.161
        cx, cy = 318.828, 241.382
        scale = 1000.0

        # Show original point cloud
        # print("Showing original point cloud...")
        # show_point_cloud(colors, depths, fx, fy, cx, cy, scale)

        # Set workspace
        xmin, xmax = -0.35, 0.35
        ymin, ymax = -0.35, 0.35
        zmin, zmax = 0, 1.0
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # Get point cloud
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        print("querying grasp")

        # Use torch.no_grad() to reduce memory usage
        with torch.no_grad():
            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, 
                                         apply_object_mask=True, 
                                         dense_grasp=True, 
                                         collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return None
        else:
            gg = gg.nms().sort_by_score()
            gg_pick = gg[0:20]
            print(gg_pick.scores)
            print('grasp score:', gg_pick[0].score)
            print('gg_pick :', gg_pick[0])

            # Construct return dictionary
            best_grasp = gg_pick[0]
            grasp_result = {
                'translation': best_grasp.translation,
                'rotation_matrix': best_grasp.rotation_matrix,
                'score': float(best_grasp.score)
            }

        print("finished")
        cv2.waitKey(0)

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            o3d.visualization.draw_geometries([*grippers, cloud])
            o3d.visualization.draw_geometries([grippers[0], cloud])

        # Return grasp pose dictionary
        return grasp_result

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU memory is insufficient, try to clean memory...")
            torch.cuda.empty_cache()
            return None
        else:
            raise e

def capture_images(visualize=True):

    pipeline = rs.pipeline()
    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream
    # Start pipeline
    pipeline.start(config)

    # Create an alignment object, the target is the RGB stream
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        # Wait for frames
        frames = pipeline.wait_for_frames()

        # Align the depth frame to the RGB
        aligned_frames = align.process(frames)

        # Get the aligned depth image and RGB image
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_profile = aligned_depth_frame.get_profile()
        # print("parameters:", depth_profile)
        dvsprofile = rs.video_stream_profile(depth_profile)
        depth_intrin = dvsprofile.get_intrinsics()
        print("depth_intrin", depth_intrin)
        # Convert the depth image to a NumPy array
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print("depth_image", depth_image)
        # 
        class_ids, class_labels, boxes, confidences, bg_color_img, bg_depth_img = process_images(color_image, depth_image)

        print("class id:", class_ids)
        if visualize:
            # Visualize the depth image (need to convert to pseudocolor)
            # print("bg_depth_img", bg_depth_img)
            bg_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(bg_depth_img, alpha=0.4), cv2.COLORMAP_JET)
            if class_ids is not None:
                for box, class_id, confidence in zip(boxes, class_ids, confidences):
                    x, y, w, h = box
                    # Convert xywh to xyxy
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    
                    # Draw the bounding box on the image
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2
                    # Prepare the label text
                    label = f"{class_labels[int(class_id)]}: {confidence:.2f}"
                    # Draw the label at the top-left corner of the bounding box
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    
                    # print the depth pixels within bbox
                    box_depth = bg_depth_img[y1:y2, x1:x2]
                    if class_id > 6:
                        print("Depth values within box:")
                        print(box_depth)
                        # np.savetxt(f'depths_data_class_{int(class_id)}.txt', box_depth, fmt='%.2f')
            # Show RGB and depth images
            cv2.imshow('bg_depth_image', bg_depth_img)
            cv2.imshow('bg_depth_colormap', bg_depth_colormap)
            cv2.imshow('Color Image', color_image)
            cv2.imshow('bg_color_img', bg_color_img)
            cv2.waitKey(0)

    finally:
        # Stop pipeline
        pipeline.stop()
        cv2.destroyAllWindows()

    return bg_depth_img, bg_color_img  # Return processed depth image and RGB image


def process_images(rgb_image, depth_image):
    model = YOLO("./yolo11_hitv2.pt")
    # rgb_image = cv2.imread("examples/pic/01.jpg")
    annotator = Annotator(rgb_image, line_width=1)

    # Execute object tracking
    results = model.predict(source=rgb_image, conf=0.25, show=False)

    # Extract bounding boxes, object category id, category label, mask
    boxes = results[0].boxes.xywh.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_labels = results[0].names
    confidences = results[0].boxes.conf.cpu().numpy()

    # Create a mask of all True values
    mask = np.ones(rgb_image.shape[:2], dtype=bool)
    # Iterate over each box, setting the area to False
    confidences = results[0].boxes.conf.cpu().numpy()
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        if confidence > 0.5 and class_id > 6:
            x, y, w, h = box
            # x1 = int(x - w / 2)
            # y1 = int(y - h / 2)
            # x2 = int(x + w / 2)
            # y2 = int(y + h / 2)

            # Calculate the expanded width and height by 10%, considering the edge depth information
            w_expanded = w * 1.6
            h_expanded = h * 1.6

            # Calculate the new bounding box coordinates
            x1 = int(x - w_expanded / 2)
            y1 = int(y - h_expanded / 2)
            x2 = int(x + w_expanded / 2)
            y2 = int(y + h_expanded / 2)
            
            # Ensure the coordinates do not exceed the image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(rgb_image.shape[1], x2)
            y2 = min(rgb_image.shape[0], y2)
            mask[y1:y2, x1:x2] = False

    # print("mask:", mask) 
    # Calculate the mean of pixels outside boxes
    background_mean = rgb_image[mask].mean(axis=0)
    bg_color_img = np.copy(rgb_image)
    bg_depth_img = np.copy(depth_image)
    bg_color_img[mask] = background_mean
    bg_depth_img[mask] = 10000

    # # Return the required results
    return class_ids, class_labels, boxes, confidences, bg_color_img, bg_depth_img

# Example call
# rgb_image, depth_image = capture_images(visualize=False)
# masks, class_ids, class_labels, boxes, boxes_shape = process_images(rgb_image, depth_image)

def detect_grasp_box(rgb_image, depth_image):
    """Detect grasp pose, accept complete RGB image and depth image (depth values outside boxes are set to 2m)"""
    try:
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
        print("anygrasp loaded")

        # Use the incoming RGB image
        colors = rgb_image
        
        # Use the incoming depth data
        depths = depth_image

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depths, alpha=0.4), cv2.COLORMAP_JET)
        colors = colors.astype(np.float32) / 255.0

        # Use original camera intrinsic parameters
        fx, fy = 608.013, 608.161
        cx, cy = 318.828, 241.382
        scale = 1000.0

        # Set workspace
        xmin, xmax = -0.35, 0.35
        ymin, ymax = -0.35, 0.35
        zmin, zmax = 0, 1.0
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # Get point cloud
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # Only keep points within the valid depth range (automatically filter out points beyond 2m)
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        print("querying grasp")

        # Use torch.no_grad() to reduce memory usage
        with torch.no_grad():
            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, 
                                         apply_object_mask=True, 
                                         dense_grasp=True,                            # Whether to perform dense grasping
                                         collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return None
        else:
            gg = gg.nms().sort_by_score()
            gg_pick = gg[0:20]
            print(gg_pick.scores)
            print('grasp score:', gg_pick[0].score)
            print('gg_pick :', gg_pick[0])

            # Construct return dictionary
            best_grasp = gg_pick[0]
            grasp_result = {
                'translation': best_grasp.translation,
                'rotation_matrix': best_grasp.rotation_matrix,
                'score': float(best_grasp.score)
            }

        print("finished")
        cv2.waitKey(0)

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            o3d.visualization.draw_geometries([*grippers, cloud])
            o3d.visualization.draw_geometries([grippers[0], cloud])

        # Clean GPU cache
        torch.cuda.empty_cache()
        # Return grasp pose dictionary
        return grasp_result

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU memory is insufficient, try to clean memory...")
            torch.cuda.empty_cache()
            return None
        else:
            raise e

def visualize_grasp_pose(rgb_image, depth_image):
    """Visualize grasp pose"""
    try:
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
        print("anygrasp loaded")

        # Use the incoming RGB image
        colors = rgb_image
        
        # Use the incoming depth data
        depths = depth_image

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depths, alpha=0.4), cv2.COLORMAP_JET)
        colors = colors.astype(np.float32) / 255.0

        # Use original camera intrinsic parameters
        fx, fy = 608.013, 608.161
        cx, cy = 318.828, 241.382
        scale = 1000.0

        # Set workspace
        xmin, xmax = -0.35, 0.35
        ymin, ymax = -0.35, 0.35
        zmin, zmax = 0, 1.0
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # Get point cloud
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # Only keep points within the valid depth range (automatically filter out points beyond 2m)
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        print("querying grasp")
        time.sleep(7)

        # Use torch.no_grad() to reduce memory usage
        with torch.no_grad():
            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, 
                                         apply_object_mask=True, 
                                         dense_grasp=True,                            # Whether to perform dense grasping
                                         collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return None
        else:
            gg = gg.nms().sort_by_score()
            gg_pick = gg[0:20]
            print(gg_pick.scores)
            print('grasp score:', gg_pick[0].score)
            print('gg_pick :', gg_pick[0])

            # Construct return dictionary
            best_grasp = gg_pick[0]
            grasp_result = {
                'translation': best_grasp.translation,
                'rotation_matrix': best_grasp.rotation_matrix,
                'score': float(best_grasp.score)
            }

        print("finished")
        cv2.waitKey(0)

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            o3d.visualization.draw_geometries([*grippers, cloud])
            o3d.visualization.draw_geometries([grippers[0], cloud])

        # Clean GPU cache
        torch.cuda.empty_cache()
        # Return grasp pose dictionary
        return grasp_result

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU memory is insufficient, try to clean memory...")
            torch.cuda.empty_cache()
            return None
        else:
            raise e
    



if __name__ == '__main__':
    # capture_images()

    # grasp_box()

    # Read RGB image from local
    colors = cv2.imread('../assets/color_image.jpg')
    # Read depth data from local
    depths = np.load('../assets/depth_image_data.npy')

    # Call detect_grasp_box function
    grasp_result = detect_grasp_box(colors, depths)  # Pass RGB image and depth image
