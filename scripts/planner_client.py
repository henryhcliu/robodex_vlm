#!/usr/bin/env python3
from request_gpt4 import GPT4Requester

import socket
import pickle
import time
import rospy
import pyrealsense2 as rs
import numpy as np
from cv_bridge import CvBridge

IP_ADDRESS = 'xx.xx.xx.xx' # replace with your server IP address

class PlannerClient:
    def __init__(self):
        self.server_address = (IP_ADDRESS, 12345)
        self.bridge = CvBridge()
        
        # add socket server to receive execution complete signal
        self.execution_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.execution_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.execution_server.settimeout(1.0)
        # need to modify IP address when rebooting
        self.execution_server.bind((IP_ADDRESS, 12346))  # use different port
        self.execution_server.listen(1)
        print("Execution server started on port 12346")
    
    def capture_images(self):
        """Capture images and automatically handle camera initialization and shutdown"""
        pipeline = None
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
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Error capturing images: {str(e)}")
            return None, None
        
        finally:
            # ensure camera is closed in any case
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass

    def wait_for_execution(self, timeout=60):
        """Wait for execution complete signal"""
        print("Waiting for execution complete signal...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                client_socket, address = self.execution_server.accept()
                try:
                    data_size = int.from_bytes(client_socket.recv(4), 'big')
                    data = b''
                    while len(data) < data_size:
                        packet = client_socket.recv(data_size - len(data))
                        if not packet:
                            break
                        data += packet
                    
                    command = pickle.loads(data)
                    if isinstance(command, dict) and command['type'] == 'execution_complete':
                        print("Received execution complete signal")
                        return True
                finally:
                    client_socket.close()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error waiting for execution: {e}")
        
        print("Timeout waiting for execution complete signal")
        return False
    
    def send_command(self, command):
        """Send command to planner server"""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect(self.server_address)
            
            # send command
            data = pickle.dumps(command)
            client_socket.send(len(data).to_bytes(4, 'big'))
            client_socket.sendall(data)
            
            # receive response
            response_size = int.from_bytes(client_socket.recv(4), 'big')
            response = b''
            while len(response) < response_size:
                packet = client_socket.recv(response_size - len(response))
                if not packet:
                    raise Exception("Connection closed by server")
                response += packet
            
            return pickle.loads(response)
            
        finally:
            client_socket.close()
    
    def execute_task(self, task_name, **kwargs):
        """Execute single task"""
        command = {'type': task_name}
        command.update(kwargs)
        
        print(f"\nExecuting task: {task_name}")
        
        # execute move_home before generate_mask
        if task_name == 'generate_mask':
            print("Executing automatic move_home before mask generation")
            if not self.execute_task('move_home'):
                print("Failed to complete automatic move_home")
                return False
        
        success = self.send_command(command)
        
        if success:
            print(f"Task {task_name} initiated successfully")
            # if task is move task, wait for execution complete
            if task_name in ['move_to_target_pose', 'move_home', 'pull']:
                if not self.wait_for_execution():
                    print(f"Failed to complete {task_name}: execution timeout")
                    return False
            
            print(f"Successfully completed {task_name}")
        else:
            print(f"Failed to initiate {task_name}")
            return False
        
        return True
    
    def execute_pipeline(self, tasks):
        """Execute a series of tasks"""

        for task in tasks:
            task_name = task.get('name')
            task_params = task.get('params', {})
            
            if not self.execute_task(task_name, **task_params):
                print(f"Pipeline failed at task: {task_name}")
                return False
            
            # add short delay between tasks
            time.sleep(task.get('delay', 1))
        
        print("\nPipeline completed successfully")
        return True
    
    def __del__(self):
        """Clean up resources"""
        try:
            self.execution_server.close()
        except:
            pass

    def parse_skill_sequence(self, response):
        """Parse GPT-4 returned skill sequence, convert to task flow"""
        # split skill sequence
        skills = response.split('##')[1:-1]  # remove empty strings at the beginning and end
        tasks = []
        
        for skill in skills:
            if not skill:
                continue
                
            # parse skill number and parameters
            if '(' in skill:
                skill_num = skill[:skill.find('(')].strip()
                param = skill[skill.find('(')+1:skill.find(')')].strip()
            else:
                skill_num = skill.strip()
                param = None
            
            # create corresponding task according to skill number
            if skill_num == 'skill01':
                tasks.append({
                    'name': 'generate_mask',
                    'params': {'prompt': param},
                    'delay': 1
                })
            elif skill_num == 'skill02':
                tasks.append({
                    'name': 'detect_grasp_pose',
                    'delay': 1
                })
            elif skill_num == 'skill03':
                tasks.append({
                    'name': 'detect_release_pose',
                    'delay': 1
                })
            elif skill_num == 'skill04':
                tasks.append({
                    'name': 'move_to_target_pose',
                    'delay': 1
                })
            elif skill_num == 'skill05':
                tasks.append({
                    'name': 'grasp',
                    'delay': 1
                })
            elif skill_num == 'skill06':
                tasks.append({
                    'name': 'release',
                    'delay': 1
                })
            elif skill_num == 'skill07':
                tasks.append({
                    'name': 'rotate',
                    'delay': 1
                })
            elif skill_num == 'skill08':
                tasks.append({
                    'name': 'move_home',
                    'delay': 2
                })
            elif skill_num == 'skill09':
                tasks.append({
                    'name': 'pull',
                    'delay': 1
                })
        
        return tasks

def main():
    # create client instance
    client = PlannerClient()

    # # image path
    # image_path = r"/home/rl/dexVLM/src/grasp/assets/kitchen.jpg"

    # capture images
    color_image, _ = client.capture_images()
    # get GPT-4 response
    requester = GPT4Requester(question_id=4,ignore_former_conversation=False)
    print("waiting for response...")
    response = requester.request_gpt4(requester.question, color_image)
    
    # parse response to generate task sequence
    tasks = client.parse_skill_sequence(response)
    print("\nGenerated task sequence:")
    for i, task in enumerate(tasks):
        print(f"{i+1}. {task['name']}", end='')
        if 'params' in task:
            print(f" with params: {task['params']}")
        else:
            print()
    
    # execute task sequence
    print("\nExecuting task sequence...")
    client.execute_pipeline(tasks)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nError executing pipeline: {e}") 
