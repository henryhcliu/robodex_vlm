import requests
import json
import base64
from pathlib import Path
import time  
import numpy as np
import cv2


def convert_image_to_base64(color_image):
    # convert image from BGR to RGB (OpenCV uses BGR by default)
    # color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # encode image to JPEG format
    _, buffer = cv2.imencode('.jpg', color_image)
    
    # convert byte data to Base64 encoding
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    # show image
    # cv2.imshow("Color Image", color_image)  # show RGB image
    # cv2.waitKey(0)  # wait for key
    # cv2.destroyAllWindows()  # close all windows
    
    return base64_image

class GPT4Requester:
    def __init__(self,question_id,ignore_former_conversation=False):
        self.api_key = 'sk-F5TSmOakXsYXbz5Z3c4V2KWlZBe82PwQjgtfXSNWBKI2xLAP'
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.url = 'https://api.chsdw.top/v1/chat/completions'  # GPT-4 API URL
        self.system_prompt = """
            Skill Library (The skills you can use to solve the task. ):
            1 Identifying Object(label of the object) (According to input, this skill can segment the object in the image with a mask. You should generate the label to substitute label of the object in skill name. Input: “label of the object”; output: “RGBD image with mask”);
            2 Generating Grasping Point and Posture (Received the RGBD image with mask from skill 1, this skill can generate the desire pose and position for end effector to grasp the object. Input: “RGBD image with mask”; Output: “desire position and posture”);
            3 Generating Releasing Point (Received the RGBD image with mask from skill 1, this skill output desire position as  the destination point for end effector moving to.  Input: “RGBD image with mask”; Output: “desire position”);
            4 Moving Hand (Received the desire pose, this skill can move the hand to the desire position in the specified posture. Input: “desire position and posture”or “desire position”; Output: “None”);
            5 Grasping Object (Close end effector to grasp the object after skill 4. Prerequisite of this skill is that the hand is not holding other objects. Input: “None”; Output: “None”);
            6 Releasing Object (Open end effector to release the object after skill 4. Input: “None”; Output: “None“);
            7 Rotating Hand (The end effector rotate at a fixed angle around a fixed axis, while remaining its' posture. Input: “None”; Output: “None”);
            
            Hardware Description(the tools you can use):
            1 RGBD camera (Installed on the wrist, it can offer RGBD image including RGB and pixel-level depth information);
            2 Dexterous hand-arm system (Only single dexterous hand-arm system, which means the hand can not grasp the other object during holding one.).
            
            Response Format:
            Give the skill number sequence and add ## among them. For instance: 
            Final result:
            ##skill01(label of the object)##skill02##skill04##.
            Note: The input of one skill must match the output of the former one.
        """
        self.task_description = " "
        if question_id == 0:
            self.task_description = '''
            Task Description:
            Analyze the image, use the hand-arm robotic system to pick the apple and place it in the basket. 
            '''
        if question_id == 1:    
            self.task_description = """
            Task Description:
            Analyze the image, use the hand-arm robotic system to first open the microwave oven door, then pick the mango and place it inside the opened microwave oven. 
            """
        if question_id == 2:
            self.task_description = """
            Task Description:
            Analyze the image, use the hand-arm robotic system to open the pot with lid, then pick the orange and place it inside the opened pot.
            """
        if question_id == 3:
            self.task_description = """
            Task Description:
            Analyze the image, use the hand-arm robotic system to place all the fruits in the image inside the box.
            """
        
        
        self.question = self.task_description + self.system_prompt
        if ignore_former_conversation:
            self.ignore_former_conversation = "Ignore the former conversation."
            self.question = self.ignore_former_conversation + self.question
            print("ignore_former_conversation")
        if question_id == 4:
            self.question = """
            I want to put orange firstly
            
            """

        # good response:
        # """
#           Analyze the image, list all the objects in the image and analyze their affordance. 
#           Then, tell me how to put all the fruits into the pot. 
#           Your answer last time is:
#             "1. Remove the yellow lid from the toy pot.
# 2. Place the orange into the pot.
# 3. Place the red apple into the pot.
# 4. Place the green apple into the pot.
# 5. Put the yellow lid back onto the pot."
#         Now, suppose that I am a robot with only one hand and one arm, you should turn the process above into a skill sequence in number, using the skill library that I provided below:
#             1. Identifying Object(args: a label of the object) (Input: none; output: “RGBD image with mask”)
#             2. Generating Grasping Point(args: none) (Input: “RGBD image with mask”; output: “desire position”)
#             3. Generating Releasing Point (args: none) (Input: “RGBD image with mask”; output: “desire position”)
#             4. Moving Hand (args: none) (Input: “desire position”; output: none)
#             5. Grasping Object(args: none) (Input:none; output: none)
#             6. Releasing Object (args: none) (Input: none; output: none)
#             7. Rotating Object(args: label of the object that you just grasped) (Input: none; output: none)
            
#             Note: After generating, check to ensure that the input of the latter skill can strictly match the output of the former one.
            
#             Finally, give me a skill sequence in number after checking. A template is like this:
#             "Final result:
#             ##skill01(box)##skill03##skill04##skill06##"
        
        # """
        print("task_description: ", self.task_description)


    def request_gpt4(self, question, color_image):
        start_time = time.time()  # record start time
        data = self.create_request_data(question, color_image)
        response_text = ""  # initialize response text
        response = requests.post(self.url, headers=self.headers, data=json.dumps(data), stream=True)
        
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode('utf-8')
                if decoded_chunk.startswith('data:'):
                    # Remove the 'data: ' prefix and parse the JSON object
                    try:
                        parsed_chunk = json.loads(decoded_chunk[5:])
                        # print(parsed_chunk['choices'][0]['delta']['content'], end='')
                        # check if 'choices' and 'content' exist
                        if 'choices' in parsed_chunk and len(parsed_chunk['choices']) > 0:
                            delta = parsed_chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                # print(delta['content'])
                                response_text += delta['content']  # append content to response_text
                    except:
                        
                        pass
        
        print("response_text:", response_text)
        end_time = time.time()  # record end time
        execution_time = end_time - start_time  # calculate execution time
        print(f"execution_time: {execution_time:.2f} seconds")  # print execution time
        
        return response_text  # return collected text content

    def create_request_data(self, question, color_image):
        base64_image = convert_image_to_base64(color_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        data = {
            'model': 'gpt-4-vision-preview',  # use model that supports images
            'messages': messages,
            'stream': True,
            'max_tokens': 1024  # can be adjusted as needed
        }
        return data

if __name__ == '__main__':
    # image path
    image_path = r"/home/rl/dexVLM/src/grasp/assets/kitchen.jpg"  # use original string
    # microwave oven
    question = """
    Analyze the image, give one sentence to describe the image.
    """

    requester = GPT4Requester()
    # request GPT-4
    response = requester.request_gpt4(question, image_path)
    
