''' This file contains the helper function for the detection + tracking'''
import numpy as np
import math
import cv2
import tensorflow as tf

def get_bbox(results, frame): #pd is a 
    pd = results.pandas().xyxy[0].sort_values('confidence')
    
    nb_detected_objs = len(pd.index)
    
    bbox = np.zeros((4,nb_detected_objs),dtype=int)#4 rows for x0,y0,x1,y1 --> each column is the bb of 1 img
    confidence = np.zeros(nb_detected_objs)
    class_names = np.zeros(nb_detected_objs,dtype=object)
    images = []
    
    counter = 0
    for obj in pd.iloc:
        x0, y0, x1, y1 = obj.to_numpy()[0:4].astype(int)
        images.append(frame[y0:y1, x0:x1])
        confidence[counter] = obj.to_numpy()[4]
        class_names[counter] = obj.to_numpy()[6]
        #print(obj.to_numpy()[6])
        bbox[:,counter] = np.array([x0,y0,x1,y1],dtype=int)
        counter += 1
    
    #print(class_names)
    return bbox, confidence, class_names, images

def format_boxes(bboxes):#for deep sort
    for box in bboxes:
        ymin = int(box[1])
        xmin = int(box[0])
        ymax = int(box[3])
        xmax = int(box[2])
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height
    return bboxes

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''

    # Get the required landmarks coordinates.
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle = np.abs(angle)
    # Check if the angle is less than zero.
    if angle > 180.0:

        angle = 360-angle
        
    # Return the calculated angle.
    return angle

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 
            
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            
def classifyPose(kp_array, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        kp_array: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(kp_array[5],
                                      kp_array[7],
                                      kp_array[9])
    
    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_angle = calculateAngle(kp_array[6],
                                       kp_array[8],
                                       kp_array[10])
    
    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_angle = calculateAngle(kp_array[7],
                                         kp_array[5],
                                         kp_array[11])
    # Get the angle between the right hip, shoulder and elbow points.
    right_shoulder_angle = calculateAngle(kp_array[12],
                                          kp_array[6],
                                          kp_array[8])
    
    #print('left_elbow_angle: ', left_elbow_angle ,'\n right_elbow_angle: ', right_elbow_angle)
    #print('left_shoulder_angle: ', left_shoulder_angle ,'\n right_shoulder_angle: ', right_shoulder_angle)
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if left_elbow_angle > 125 and left_elbow_angle < 220 and right_elbow_angle > 125 and right_elbow_angle < 220:
        #label = 'T Pose'
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 70 and left_shoulder_angle < 110 and right_shoulder_angle > 70 and right_shoulder_angle < 110:
            label = 'T Pose'
                        
    if right_elbow_angle > 50 and right_elbow_angle < 130 and right_shoulder_angle > 70 and right_shoulder_angle < 110:
        label = "power to the people"
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return output_image, label
    
def get_pose_from_image_and_bounding_box(bbox, frame, bbox_width, bbox_height, interpreter):
    #IDEA: THE POSE HAS TO BE DETECTED IN 10 CONSECUTIVE IMAGES
    
    
    bbox = np.int32(bbox.clip(min=0))
    
    WIDTH_THRESHOLD,HEIGHT_THRESHOLD = 100,100
    
    #add condition to neglect bounding boxes under a certain size threshold
    bbox_size_sufficient = False
    
    if bbox_width > WIDTH_THRESHOLD and bbox_height > HEIGHT_THRESHOLD:
        bbox_size_sufficient = True
        
    
    if len(bbox)>0 and np.all(bbox>0) and bbox_size_sufficient:#we do not want to cover the cases where we are the boundary of the image --> pose classification behaves strangely
        
        frame_copy = frame
        img = frame_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        

        #perform movenet on img(img is the cropped image)
        img1 = img.copy()
        img1 = tf.image.resize_with_pad(np.expand_dims(img1, axis=0), 192,192)
        input_image = tf.cast(img1, dtype=tf.float32)

        # Setup input and output 
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions 
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # Rendering 
        draw_connections(img, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(img, keypoints_with_scores, 0.4)
        
        #cv2.putText(img, "left hip condfidence" + str(keypoints_with_scores.reshape((17,3))[11,2]),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)  
        
        landmarks = keypoints_with_scores.reshape((17,3))[:,0:2]#array of landmarks (x,y)

        label = ""
        #Classification
        if np.size(landmarks) != 0:
            img, label = classifyPose(landmarks, img, display=False)
            
    else:
        img = frame
        label = "no bounding box detected"
    return img, label