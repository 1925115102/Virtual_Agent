from typing import Any
import pygame
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os
from pygame.locals import *
from Action import Speak
import Velocity

# basic variables
FPS = 60
WIDTH = 720
HEIGHT = 720
WHITE = (255,255,255)

# initialize game
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption('virtual Agent')
clock = pygame.time.Clock()

# Load images 
static_image = pygame.image.load('library/images/agent/static/static.png')
static_image = pygame.transform.scale(static_image, (720, 720))

# Load sounds
sound_1 = pygame.mixer.Sound('library/audios/leigh_flat_lookaball.mp3')

# set cammera
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


speak_group = pygame.sprite.Group()
running = True
second_count = 0

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:

    while running:

        # cammera setting
        ret, frame = cap.read()
        # Detect stuff and render
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Print nose landmark.
        image_hight, image_width, _ = image.shape
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            nose = (f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})')
            leftEye = (f'Left eye coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * image_hight})')
            rightEye = (f'Right eye coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * image_hight})')

            # print("===========Current time:",datetime.utcnow(),"===========")
            # print(nose)
            # print(leftEye)
            # print(rightEye)
            nose_velocity = Velocity.appendPosition(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width)
            if (nose_velocity != None):
                print(nose_velocity)
                # detect if velocity exceed 400
                if nose_velocity >= 400 or nose_velocity <= -400:
                    speak = Speak()
                    speak_group.add(speak)
                    pygame.mixer.Sound.play(sound_1)

            
            
                
            lines = [str(datetime.utcnow()),nose,leftEye,rightEye,'\n']
            with open('writingData.txt','a') as f:
                f.write('\n'.join(lines))
        except:
            pass
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Webcam Feed', image)

        # set FPS
        clock.tick(FPS)

        # set background color
        screen.fill(WHITE)

        # Display static image
        screen.blit(static_image, (0, 0))

        speak_group.draw(screen)
        speak_group.update()

        # operation
        for event in pygame.event.get():
            # quit game
            if event.type == pygame.QUIT:
                running = False
            # keyboard event
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    speak = Speak()
                    speak_group.add(speak)
                    pygame.mixer.Sound.play(sound_1)

        pygame.display.update()



pygame.quit()

cap.release()
cv2.destroyAllWindows()