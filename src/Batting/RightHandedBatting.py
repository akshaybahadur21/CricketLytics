from collections import deque

import mediapipe as mp

from src.Batting.Batting import Batting
from src.ThreadedCamera import ThreadedCamera
from src.utils import *


class RightHandedBatting(Batting):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_landmark_drawing_spec = self.mp_drawing.DrawingSpec(thickness=5, circle_radius=2, color=(0, 0, 255))
        self.pose_connection_drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        self.PRESENCE_THRESHOLD = 0.5
        self.VISIBILITY_THRESHOLD = 0.5
        self.threaded_camera_front = ThreadedCamera("resources/right_front_1.mp4")
        self.cap_front = cv2.VideoCapture("resources/right_front_1.mp4")
        self.threaded_camera_side = ThreadedCamera("resources/right_side.mp4")
        self.cap_side = cv2.VideoCapture("resources/right_side.mp4")

    def front_batting(self):
        pts = deque(maxlen=64)
        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while True:
                success, image = self.threaded_camera_front.show_frame()
                # success, image = self.cap_front.read()
                if not success or image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.pose_landmark_drawing_spec,
                    connection_drawing_spec=self.pose_connection_drawing_spec)
                idx_to_coordinates = get_idx_to_coordinates(image, results)

                # Will only look at left side for right handed batsman
                try:
                    # knee angle for left knee
                    l1 = np.linspace(idx_to_coordinates[23], idx_to_coordinates[25], 100)
                    l2 = np.linspace(idx_to_coordinates[25], idx_to_coordinates[27], 100)
                    cv2.line(image, (int(l1[99][0]), int(l1[99][1])), (int(l1[59][0]), int(l1[59][1])), thickness=6,
                             color=(0, 0, 255))
                    cv2.line(image, (int(l2[0][0]), int(l2[0][1])), (int(l2[40][0]), int(l2[40][1])), thickness=6,
                             color=(0, 0, 255))
                    ang2 = ang((idx_to_coordinates[23], idx_to_coordinates[25]),
                               (idx_to_coordinates[25], idx_to_coordinates[27]))
                    cv2.putText(image, str(round(ang2, 2)), (idx_to_coordinates[25][0] + 20, idx_to_coordinates[25][1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(0, 255, 0), thickness=2)
                    center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=15)
                    axes = (radius, radius)
                    draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)

                except:
                    pass

                try:
                    # wrist elbow shoulder
                    l1 = np.linspace(idx_to_coordinates[15], idx_to_coordinates[13], 100)
                    l2 = np.linspace(idx_to_coordinates[13], idx_to_coordinates[11], 100)
                    cv2.line(image, (int(l1[99][0]), int(l1[99][1])), (int(l1[59][0]), int(l1[59][1])), thickness=6,
                             color=(0, 0, 255))
                    cv2.line(image, (int(l2[0][0]), int(l2[0][1])), (int(l2[40][0]), int(l2[40][1])), thickness=6,
                             color=(0, 0, 255))
                    ang2 = ang((idx_to_coordinates[15], idx_to_coordinates[13]),
                               (idx_to_coordinates[13], idx_to_coordinates[11]))
                    cv2.putText(image, str(round(ang2, 2)), (idx_to_coordinates[13][0] + 10, idx_to_coordinates[13][1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(0, 255, 0), thickness=2)
                    center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=15)
                    axes = (radius, radius)
                    draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)

                except:
                    pass

                try:
                    # shoulder back knee
                    l1 = np.linspace(idx_to_coordinates[11], idx_to_coordinates[23], 100)
                    l2 = np.linspace(idx_to_coordinates[23], idx_to_coordinates[25], 100)
                    cv2.line(image, (int(l1[99][0]), int(l1[99][1])), (int(l1[75][0]), int(l1[75][1])), thickness=6,
                             color=(0, 0, 255))
                    cv2.line(image, (int(l2[0][0]), int(l2[0][1])), (int(l2[30][0]), int(l2[30][1])), thickness=6,
                             color=(0, 0, 255))
                    ang2 = ang((idx_to_coordinates[11], idx_to_coordinates[23]),
                               (idx_to_coordinates[23], idx_to_coordinates[25]))
                    cv2.putText(image, str(round(ang2, 2)), (idx_to_coordinates[23][0] + 20, idx_to_coordinates[23][1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(0, 255, 0), thickness=2)
                    center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=15)
                    axes = (radius, radius)
                    draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)

                except:
                    pass

                try:
                    # Distance between my left foot and right hand
                    left_foot = idx_to_coordinates[29]
                    left_hand = idx_to_coordinates[17]
                    cv2.line(image, (left_foot[0], left_foot[1] + 50), (left_hand[0], left_foot[1] + 50), thickness=2,
                             color=(255, 255, 0))
                    cv2.line(image, (left_foot[0], left_foot[1] + 50), (left_foot[0], left_foot[1] + 60), thickness=2,
                             color=(255, 255, 0))
                    cv2.line(image, (left_hand[0], left_foot[1] + 50), (left_hand[0], left_foot[1] + 60), thickness=2,
                             color=(255, 255, 0))
                    cv2.putText(image, str(max(0, left_foot[0] - left_hand[0])) + " px",
                                (left_foot[0] - 50, left_foot[1] + 80),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.9, color=(0, 255, 0), thickness=2)
                except:
                    pass

                if 0 in idx_to_coordinates:
                    cv2.putText(image, "Batsman : Right Handed",
                                (idx_to_coordinates[0][0] - 120, idx_to_coordinates[0][1] - 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.9, color=(0, 255, 0), thickness=2)

                # Tracking the green colored ball

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                kernel = np.ones((5, 5), np.uint8)
                Lower_green = np.array([29, 86, 6])
                Upper_green = np.array([64, 255, 255])
                mask = cv2.inRange(hsv, Lower_green, Upper_green)
                # mask = cv2.erode(mask, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
                res = cv2.bitwise_and(image, image, mask=mask)
                cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                center = None

                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    if radius > 10:
                        cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.circle(image, center, 5, (0, 0, 255), -1)

                # pts.appendleft(center)
                # for i in range(1, len(pts)):
                #     if pts[i - 1] is None or pts[i] is None:
                #         continue
                #     thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)p
                #     cv2.line(image, pts[i - 1], pts[i], (0, 0, 225), thick)

                cv2.imshow('Cricket', rescale_frame(image, percent=180))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                key = cv2.waitKey(1)
                if key == ord('p'):
                    cv2.waitKey(-1)  # wait until any key is pressed
        self.pose.close()


    def side_batting(self):
        pts = deque(maxlen=64)
        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while True:
                success, image = self.threaded_camera_side.show_frame()
                # success, image = self.cap_side.read()
                if not success or image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.pose_landmark_drawing_spec,
                    connection_drawing_spec=self.pose_connection_drawing_spec)
                idx_to_coordinates = get_idx_to_coordinates(image, results)

                # Will only look at left side for right handed batsman
                try:
                    # knee angle for left knee
                    l1 = np.linspace(idx_to_coordinates[23], idx_to_coordinates[25], 100)
                    l2 = np.linspace(idx_to_coordinates[25], idx_to_coordinates[27], 100)
                    cv2.line(image, (int(l1[99][0]), int(l1[99][1])), (int(l1[59][0]), int(l1[59][1])), thickness=6,
                             color=(0, 0, 255))
                    cv2.line(image, (int(l2[0][0]), int(l2[0][1])), (int(l2[40][0]), int(l2[40][1])), thickness=6,
                             color=(0, 0, 255))
                    ang2 = ang((idx_to_coordinates[23], idx_to_coordinates[25]),
                               (idx_to_coordinates[25], idx_to_coordinates[27]))
                    cv2.putText(image, str(round(ang2, 2)), (idx_to_coordinates[25][0] + 10, idx_to_coordinates[25][1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(0, 255, 0), thickness=2)
                    center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=15)
                    axes = (radius, radius)
                    draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)

                except:
                    pass

                try:
                    # knee angle for right knee
                    l1 = np.linspace(idx_to_coordinates[24], idx_to_coordinates[26], 100)
                    l2 = np.linspace(idx_to_coordinates[26], idx_to_coordinates[28], 100)
                    cv2.line(image, (int(l1[99][0]), int(l1[99][1])), (int(l1[59][0]), int(l1[59][1])), thickness=6,
                             color=(0, 0, 255))
                    cv2.line(image, (int(l2[0][0]), int(l2[0][1])), (int(l2[40][0]), int(l2[40][1])), thickness=6,
                             color=(0, 0, 255))
                    ang2 = ang((idx_to_coordinates[24], idx_to_coordinates[26]),
                               (idx_to_coordinates[26], idx_to_coordinates[28]))
                    cv2.putText(image, str(round(ang2, 2)), (idx_to_coordinates[26][0] + 20, idx_to_coordinates[26][1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(0, 255, 0), thickness=2)
                    center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=15)
                    axes = (radius, radius)
                    draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)

                except:
                    pass

                try:
                    # wrist elbow shoulder
                    l1 = np.linspace(idx_to_coordinates[15], idx_to_coordinates[13], 100)
                    l2 = np.linspace(idx_to_coordinates[13], idx_to_coordinates[11], 100)
                    cv2.line(image, (int(l1[99][0]), int(l1[99][1])), (int(l1[59][0]), int(l1[59][1])), thickness=6,
                             color=(0, 0, 255))
                    cv2.line(image, (int(l2[0][0]), int(l2[0][1])), (int(l2[40][0]), int(l2[40][1])), thickness=6,
                             color=(0, 0, 255))
                    ang2 = ang((idx_to_coordinates[15], idx_to_coordinates[13]),
                               (idx_to_coordinates[13], idx_to_coordinates[11]))
                    cv2.putText(image, str(round(ang2, 2)), (idx_to_coordinates[13][0] + 10, idx_to_coordinates[13][1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(0, 255, 0), thickness=2)
                    center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=15)
                    axes = (radius, radius)
                    draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)

                except:
                    pass

                try:
                    # shoulder back knee
                    l1 = np.linspace(idx_to_coordinates[11], idx_to_coordinates[23], 100)
                    l2 = np.linspace(idx_to_coordinates[23], idx_to_coordinates[25], 100)
                    cv2.line(image, (int(l1[99][0]), int(l1[99][1])), (int(l1[75][0]), int(l1[75][1])), thickness=6,
                             color=(0, 0, 255))
                    cv2.line(image, (int(l2[0][0]), int(l2[0][1])), (int(l2[30][0]), int(l2[30][1])), thickness=6,
                             color=(0, 0, 255))
                    ang2 = ang((idx_to_coordinates[11], idx_to_coordinates[23]),
                               (idx_to_coordinates[23], idx_to_coordinates[25]))
                    cv2.putText(image, str(round(ang2, 2)), (idx_to_coordinates[23][0] + 10, idx_to_coordinates[23][1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, color=(0, 255, 0), thickness=2)
                    center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=15)
                    axes = (radius, radius)
                    draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)

                except:
                    pass

                try:
                    # Distance between the left foot and right hand
                    left_foot = idx_to_coordinates[29]
                    left_hand = idx_to_coordinates[17]
                    cv2.line(image, (left_foot[0], left_foot[1] + 50), (left_hand[0], left_foot[1] + 50), thickness=2,
                             color=(255, 255, 0))
                    cv2.line(image, (left_foot[0], left_foot[1] + 50), (left_foot[0], left_foot[1] + 60), thickness=2,
                             color=(255, 255, 0))
                    cv2.line(image, (left_hand[0], left_foot[1] + 50), (left_hand[0], left_foot[1] + 60), thickness=2,
                             color=(255, 255, 0))
                    cv2.putText(image, str(max(0, left_foot[0] - left_hand[0])) + " px",
                                (left_foot[0] - 100, left_foot[1] + 80),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.9, color=(0, 255, 0), thickness=2)
                except:
                    pass

                try:
                    # Distance between the left foot and right foot
                    left_foot = idx_to_coordinates[29]
                    right_foot = idx_to_coordinates[30]
                    cv2.line(image, (right_foot[0], right_foot[1] + 50), (left_foot[0], right_foot[1] + 50), thickness=2,
                             color=(255, 51, 51))
                    cv2.line(image, (right_foot[0], right_foot[1] + 50), (right_foot[0], right_foot[1] + 60), thickness=2,
                             color=(255, 51, 51))
                    cv2.line(image, (left_foot[0], right_foot[1] + 50), (left_foot[0], right_foot[1] + 60), thickness=2,
                             color=(255, 51, 51))
                    cv2.putText(image, str(max(0, right_foot[0] - left_foot[0])) + " px",
                                (right_foot[0] - 100, right_foot[1] + 80),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.9, color=(0, 0, 0), thickness=2)
                except:
                    pass

                if 0 in idx_to_coordinates:
                    cv2.putText(image, "Batsman : Right Handed",
                                (idx_to_coordinates[0][0] - 120, idx_to_coordinates[0][1] - 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.9, color=(0, 255, 0), thickness=2)

                cv2.imshow('Cricket', rescale_frame(image, percent=180))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                key = cv2.waitKey(1)
                if key == ord('p'):
                    cv2.waitKey(-1)  # wait until any key is pressed
        self.pose.close()
