import os
import sys
import cv2
import numpy as np
import time
import gaitevents
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import traceback

def process(walk_direction: gaitevents.Side, raw_video_file: str, save_dir: str = ".", patientinfo = dict(), rotate_cw=False):

    taskPath = os.path.join(os.environ.get('MODEL_DIR','.'),'pose_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=taskPath)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options)
    detector = vision.PoseLandmarker.create_from_options(options)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    video_capture = cv2.VideoCapture(raw_video_file)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Video info:")
    print("FPS: "+str(fps)+" Width: "+str(video_width)+" Height: "+str(video_height))
    if fps < 1:
        raise ValueError(f"Bad video file: {raw_video_file}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video_file = os.path.splitext(os.path.basename(raw_video_file))[0] + "_mp.avi"
    video_writer = cv2.VideoWriter(save_dir+"/"+out_video_file, fourcc, round(fps), (int(video_width), int(video_height)))

    time_start = time.time()
    frame_no = 0

    landmark_lists = dict()
    world_landmark_lists = dict()
    for landmark in mp_pose.PoseLandmark:
        landmark_lists[landmark] = []
        world_landmark_lists[landmark] = []
    poseScoreList = []

    # custom annotation style
    pose_connection_style = mp_drawing.DrawingSpec(thickness=4)
    pose_landmark_style = mp_drawing_styles.get_default_pose_landmarks_style()
    for landmark in pose_landmark_style.keys():
      pose_landmark_style[landmark].circle_radius = 4

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            if frame_no==0:
                print("Can't receive frame (stream end?). Exiting ...")
            else:
                print(frameAt+"Last frame processed")
            break
        elif rotate_cw:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        frameAt = f"Frame {frame_no}|"
        frame_height, frame_width, _ = frame.shape
        # Convert the BGR image to RGB before processing.
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = detector.detect(rgb_frame)
        # results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print(frameAt+"not detected, using previous frame's data...")
            for landmark in mp_pose.PoseLandmark:
                if np.shape(landmark_lists[landmark])[0] == 0:
                    landmark_lists[landmark].append([np.nan, np.nan, np.nan])
                    world_landmark_lists[landmark].append([np.nan, np.nan, np.nan])
                    poseScoreList.append(0.0)
                else:
                    landmark_lists[landmark].append(landmark_lists[landmark][-1])
                    world_landmark_lists[landmark].append(world_landmark_lists[landmark][-1])
                    poseScoreList.append(poseScoreList[-1])
        else:
            minPoseScore = 1.0
            for landmark in mp_pose.PoseLandmark:
                x = results.pose_landmarks[0][landmark].x * frame_width
                y = results.pose_landmarks[0][landmark].y * frame_height
                z = results.pose_landmarks[0][landmark].z * frame_width
                if minPoseScore > results.pose_landmarks[0][landmark].presence:
                   minPoseScore = results.pose_landmarks[0][landmark].presence
                # x = results.pose_landmarks.landmark[landmark].x * frame_width
                # y = results.pose_landmarks.landmark[landmark].y * frame_height
                # z = results.pose_landmarks.landmark[
                #         landmark].z * frame_width  # stated on mediapipe website that z follows roughly same scale as x
                y = -y + frame_height  # flip and translate y values
                z = -z # change to right-handed axes convention
                # if landmark == mp_pose.PoseLandmark.LEFT_KNEE:
                #     print(results.pose_landmarks.landmark[landmark])
                landmark_lists[landmark].append([x, y, z])

                x = results.pose_world_landmarks[0][landmark].x
                y = -results.pose_world_landmarks[0][landmark].y
                z = results.pose_world_landmarks[0][landmark].z
                z = -z  # change to right-handed axes convention
                world_landmark_lists[landmark].append([x, y, z])
                # if landmark == mp_pose.PoseLandmark.LEFT_KNEE:
                #     print(results.pose_world_landmarks.landmark[landmark])
            poseScoreList.append(minPoseScore)

        annotated_image = frame.copy()
        # Draw pose landmarks on the image.
        if results.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                 landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in results.pose_landmarks[0]
                ])
            # landmark_count = len(pose_landmarks_proto.landmark)
            # print(f"DEBUG: landmark count {landmark_count}")
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks_proto,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)
            # for landmark in mp_pose.PoseLandmark:
            #     landmarkPoint = (int(landmark_lists[landmark][-1][0]), int(frame_height - landmark_lists[landmark][-1][1]))
            #     # print(f"(x,y): {landmarkPoint}")
            #     cv2.putText(annotated_image, str(landmark.value), landmarkPoint, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        #mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        #cv2.imshow("Mediapipe - " + raw_video_file, annotated_image)
        video_writer.write(annotated_image)
        frame_no = frame_no +1

    cv2.destroyAllWindows()

    print("Mediapipe processing complete. Time taken: " + str(
        time.time() - time_start) + "s. Average frames processed per second: " + str(
        video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / (time.time() - time_start)) + ".")

    video_capture.release()
    video_writer.release()

    gaitdata = gaitevents.GaitData("mediapipe", fps)
    gaitdata.walk_direction = walk_direction
    gaitdata.patientinfo = patientinfo

    gaitdata.data[gaitevents.Joint.LEFT_HEEL] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_HEEL])[:, 0:3]
    gaitdata.data[gaitevents.Joint.RIGHT_HEEL] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_HEEL])[:, 0:3]
    gaitdata.data[gaitevents.Joint.LEFT_TOE_BIG] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])[:, 0:3]
    gaitdata.data[gaitevents.Joint.RIGHT_TOE_BIG] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])[:, 0:3]
    gaitdata.data[gaitevents.Joint.LEFT_ANKLE] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_ANKLE])[:, 0:3]
    gaitdata.data[gaitevents.Joint.RIGHT_ANKLE] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_ANKLE])[:,0:3]
    gaitdata.data[gaitevents.Joint.LEFT_KNEE] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_KNEE])[:, 0:3]
    gaitdata.data[gaitevents.Joint.RIGHT_KNEE] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_KNEE])[:, 0:3]
    gaitdata.data[gaitevents.Joint.LEFT_HIP] = np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3]
    gaitdata.data[gaitevents.Joint.RIGHT_HIP] = np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_HIP])[:, 0:3]

    midhip = 0.5 * (np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3] + np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_HIP])[:, 0:3])
    gaitdata.data[gaitevents.Joint.MIDDLE_HIP] = midhip

    mid_shoulder = 0.5 * (np.array(landmark_lists[mp_pose.PoseLandmark.LEFT_SHOULDER])[:, 0:3] + np.array(landmark_lists[mp_pose.PoseLandmark.RIGHT_SHOULDER])[:, 0:3])
    gaitdata.data[gaitevents.Joint.MID_SHOULDER] = mid_shoulder

    gaitdata.data[gaitevents.Joint.LEFT_TOES] = gaitdata.data[gaitevents.Joint.LEFT_TOE_BIG]
    gaitdata.data[gaitevents.Joint.RIGHT_TOES] = gaitdata.data[gaitevents.Joint.RIGHT_TOE_BIG]

    gaitdata.data_world[gaitevents.Joint.LEFT_HEEL] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_HEEL])[:, 0:3]
    gaitdata.data_world[gaitevents.Joint.RIGHT_HEEL] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_HEEL])[:, 0:3]
    gaitdata.data_world[gaitevents.Joint.LEFT_TOE_BIG] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])[:, 0:3]
    gaitdata.data_world[gaitevents.Joint.RIGHT_TOE_BIG] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])[:, 0:3]
    gaitdata.data_world[gaitevents.Joint.LEFT_ANKLE] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_ANKLE])[:, 0:3]
    gaitdata.data_world[gaitevents.Joint.RIGHT_ANKLE] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_ANKLE])[:,0:3]
    gaitdata.data_world[gaitevents.Joint.LEFT_KNEE] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_KNEE])[:, 0:3]
    gaitdata.data_world[gaitevents.Joint.RIGHT_KNEE] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_KNEE])[:, 0:3]
    gaitdata.data_world[gaitevents.Joint.LEFT_HIP] = np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3]
    gaitdata.data_world[gaitevents.Joint.RIGHT_HIP] = np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_HIP])[:, 0:3]

    midhip = 0.5 * (np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_HIP])[:, 0:3] + np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_HIP])[:, 0:3])
    gaitdata.data_world[gaitevents.Joint.MIDDLE_HIP] = midhip

    mid_shoulder = 0.5 * (np.array(world_landmark_lists[mp_pose.PoseLandmark.LEFT_SHOULDER])[:, 0:3] + np.array(world_landmark_lists[mp_pose.PoseLandmark.RIGHT_SHOULDER])[:, 0:3])
    gaitdata.data_world[gaitevents.Joint.MID_SHOULDER] = mid_shoulder

    gaitdata.data_world[gaitevents.Joint.LEFT_TOES] = gaitdata.data_world[gaitevents.Joint.LEFT_TOE_BIG]
    gaitdata.data_world[gaitevents.Joint.RIGHT_TOES] = gaitdata.data_world[gaitevents.Joint.RIGHT_TOE_BIG]

    # JSLOW(20231018): add a pose score for each frame
    gaitdata.poseScore = poseScoreList
    
    save_filename = os.path.splitext(os.path.basename(raw_video_file))[0] + "_mp.pkl"
    pklkPath = os.path.join(save_dir,save_filename)
    gaitevents.save_object(gaitdata, pklkPath)
    print("Data saved to "+pklkPath)

      
if __name__ == '__main__':
  if len(sys.argv) < 6:
      print(f"Usage: {sys.argv[0]} walk_dir raw_video_file save_dir left_leg_length(mm) right_leg_length(mm)")
      sys.exit(1)

  walk_dir = sys.argv[1]
  raw_video_file = sys.argv[2]
  save_dir = sys.argv[3]
  leftLegLength = int(sys.argv[4])*0.001
  rightLegLength = int(sys.argv[5])*0.001

  patientinfo = {
    'leg_length_left': leftLegLength,
    'leg_length_right': rightLegLength
  }
  print(f"Processing {walk_dir} video: {raw_video_file} output to: {save_dir}")
  print(f"Leg Length (left, right) in m : ({patientinfo['leg_length_left']:.3f}, {patientinfo['leg_length_right']:.3f})")

  try:
    if walk_dir == "Side.RIGHT":
      process(gaitevents.Side.RIGHT, raw_video_file, save_dir, patientinfo)
    elif walk_dir == "Side.LEFT":
      process(gaitevents.Side.LEFT, raw_video_file, save_dir, patientinfo)
    else:
      raise ValueError(f"Unknown walk_dir: {walk_dir}.  Accept Side.RIGHT or Side.LEFT")
  except Exception as e:
    print(e)
    traceback.print_exc()
    raise