from ultralytics import YOLO
import cv2
import numpy as np
import os



# video_path = "./video/Kamera1.dav "
video_path = "./InputVideos/3255107-uhd_3840_2160_25fps.mp4"
# video_path = "./InputVideos/855565-hd_1920_1080_24fps.mp4"
video_capture = cv2.VideoCapture(video_path)

#get video size
w, h, = (

    int(video_capture.get(x))

    for x in (

        cv2.CAP_PROP_FRAME_WIDTH,

        cv2.CAP_PROP_FRAME_HEIGHT,

    )

)
#create result file
output_folder = "./ResultsVideoYOLO"
res_path = os.path.basename(video_path)
result_video = cv2.VideoWriter(os.path.join(output_folder, res_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)) #fps, (2 * w, 2 * h) mp4v MJPG

 



if not video_capture.isOpened():

    raise ValueError(f"Could not open video file: {video_path}")

#Do you want to skip frames?
skip = False

#skip frames at start where there is no people to detect (fps * time to skip[s])
if skip:
    skip_frames = 200

    for _ in range(skip_frames):

        ret, frame = video_capture.read()


inferencer = YOLO("yolo11n-pose.pt")

ret = True

# licz = 0

while ret:
    ret, frame = video_capture.read()
 
    if not ret:
         break


    
    results = inferencer(frame)
    
    # result_generator = inferencer(frame, show=False,return_vis = True)  # turn off showing the result for speed
    # result_frame = next(result_generator)
    
    result = results[0]
    keypoints = result.keypoints
    nb_detected = keypoints.shape[0]
    
    res = result.plot()
    
    # res = np.array(res[0])

    
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    #show counter of detected people
    text = f'Human detected: {nb_detected}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0) 
    thickness = 2
    position = (50, 50)  

    
    cv2.putText(res, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    result_video.write(res)
    # licz += 1

video_capture.release()
result.release()