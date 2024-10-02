import mmcv
import matplotlib.pyplot as plt
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import mmdet
from mmpose.apis.visualization import visualize
import time
import numpy as np
import cv2

from mmpose.apis import MMPoseInferencer
from pathlib import Path
# from mmpose.visualization.fast_visualizer import draw_pose
# from mmdet.apis import show_result_pyplot
image = mmcv.imread('./InputsJPGs/group17.jpg')
register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# please prepare an image with person

results = inference_topdown(model, image)

keypoints = results[0].pred_instances.keypoints
pred = results[0].pred_instances.keypoint_scores


res = visualize(image,keypoints=keypoints,keypoint_score=pred,show=False)


res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
output_path = 'result_with_skeleton.jpg'

cv2.imwrite(output_path, res)
# image_with_keypoints = model.show_result(image, results, kpt_score_thr=0.3, show=False)
heatmap = results[0]


