import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh,scale_coords
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
from conditions import *
from tracker.mc_bot_sort import BoTSORT

def fall_recognition(feature):
  feature_tensor = torch.Tensor(feature)
  # print("feature tensor's shape: ", feature_tensor.shape)
  bbox = feature_tensor[:,:4]
  kpt = feature_tensor[:,6:]
  # print("shape of keypoint: {}".format(kpt.shape))
  new_kpt = torch.zeros((kpt.shape[0], 17, 3))
  for i in range(kpt.shape[0]):
    # print("shape of kpt i: ", kpt[i].shape)
    new_kpt[i] = torch.reshape(kpt[i], (17,3))
  # print("shape of keypoint after: {}".format(new_kpt.shape))
  
  action = "Normal"
  color = (0, 255, 0)
  if Condition_three(bbox[-1]):
    action = "Fall"
    print(action)
    color = (0, 0, 255)
    return action, color
  elif Condition_one(new_kpt, threshold = 0.0002):
    if Condition_two(new_kpt):
      if Condition_three(bbox[-1]):
        action = "Fall"
        print(action)
        color = (0, 0, 255)
  return action, color

def main(opt):
    source, poseweights = opt.source, opt.poseweights
    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps

    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(poseweights, map_location=device) 

    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)

    cap = cv2.VideoCapture(source)  
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()
    else:
        frame_width = int(cap.get(3))  #get video frame width
        frame_height = int(cap.get(4)) #get video frame height

        
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"result/{out_video_name}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))

        while(cap.isOpened): #loop until cap opened or video not complete
            # frame_count +=1
            print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  #get frame and success from video capture

            if not ret: 
                break
            else: #if success is true, means frame exist
                orig_image = frame #store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  #convert image data to device
                image = image.float() #convert image to float precision (cpu)
                start_time = time.time() #start time for fps calculation
                # Inference
                with torch.no_grad():
                    output_data = model(image)[0]

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                                    0.1,   # Conf. Threshold.
                                                    0.65, # IoU Threshold.
                                                    nc=model.yaml['nc'], # Number of classes.
                                                    nkpt=model.yaml['nkpt'], # Number of keypoints.
                                                    kpt_label=True)
                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                # Process detections
                results = []
                # print("shape of output: {}".format(output_data[0][0:,6:]))
                for i, det in enumerate(output_data):  # detections per image
                    # Run tracker
                    detections = []
                    if len(det):
                        # print("shape of frame: ", frame.shape)
                        # boxes = scale_coords(frame.shape[:2], det[:, :4], im0.shape)
                        # boxes = boxes.cpu().numpy()
                        detections = det.cpu().numpy()
                        # detections[:, :4] = boxes

                    online_targets = tracker.update(detections, im0)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    online_cls = []
                    for t in online_targets:
                          tlwh = t.tlwh
                          tlbr = t.tlbr
                          tid = t.track_id
                          tcls = t.cls
                          kpt_lst = t.features
                          # print("tlbr: {}".format(tlbr))
                          # print("tid: {}".format(tid))
                          # print("length of kpt list: {}".format(len(kpt_lst)))
                          if len(kpt_lst) > 0 and len(kpt_lst) % 5 == 0:
                            # print("shape of kpt list: {}".format(kpt_lst[0].shape))
                        # print("type of object: {}".format(type(t)))
                            if tlwh[2] * tlwh[3] > opt.min_box_area:
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(t.score)
                                online_cls.append(t.cls)
                                action, color = fall_recognition(kpt_lst)
                                plot_one_box(tlbr, im0, label=action, color = color, line_thickness=2)
                
                end_time = time.time()  #Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps) #append FPS in list
                time_list.append(end_time - start_time) #append time in list
                out.write(im0)
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")      
    cap.release()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default="football1.mp4", help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()
    opt.jde = False
    opt.ablation = False
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)