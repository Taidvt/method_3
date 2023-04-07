from models import YOLOv7_pose
import torch
import numpy as np
import cv2
import time
import argparse
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
from cameraloader import CamLoader_Q

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def pre_proc(orig_image, frame_width, device = "cpu"):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
    image = letterbox(image, (frame_width), stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    image = image.to(device)  #convert image data to device
    image = image.float() #convert image to float precision (cpu)

def postproc(model, output_data):
    output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
    output = output_to_keypoint(output_data)
    return output

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source = r"C:\Users\taidam\Downloads\testset\6_ S_ DN_8.mp4", device = "cpu", view_img = False, save_conf = False,
        line_thinkness = 3, hide_labels = False, hide_conf = True):
    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps

    # Detection model
    detection_model = YOLOv7_pose(device)
    names = detection_model.module.names if hasattr(detection_model, 'module') else detection_model.names  # get class names

    cam = CamLoader_Q(source, queue_size=1000, preprocess=preproc).start()

   
    if (cam.grabbed() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width, frame_height = cam.getsize()  #get video frame width
        vid_write_image = letterbox(cam.get_fimg(), (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))

        while(cam.grabbed()): #loop until cap opened or video not complete
        
            print("Frame {} Processing".format(frame_count+1))

            frame = cam.getitem()  #get frame and success from video capture

            orig_image = frame #store frame
            image = pre_proc(orig_image, frame_width, device)
            start_time = time.time() #start time for fps calculation    

            output_data = detection_model(image)

            output = postproc(detection_model, output_data)
            print("output shape: {}".format(output.shape))
            im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
            im0 = im0.cpu().numpy().astype(np.uint8)
            
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for i, pose in enumerate(output_data):  # detections per image
            
                if len(output_data):  #check if no pose
                    for c in pose[:, 5].unique(): # Print results
                        n = (pose[:, 5] == c).sum()  # detections per class
                        print("No of Objects in Current Frame : {}".format(n))
                    
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                        c = int(cls)  # integer class
                        kpts = pose[det_index, 6:]
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                    line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                                    orig_shape=im0.shape[:2])

            
            end_time = time.time()  #Calculatio for FPS
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            
            fps_list.append(total_fps) #append FPS in list
            time_list.append(end_time - start_time) #append time in list
            
            # Stream results
            if view_img:
                cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                cv2.waitKey(1)  # 1 millisecond

            out.write(im0)  #writing the video frame

        cam.stop()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view_img', action='store_true', help='display results')  #display results
    parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    # parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    # parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels') #box hidelabel
    # parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    # strip_optimizer(opt.device,opt.poseweights)
    main(opt)
    