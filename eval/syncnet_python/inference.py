#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, os, argparse, glob, subprocess, cv2
import numpy as np
from shutil import rmtree

# ==================== 1. Compatibility patch ====================
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool

sys.path.append(os.getcwd())

# ==================== 2. Imports ====================
try:
    import scenedetect
    from scenedetect.video_manager import VideoManager
    from scenedetect.scene_manager import SceneManager
    from scenedetect.stats_manager import StatsManager
    from scenedetect.detectors import ContentDetector
except ImportError:
    print("Error: scenedetect not found. Please run: pip install scenedetect==0.5.6.1")
    sys.exit(1)

from scipy.interpolate import interp1d
from scipy import signal
from detectors import S3FD
from SyncNetInstance import SyncNetInstance

# ==================== 3. Argument configuration ====================
parser = argparse.ArgumentParser(description = "SyncNet End-to-End Tool")
parser.add_argument('--videofile',      type=str, required=True, help='Input video file path')
parser.add_argument('--data_dir',       type=str, default='data/work', help='Output directory')
parser.add_argument('--reference',      type=str, default='',   help='Name of the video')
parser.add_argument('--batch_size',     type=int, default=20,   help='SyncNet batch size')
parser.add_argument('--vshift',         type=int, default=15,   help='SyncNet video shift range')
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate')
parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Face detection scale')
parser.add_argument('--crop_scale',     type=float, default=0.40, help='Bounding box scale')
parser.add_argument('--min_track',      type=int, default=30,   help='Min track duration')
parser.add_argument('--num_failed_det', type=int, default=25,   help='Missed detections allowed')
parser.add_argument('--min_face_size',  type=int, default=100,  help='Min face size')

opt = parser.parse_args()

if opt.reference == '':
    opt.reference = os.path.splitext(os.path.basename(opt.videofile))[0]

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))

# ==================== 4. Core functions ====================

def bb_intersection_over_union(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def track_shot(opt, scenefaces):
    iouThres, tracks = 0.5, []
    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if track == []:
                    track.append(face)
                    framefaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
                    if bb_intersection_over_union(face['bbox'], track[-1]['bbox']) > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else: break
        if track == []: break
        elif len(track) > opt.min_track:
            framenum = np.array([ f['frame'] for f in track ])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            frame_i = np.arange(framenum[0],framenum[-1]+1)
            bboxes_i = []
            for ij in range(0,4):
                interpfn = interp1d(framenum, bboxes[:,ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
                tracks.append({'frame':frame_i,'bbox':bboxes_i})
    return tracks

def crop_video(opt, track, cropfile):
    flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
    flist.sort()
    vOut = cv2.VideoWriter(cropfile+'t.avi', cv2.VideoWriter_fourcc(*'XVID'), opt.frame_rate, (224,224))
    dets = {'x':[], 'y':[], 's':[]}
    for det in track['bbox']:
        dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) 
        dets['x'].append((det[0]+det[2])/2) 
    dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
    dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'],kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs, bs = opt.crop_scale, dets['s'][fidx]
        bsi = int(bs*(1+2*cs))
        image = cv2.imread(flist[frame])
        frame_img = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
        my, mx = dets['y'][fidx]+bsi, dets['x'][fidx]+bsi
        face = frame_img[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face,(224,224)))
    audiotmp = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
    audiostart, audioend = (track['frame'][0])/opt.frame_rate, (track['frame'][-1]+1)/opt.frame_rate
    vOut.release()
    subprocess.call(f"ffmpeg -y -i \"{os.path.join(opt.avi_dir,opt.reference,'audio.wav')}\" -ss {audiostart:.3f} -to {audioend:.3f} \"{audiotmp}\"", shell=True, stdout=None, stderr=subprocess.DEVNULL)
    subprocess.call(f"ffmpeg -y -i \"{cropfile}t.avi\" -i \"{audiotmp}\" -c:v copy -c:a copy \"{cropfile}.avi\"", shell=True, stdout=None, stderr=subprocess.DEVNULL)
    os.remove(cropfile+'t.avi')
    return {'track':track, 'proc_track':dets}

def inference_video(opt):
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
    flist.sort()
    dets = []
    print(f"Running face detection over {len(flist)} frames...")
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
        if fidx % 50 == 0: print(f"  Processed {fidx} / {len(flist)} frames")
    return dets

def scene_detect(opt):
    video_manager = VideoManager([os.path.join(opt.avi_dir,opt.reference,'video.avi')])
    scene_manager = SceneManager(StatsManager())
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    return scene_list if scene_list else [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

# ==================== 5. Main ====================

def main():
    if not os.path.exists("data/syncnet_v2.model"):
        print("Error: model file not found: data/syncnet_v2.model")
        return

    # Reset output directories.
    for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
        full_d = os.path.join(d, opt.reference)
        if os.path.exists(full_d): rmtree(full_d)
        os.makedirs(full_d)

    print(f"Start processing video: {opt.videofile}")

    # Video preprocessing.
    cmd_base = f"ffmpeg -y -i \"{opt.videofile}\" -qscale:v 2"
    subprocess.call(f"{cmd_base} -async 1 -r 25 \"{os.path.join(opt.avi_dir,opt.reference,'video.avi')}\"", shell=True, stderr=subprocess.DEVNULL)
    subprocess.call(f"ffmpeg -y -i \"{os.path.join(opt.avi_dir,opt.reference,'video.avi')}\" -qscale:v 2 -threads 1 -f image2 \"{os.path.join(opt.frames_dir,opt.reference,'%06d.jpg')}\"", shell=True, stderr=subprocess.DEVNULL)
    subprocess.call(f"ffmpeg -y -i \"{os.path.join(opt.avi_dir,opt.reference,'video.avi')}\" -ac 1 -vn -acodec pcm_s16le -ar 16000 \"{os.path.join(opt.avi_dir,opt.reference,'audio.wav')}\"", shell=True, stderr=subprocess.DEVNULL)

    # Detection and cropping.
    faces = inference_video(opt)
    scene = scene_detect(opt)
    alltracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= opt.min_track :
            alltracks.extend(track_shot(opt,faces[shot[0].frame_num:shot[1].frame_num]))
            
    if not alltracks:
        print("Error: no valid face tracks detected (no faces, faces too small, or video too short).")
        return

    print(f"Detected {len(alltracks)} face tracks, starting crop...")
    for ii, track in enumerate(alltracks):
        crop_video(opt,track,os.path.join(opt.crop_dir,opt.reference,'%05d'%ii))

    # SyncNet inference.
    print("Loading SyncNet model...")
    s = SyncNetInstance()
    s.loadParameters("data/syncnet_v2.model")
    crop_files = sorted(glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi')))
    
    print("\n" + "="*20 + " Final Results " + "="*20)

    for idx, fname in enumerate(crop_files):
        offset, conf, minval = s.evaluate(opt, videofile=fname)
        
        print(f"Track {idx}:")
        print(f"  AV offset:  {offset} frames")
        print(f"  Min dist:   {float(minval):.3f}")  
        print(f"  Confidence: {float(conf):.3f}")
        
        print("-" * 30)

if __name__ == '__main__':
    main()
