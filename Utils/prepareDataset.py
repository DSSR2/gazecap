import math, shutil, os, time, argparse, json, re, sys
import numpy as np
import scipy.io as sio
from PIL import Image
import shutil
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, Process

parser = argparse.ArgumentParser(description='Gazecap dataset preparation')
parser.add_argument('--dataset_path', help="Path to extracted files. It should have folders called '%%05d' in it.")
parser.add_argument('--output_path', default=None, help="Where to write the output. Can be the same as dataset_path if you wish (=default).")
parser.add_argument('--threads', default=1, type=int, help="Number of threads to process dataset")

def convert_dataset(files, out_root):
    for i in tqdm(files): 
        with open(i+"/info.json") as f:
            data = json.load(f)
            ds = data['Dataset']
            device = data['DeviceName']
        if(not('iPhone' in device)):
            continue
        
        expt_name = i.split('/')[-2]
        screen_info = json.load(open(i+'/screen.json'))
        face_det = json.load(open(i+'/appleFace.json'))
        l_eye_det = json.load(open(i+'/appleLeftEye.json'))
        r_eye_det = json.load(open(i+'/appleRightEye.json'))
        dot = json.load(open(i+'/dotInfo.json'))
        faceGrid = json.load(open(i+'/faceGrid.json'))
    
        l_eye_valid, r_eye_valid = np.array(l_eye_det['IsValid']), np.array(r_eye_det['IsValid'])
        face_valid = np.array(face_det['IsValid']) 
        face_grid_valid = np.array(faceGrid['IsValid'])
        valid_ids = l_eye_valid*r_eye_valid*face_valid*face_grid_valid
        
        frame_ids = np.where(valid_ids==1)[0]
        
        for frame_idx in frame_ids:
            fname = str(frame_idx).zfill(5)
            outdir = out_root+"/"+ds
            
            meta = {}
            meta['device'] = device
            
            meta['orientation'] = screen_info["Orientation"][frame_idx]
            
            meta['screen_h'], meta['screen_w'] = screen_info["H"][frame_idx], screen_info["W"][frame_idx]
            
            meta['face_valid'] = face_det["IsValid"][frame_idx]
            meta['face_x'], meta['face_y'], meta['face_w'], meta['face_h'] = round(face_det['X'][frame_idx]), round(face_det['Y'][frame_idx]), round(face_det['W'][frame_idx]), round(face_det['H'][frame_idx])
            meta['face_grid'] = [faceGrid["X"][frame_idx], faceGrid["Y"][frame_idx], faceGrid["W"][frame_idx], faceGrid["H"][frame_idx]]
            
            meta['leye_x'], meta['leye_y'], meta['leye_w'], meta['leye_h'] = meta['face_x']+round(l_eye_det['X'][frame_idx]), meta['face_y']+round(l_eye_det['Y'][frame_idx]), round(l_eye_det['W'][frame_idx]), round(l_eye_det['H'][frame_idx])
            meta['reye_x'], meta['reye_y'], meta['reye_w'], meta['reye_h'] = meta['face_x']+round(r_eye_det['X'][frame_idx]), meta['face_y']+round(r_eye_det['Y'][frame_idx]), round(r_eye_det['W'][frame_idx]), round(r_eye_det['H'][frame_idx])
            meta['dot_xcam'], meta['dot_y_cam'] = dot['XCam'][frame_idx], dot['YCam'][frame_idx]
            meta['dot_x_pix'], meta['dot_y_pix'] = dot['XPts'][frame_idx], dot['YPts'][frame_idx]
            

            shutil.copy(i+'/frames/'+fname+".jpg", outdir+"/images/"+expt_name+'__'+fname+'.jpg')
            
            meta_file = outdir+'/meta/'+expt_name+'__'+fname+'.json'
            with open(meta_file, 'w') as outfile:
                json.dump(meta, outfile)

def preparePath(path, clear=True):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)
    os.mkdir(path+"/train/")
    os.mkdir(path+"/val/")
    os.mkdir(path+"/test/")
    os.mkdir(path+"/train/images/")
    os.mkdir(path+"/train/meta/")
    os.mkdir(path+"/val/images/")
    os.mkdir(path+"/val/meta/")
    os.mkdir(path+"/test/images")
    os.mkdir(path+"/test/meta")
    return path

def main():
    args = parser.parse_args()
    threads = args.threads
    preparePath(args.output_path)
    procs = []
    files = glob(args.dataset_path+"/*/")
    chunk = len(files)//threads
    print(len(files))
    for i in range(threads): 
        f = files[i*chunk:(i+1)*chunk]
        if(i==threads-1):
            f = files[i*chunk:]
        
        proc = Process(target=convert_dataset, args=(f, args.output_path))
        procs.append(proc)
        proc.start()
        print(i)
        
    for proc in procs:
        proc.join()
        
    output_path = args.output_path
    print("DONE")
    print("Total frames: ", len(glob(output_path+"/*/images/*.jpg")), len(glob(output_path+"/*/meta/*.json")))
    print("Train frames: ", len(glob(output_path+"/train/images/*.jpg")), len(glob(output_path+"/train/meta/*.json")))
    print("Test frames: ", len(glob(output_path+"/test/images/*.jpg")), len(glob(output_path+"/test/meta/*.json")))
    print("Val frames: ", len(glob(output_path+"/val/images/*.jpg")), len(glob(output_path+"/val/meta/*.json")))
    return 0

if __name__ == '__main__':
    main()