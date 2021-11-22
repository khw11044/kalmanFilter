import os
import argparse

import cv2
import numpy as np
from glob import glob


parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--video_name', default='saved', type=str,
                    help='videos or image files')
args = parser.parse_args()
root=args.video_name


video_frame = []

def get_frames(video_name):
    global width,height
    images = glob(os.path.join(video_name, '*.*'))
    images = sorted(images, key=lambda x: x.split('/')[-1].split('.')[0])
        # key=lambda x: int(x.split('/')[-1].split('.')[0]))
    height,width,_ =cv2.imread(images[0]).shape
    for img in images:
        frame = cv2.imread(img)
        yield frame

def main():
    name = root.split('/')[-1].split('.')[0]
    for fram in get_frames(root):
        cv2.imshow(name, fram)
        cv2.waitKey(40)
        video_frame.append(fram)

if __name__=="__main__":
    
    main()
    Output = 'demo.mp4'
    fps=10
    out = cv2.VideoWriter(Output, cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))
    for i in range(len(video_frame)):
        out.write(video_frame[i])
    out.release()
    print('done')
    print(Output)