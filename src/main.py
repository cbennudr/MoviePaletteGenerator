import cv2
import numpy as np
import os
import argparse
from math import ceil
from tqdm import tqdm
import multiprocessing as mp

from utils import *


def stitch_colors(colors:list, width:int, height:int) -> np.array:
    """Function to go through a list of colors and stitch strips of each color into an image"""
    
    # Adjust size to fit frames correctly
    num_colors = len(colors)
    width_of_each_strip = 1#ceil(width/num_colors)

    # Loop through the colors, create a strip for each, then combine all the strips into the final image
    strips = []
    with tqdm(total=len(colors), desc="Creating palette image") as pbar:
        for color in colors:
            cur_strip = np.full((height,width_of_each_strip,3), color)
            strips.append(cur_strip)
            pbar.update(1)

    print("Combining strips")
    palette = np.hstack(strips)

    return palette


def create_palette_img(video_dir:str, palette_width:int, palette_height:int, display_frames:bool, color_selection_method:str, use_every_n_frames:int):
    """Function to create a palette image from frames in a video"""

    # Open the video
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened(): 
        raise Exception("Unable to open VideoCapture")

    # Calculate frames to skip to fit set width 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (use_every_n_frames is not None):    every_n_frames = use_every_n_frames
    elif (palette_width >= total_frames):   every_n_frames = 0
    else:                                   every_n_frames = int(total_frames/palette_width)
    print("Using every frame") if every_n_frames==0 else print(f"Using every {every_n_frames} frames")
    
    # Get dominant (or average) color from each frame
    colors = get_colors_from_frames(cap, every_n_frames, color_selection_method=color_selection_method, display_frames=display_frames)

    # Create the palette image from the colors array
    palette_img = stitch_colors(colors, palette_width, palette_height)
    return palette_img
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to create a palette image using the average (or dominant) color from frames in a video")
    ######################### Required args
    parser.add_argument('--video', '-v', required=True)
    parser.add_argument('--output_dir', '-o', required=True)
    parser.add_argument('--output_filename', '-f', required=True)
    ######################### Program behavior args
    parser.add_argument('--display', '-d', action='store_true')
    parser.add_argument('--color_selection_method', '-c', choices=['average', 'dominant'], default='average')
    parser.add_argument('--use_every_n_frames', type=int,                   help='How many frames to skip between color grabs - overrides the "width" arg. Using a value of "0" results in every frame being used')
    ######################### Output args
    parser.add_argument('--width', '-w', type=int, default=2560,        help='The width of the resulting palette')
    parser.add_argument('--height', '-g', type=int, default=1080,       help='The height of the resulting palette')
    parser.add_argument('--border_size', type=int, default=0,           help="The size of the border that is added to the final image (0 = no border)")
    parser.add_argument('--border_color', default='(255,255,255)',      help='The color of the border in the final image (in BGR format) - eg "(0,0,255)"')

    args = parser.parse_args()


    # Check if the video exists
    if not os.path.exists(args.video):
        raise Exception("Provided path to video does not reference an existing file")
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    # Create the palette
    palette_img = create_palette_img(args.video, args.width, args.height, args.display, args.color_selection_method, args.use_every_n_frames)
    # Save the image
    file_path = os.path.join(args.output_dir, args.output_filename) 
    cv2.imwrite(file_path, palette_img)