import cv2
import numpy as np
import os
import argparse
from math import ceil
from tqdm import tqdm
import multiprocessing as mp

from utils import *


def create_pallete_img(colors:list, width:int, height:int) -> np.array:
    
    # Adjust size to fit frames correctly
    num_colors = len(colors)
    width_of_each_strip = ceil(width/num_colors)
    # pallete_width = width_of_each_color_strip * num_colors
    # img_size = (pallete_width, height)

    # Loop through the colors, create a strip for each, and add it to the pallete image
    subpallets = []
    with tqdm(total=len(colors), desc="Creating pallete image") as pbar:
        i=1
        for color in colors:
            cur_strip = np.full((height,width_of_each_strip,3), color)
            if i==1: pallete = cur_strip.copy() # on the first strip for a new subpalette, just set the palette to be the strip
            else: pallete = np.hstack([pallete, cur_strip]) 
            # Every 500 iterations, start creating a new sub-pallete or else the hstack operation starts taking way too long
            if i % 500 == 0:
                subpallets.append(pallete)
                i=0
            i+=1
            pbar.update(1)
        if i!=1: subpallets.append(pallete) # if the number of frames was not a multiple of 500, we need to hstack the subpalette that was still being created

    print("Combining sub-pallets")
    pallete = np.hstack(subpallets)


    return pallete


def create_pallete(video_dir:str, output_dir:str, output_filename:str, pallete_width:int, pallete_height:int, display_frames:bool, color_selection_method:str):
    # Check if the video exists
    if not os.path.exists(video_dir):
        raise Exception("Provided path to video does not reference an existing file")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened(): 
        raise Exception("Unable to open VideoCapture")

    # Get dominant (or average) color from each frame
    colors = get_colors_from_frames(cap, 1, color_selection_method=color_selection_method, display_frames=display_frames)

    # Create the pallete image from the colors array
    pallete_img = create_pallete_img(colors, pallete_width, pallete_height)

    # Save the image
    file_path = os.path.join(output_dir, output_filename) 
    cv2.imwrite(file_path, pallete_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    ######################### Required args
    parser.add_argument('--video', '-v', required=True)
    parser.add_argument('--output_dir', '-o', required=True)
    parser.add_argument('--output_filename', '-f', required=True)
    ######################### Program behavior args
    parser.add_argument('--display', '-d', action='store_true')
    parser.add_argument('--color_selection_method', '-c', choices=['average', 'dominant'], default='average')
    ######################### Output args
    parser.add_argument('--width', '-w', type=int, default=1920,        help='The width of the resulting pallete (the final result will be slightly different to adjust for the number of frames)')
    parser.add_argument('--height', '-g', type=int, default=1080,       help='The height of the resulting pallete')
    parser.add_argument('--border_size', type=int, default=0,           help="The size of the border that is added to the resulting image (0 = no border)")
    parser.add_argument('--border_color', default='(255,255,255)',      help='The color of the border in the resuluting image (in BGR format) - eg "(0,0,255)"')

    args = parser.parse_args()

    create_pallete(args.video, args.output_dir, args.output_filename, args.width, args.height, args.display, args.color_selection_method)