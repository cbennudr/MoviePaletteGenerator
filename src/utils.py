import cv2
import numpy as np
from tqdm import tqdm

def get_avg_color_from_frame(frame:np.array) -> list[3]:
    ### From `https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv`
    average = frame.mean(axis=0).mean(axis=0)

    return average

def get_dominant_color_from_frame(frame:np.array) -> list[3]:
    ### From `https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv`
    pixels = np.float32(frame.reshape(-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    return dominant


def get_colors_from_frames(cap, every_n_frames:int, color_selection_method:str, display_frames:bool) -> list:
    """
    get_colors_from_frames

    Args:
        cap: a cv2.VideoCapture instance
        every_n_frames: the number of frames between color grabs (eg a value of 1 results in every frame having a strip in the resulting pallete while a value of 2 will use every other frame)
        color_selection_method: a string in ["average", "dominant"] that chooses how the color to represent a frame will be chosen

    Returns:
        list: a list containing the BGR tuples in chronological order for the selected colors from each frame
    """

    colors = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # for progress bar
    running = True
    with tqdm(total=total_frames, desc="Extracting colors from frames") as pbar:
        while running:
            ret, frame = cap.read()
            if not ret: 
                running = False
                break
            pbar.update(every_n_frames)

            # Get the color for the current frame using whatever method was selected
            if color_selection_method=='average':
                frame_color = get_avg_color_from_frame(frame)
            elif color_selection_method=='majority':
                frame_color = get_dominant_color_from_frame(frame)
            else:
                raise ValueError("Invalid color selection method")
            
            colors.append(frame_color)

            # If displaying frames - create a strip, add it to the side of the frame, and display the result
            display_strip_width = 20
            if display_frames:
                display_strip = np.full((frame.shape[0],display_strip_width,3), frame_color)
                display_frame = np.hstack([frame, display_strip])
                cv2.imshow('Video', display_frame)
                cv2.waitKey(1)

    return colors