import sys
import numpy as np
import cv2
import math
import subprocess
import os
import argparse
from pathlib import Path
from PIL import Image

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2
SAMPLE_SECONDS = 2 # Extracts color correction from every N seconds
RED_FILTER_REDUCTION = 0.7 # Reduction factor for red channel when using red filter

def hue_shift_red(mat, h):

    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])

def normalizing_interval(array):

    high = 255
    low = 0
    max_dist = 0

    for i in range(1, len(array)):
        dist = array[i] - array[i-1]
        if(dist > max_dist):
            max_dist = dist
            high = array[i]
            low = array[i-1]

    return (low, high)

def apply_filter(mat, filt, red_filter=False):
    filtered_mat = np.zeros_like(mat, dtype=np.float32)
    filtered_mat[..., 0] = mat[..., 0] * filt[0] + mat[..., 1] * filt[1] + mat[..., 2] * filt[2] + filt[4] * 255
    filtered_mat[..., 1] = mat[..., 1] * filt[6] + filt[9] * 255
    filtered_mat[..., 2] = mat[..., 2] * filt[12] + filt[14] * 255
    
    # If red filter is enabled, reduce red channel saturation
    if red_filter:
        filtered_mat[..., 0] = filtered_mat[..., 0] * RED_FILTER_REDUCTION
    
    return np.clip(filtered_mat, 0, 255).astype(np.uint8)

def get_filter_matrix(mat):

    mat = cv2.resize(mat, (256, 256))

    # Get average values of RGB
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)
    
    # Find hue shift so that average red reaches MIN_AVG_RED
    new_avg_r = avg_mat[0]
    hue_shift = 0
    while(new_avg_r < MIN_AVG_RED):

        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            new_avg_r = MIN_AVG_RED

    # Apply hue shift to whole image and replace red channel
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat[..., 0] = new_r_channel

    # Get histogram of all channels
    hist_r = hist = cv2.calcHist([mat], [0], None, [256], [0,256])
    hist_g = hist = cv2.calcHist([mat], [1], None, [256], [0,256])
    hist_b = hist = cv2.calcHist([mat], [2], None, [256], [0,256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0]*mat.shape[1])/THRESHOLD_RATIO
    for x in range(256):
        
        if hist_r[x] < threshold_level:
            normalize_mat[x][0] = x

        if hist_g[x] < threshold_level:
            normalize_mat[x][1] = x

        if hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])


    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain = 256 / (adjust_r_high - adjust_r_low)
    green_gain = 256 / (adjust_g_high - adjust_g_low)
    blue_gain = 256 / (adjust_b_high - adjust_b_low)

    redOffset = (-adjust_r_low / 256) * red_gain
    greenOffset = (-adjust_g_low / 256) * green_gain
    blueOffset = (-adjust_b_low / 256) * blue_gain

    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * BLUE_MAGIC_VALUE

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])

def correct(mat, red_filter=False):
    original_mat = mat.copy()

    filter_matrix = get_filter_matrix(mat)
    
    corrected_mat = apply_filter(original_mat, filter_matrix, red_filter=red_filter)
    corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)

    return corrected_mat

def correct_image(input_path, output_path, red_filter=False):
    exif_data = None
    with Image.open(input_path) as image:
        exif_data = image.info.get("exif")
        if image.mode != "RGB":
            image = image.convert("RGB")
        mat = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    corrected_mat = correct(rgb_mat, red_filter=red_filter)

    output_image = Image.fromarray(cv2.cvtColor(corrected_mat, cv2.COLOR_BGR2RGB))
    save_kwargs = {}
    if exif_data:
        save_kwargs["exif"] = exif_data
    output_image.save(output_path, **save_kwargs)
    
    preview = mat.copy()
    width = preview.shape[1] // 2
    preview[::, width:] = corrected_mat[::, width:]

    preview = cv2.resize(preview, (960, 540))

    return cv2.imencode('.png', preview)[1].tobytes()


def analyze_video(input_video_path, output_video_path):
    
    # Initialize new video writer
    cap = cv2.VideoCapture(input_video_path)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get filter matrices for every 10th frame
    filter_matrix_indexes = []
    filter_matrices = []
    count = 0
    
    print("Analyzing...")
    while(cap.isOpened()):
        
        count += 1  
        print(f"{count} frames", end="\r")
        ret, frame = cap.read()
        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break

            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break

            # Otherwise this is just a faulty frame read, try reading next frame
            continue

        # Pick filter matrix from every N seconds
        if count % (fps * SAMPLE_SECONDS) == 0:
            mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filter_matrix_indexes.append(count) 
            filter_matrices.append(get_filter_matrix(mat))

        yield count
        
    cap.release()

    # Build a interpolation function to get filter matrix at any given frame
    filter_matrices = np.array(filter_matrices)

    yield {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "fps": fps,
        "frame_count": count,
        "filters": filter_matrices,
        "filter_indices": filter_matrix_indexes
    }

def precompute_filter_matrices(frame_count, filter_indices, filter_matrices):
    filter_matrix_size = len(filter_matrices[0])
    frame_numbers = np.arange(frame_count)
    interpolated_matrices = np.zeros((frame_count, filter_matrix_size))
    for x in range(filter_matrix_size):
        interpolated_matrices[:, x] = np.interp(frame_numbers, filter_indices, filter_matrices[:, x])
    return interpolated_matrices

def process_video(video_data, yield_preview=False, red_filter=False):
    cap = cv2.VideoCapture(video_data["input_video_path"])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_data["fps"]
    frame_count = video_data["frame_count"]

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_video = cv2.VideoWriter(video_data["output_video_path"], fourcc, fps, (frame_width, frame_height))

    # Precompute interpolated filter matrices
    print("Precomputing filter matrices...")
    interpolated_matrices = precompute_filter_matrices(
        frame_count, video_data["filter_indices"], np.array(video_data["filters"])
    )

    print("Processing...")
    count = 0
    while cap.isOpened():
        count += 1
        percent = 100 * count / frame_count
        print("{:.2f}%".format(percent), end="\r")
        ret, frame = cap.read()
        
        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break

            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break

            # Otherwise this is just a faulty frame read, try reading next
            continue

        # Apply the filter using precomputed matrix
        rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        corrected_mat = apply_filter(rgb_mat, interpolated_matrices[count - 1], red_filter=red_filter)
        corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)
        new_video.write(corrected_mat)

        if yield_preview:
            preview = frame.copy()
            width = preview.shape[1] // 2
            height = preview.shape[0] // 2
            preview[:, width:] = corrected_mat[:, width:]

            preview = cv2.resize(preview, (width, height))

            yield percent, cv2.imencode('.png', preview)[1].tobytes()
        else:
            yield None

    
    
    cap.release()
    new_video.release()
    
    # Add audio from original video using ffmpeg
    temp_output = video_data["output_video_path"] + ".temp.mp4"
    os.rename(video_data["output_video_path"], temp_output)
    
    print("\nAdding audio...")
    ffmpeg_cmd = [
        'ffmpeg', '-i', temp_output, '-i', video_data["input_video_path"],
        '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0?',
        '-shortest', video_data["output_video_path"], '-y'
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        os.remove(temp_output)
    except subprocess.CalledProcessError:
        # If ffmpeg fails, keep the video without audio
        os.rename(temp_output, video_data["output_video_path"])
        print("Warning: Could not add audio track")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dive video/image color corrector')
    parser.add_argument('mode', choices=['image', 'video', 'batch'], help='Processing mode')
    parser.add_argument('input', help='Input file or directory path')
    parser.add_argument('output', help='Output file or directory path')
    parser.add_argument('--red-filter', action='store_true', 
                        help='Reduce red saturation (use when camera has red filter)')
    
    args = parser.parse_args()

    if args.mode == "image":
        mat = cv2.imread(args.input)
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        
        corrected_mat = correct(mat, red_filter=args.red_filter)

        cv2.imwrite(args.output, corrected_mat)
    
    elif args.mode == "video":
        for item in analyze_video(args.input, args.output):
            if type(item) == dict:
                video_data = item
            
        [x for x in process_video(video_data, yield_preview=False, red_filter=args.red_filter)]
    
    else:  # batch mode
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        
        if not input_dir.is_dir():
            print(f"Error: {args.input} is not a directory")
            exit(1)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported video extensions
        video_extensions = ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV']
        image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
        
        # Find all videos and images
        files = []
        for ext in video_extensions + image_extensions:
            files.extend(list(input_dir.glob(f'*{ext}')))
        
        if not files:
            print(f"No video or image files found in {args.input}")
            exit(0)
        
        print(f"Found {len(files)} file(s) to process")
        
        for i, input_file in enumerate(files, 1):
            output_file = output_dir / input_file.name
            print(f"\n[{i}/{len(files)}] Processing {input_file.name}...")
            
            # Check if it's a video or image
            if input_file.suffix in video_extensions:
                try:
                    for item in analyze_video(str(input_file), str(output_file)):
                        if type(item) == dict:
                            video_data = item
                    
                    [x for x in process_video(video_data, yield_preview=False, red_filter=args.red_filter)]
                    print(f"✓ Completed {input_file.name}")
                except Exception as e:
                    print(f"✗ Error processing {input_file.name}: {e}")
            
            elif input_file.suffix in image_extensions:
                try:
                    mat = cv2.imread(str(input_file))
                    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
                    corrected_mat = correct(mat, red_filter=args.red_filter)
                    cv2.imwrite(str(output_file), corrected_mat)
                    print(f"✓ Completed {input_file.name}")
                except Exception as e:
                    print(f"✗ Error processing {input_file.name}: {e}")
        
        print(f"\nBatch processing complete! Processed {len(files)} file(s).")
        
