#!/usr/bin/env python
"""
Make MCS tracking animation
- Call plotting script to generate PNG files
- Options to make 1-panel or 2-panel plots
- Create a video animation using FFmpeg

Author: Zhe Feng | zhe.feng@pnnl.gov
"""

import os
import glob
import subprocess
import pandas as pd
import tempfile

###############################################################################################
# Script parameters
###############################################################################################

# Choose feature type: 
# "1panel": Tb and Precipitation in the same panel
# "2panel": Tb and Precipitation in separate panels
plot_type = "1panel"

# Script parameters
# start_date = "2006-06-24T00"
# end_date = "2006-06-27T23"
start_date = "2005-08-31T00"
end_date = "2005-09-03T23"
# Define domain map extent
lon_min = -22.0
lon_max = 48.0
lat_min = 0.0
lat_max = 25.0
# 2-panel orientation (horizontal:up/down, vertical:left/right)
orientation = 'horizontal'
# Get start year from start_date
start_year = str(start_date.split('-')[0])
# Tracking config file
config_file = f"/global/homes/f/feng045/program/pyflex_config/config/config_imerg_mcs_tbpf_{start_year}.yml"

parallel_mode = 0
n_workers = 32
figsize_x = 12  # Width in inches (height is auto-calculated to maintain aspect ratio)

# Optional: Override time frequency for plotting (set to None to auto-calculate)
# Examples: '1H', '3H', '6H'
# plot_freq = '1H'  # Set to desired frequency string or None for auto-calculation

# Tracking pixel-level file time format
time_format = 'yyyymodd_hhmm'
# Title prefix for plots (added before date/time, can be an empty string)
title_prefix = 'GPM Tb+IMERGv06'

# Execution control options
run_plotting = False   # Set to False to skip plotting and use existing PNG files
run_ffmpeg = True     # Set to False to skip animation creation (plotting only)

# FFmpeg animation parameters
input_framerate = 2    # (frames per second) - how fast to transition between frames
output_framerate = 10  # (frames per second) - video playback speed (lower values = smaller file size, e.g., 24 fps is cinema)
video_quality = 20     # CRF value (lower = better quality, range 0-51, 18-28 is good)
output_width = 1920    # Output video width in pixels (height auto-calculated to maintain aspect ratio, set to None to keep original size)

# Set paths and options based on feature type
if plot_type == "1panel":
    # 1-panel configuration
    fig_basename = "mcs_tbpf_"
    figdir = f"/global/cfs/cdirs/m1867/zfeng/gpm/sahel/quicklooks_{plot_type}/{start_year}/"
    animation_dir = f"/global/cfs/cdirs/m1867/zfeng/gpm/sahel/animations_{plot_type}/"
    plotting_code = "plot_subset_tbpf_mcs_tracks_1panel_demo.py"
elif plot_type == "2panel":
    # 2-panel configuration
    fig_basename = "mcs_tbpf_"
    # figdir = f"/pscratch/sd/f/feng045/waccem/mcs_global/quicklooks/2019/"
    figdir = f"/global/cfs/cdirs/m1867/zfeng/gpm/sahel/quicklooks_{plot_type}/{start_year}/"
    animation_dir = f"/global/cfs/cdirs/m1867/zfeng/gpm/sahel/animations_{plot_type}/"
    plotting_code = "plot_subset_tbpf_mcs_tracks_demo.py"

# Animation parameters
start_date_str = start_date.split('T')[0]  # Extract YYYY-MM-DD
end_date_str = end_date.split('T')[0]      # Extract YYYY-MM-DD
animation_filename = f"{animation_dir}{fig_basename}{start_date_str}_{end_date_str}.mp4"

###############################################################################################
# Main execution
###############################################################################################

print("Make MCS tracking animation")
print(f"Plot type: {plot_type}")
print(f"Date range: {start_date} to {end_date}")
print(f"Execution mode: Plotting={'✅' if run_plotting else '❌'}, FFmpeg={'✅' if run_ffmpeg else '❌'}")

# Create directories if they don't exist
if run_ffmpeg:
    os.makedirs(animation_dir, exist_ok=True)
if run_plotting:
    os.makedirs(figdir, exist_ok=True)

#########################################################
# Run plotting script
#########################################################
if run_plotting:
    print(f"📊 Running plotting script with {n_workers} workers...")
    cmd = [
        'python', plotting_code,
        '--start', start_date,
        '--end', end_date,
        '--config', config_file,
        '--extent', str(lon_min), str(lon_max), str(lat_min), str(lat_max),
        '--subset', '1',
        '--time_format', str(time_format),
        '--parallel', str(parallel_mode),
        '--workers', str(n_workers),
        '--output', figdir,
        '--title_prefix', title_prefix,
        '--figsize_x', str(figsize_x),
        '--figbasename', fig_basename,
    ]
    
    # Add orientation argument only for 2-panel plotting (1-panel doesn't use it)
    if plot_type == "2panel":
        cmd.extend(['--orientation', str(orientation)])

    # # Add plot frequency if specified
    # if plot_freq is not None:
    #     cmd.extend(['--plot-freq', plot_freq])
    #     print(f"Using custom plot frequency: {plot_freq}")
    # else:
    #     print("Using auto-calculated plot frequency from dataset")

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ Error: Plotting script failed with exit code {result.returncode}")
        exit(1)
        
    print("✅ Plotting completed successfully!")
else:
    print("⏭️  Skipping plotting - using existing PNG files")

#########################################################
# Make animation using ffmpeg
#########################################################
if run_ffmpeg:
    print("🎬 Creating animation from PNG files...")
    
    # Get all PNG files matching the basename pattern
    all_png_files = sorted(glob.glob(f'{figdir}{fig_basename}*.png'))
    
    # Parse datetime from actual filenames to filter by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    expected_files = []
    for png_file in all_png_files:
        # Extract the datetime string from filename (assumes format: basename_YYYYMMDD_HHMMSS.png)
        try:
            basename_len = len(fig_basename)
            datetime_str = os.path.basename(png_file)[basename_len:basename_len+15]  # YYYYMMDD_HHMMSS
            file_dt = pd.to_datetime(datetime_str, format='%Y%m%d_%H%M%S')
            
            # Check if file is within the specified date range
            if start_dt <= file_dt <= end_dt:
                expected_files.append(png_file)
        except (ValueError, IndexError):
            # Skip files that don't match the expected datetime format
            continue
    
    print(f"Found {len(expected_files)} PNG files within date range")
    print(f"Time range: {start_date} to {end_date}")

    if len(expected_files) == 0:
        print("❌ No PNG files found within the specified date range!")
        print(f"   Check directory: {figdir}")
        print(f"   Expected pattern: {fig_basename}YYYYMMDD_HHMMSS.png")
        exit(1)

    # Create a temporary file list for FFmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_filelist = f.name
        for png_file in sorted(expected_files):
            f.write(f"file '{png_file}'\n")

    try:
        # Build video filter string based on output_width parameter
        if output_width is not None:
            # Scale to specified width, maintaining aspect ratio, ensuring dimensions divisible by 2
            vf_scale = f'scale={output_width}:-2'
        else:
            # Just ensure dimensions are divisible by 2 (no scaling)
            vf_scale = 'scale=trunc(iw/2)*2:trunc(ih/2)*2'
        
        # Use FFmpeg concat demuxer with file list
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-r', str(input_framerate),  # Input framerate
            '-i', temp_filelist,
            '-vf', vf_scale,
            '-c:v', 'libx264',
            '-r', str(output_framerate),  # Output framerate
            '-crf', str(video_quality),
            '-pix_fmt', 'yuv420p',
            '-y', animation_filename
        ]
        
        print(f"🎬 Animation settings:")
        print(f"   Input framerate: {input_framerate} fps (PNG reading speed)")
        print(f"   Output framerate: {output_framerate} fps (video playback speed)")
        print(f"   Video quality (CRF): {video_quality} (lower=better)")
        print(f"   Output width: {output_width if output_width else 'original'} pixels (height auto-scaled)")
        print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        print(f"Using {len(expected_files)} PNG files from {expected_files[0]} to {expected_files[-1]}")
        
        result = subprocess.run(ffmpeg_cmd)
        
    finally:
        # Clean up temporary file
        os.unlink(temp_filelist)

    if result.returncode == 0:
        print(f"✅ Animation created successfully!")
        print(f"🎬 View animation here: {animation_filename}")
    else:
        print(f"❌ Error: FFmpeg failed with exit code {result.returncode}")
        
else:
    print("⏭️  Skipping animation creation - PNG files ready for manual processing")