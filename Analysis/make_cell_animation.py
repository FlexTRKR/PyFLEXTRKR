#!/usr/bin/env python
"""
Make cell tracking animation
- Call plotting script to generate PNG files
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

# Script parameters
start_date = "2020-06-13T00:00"
end_date = "2020-06-16T00:00"

# Radar location (optional - if not provided, defaults to domain center)
radar_lat = 34.723 
radar_lon = 273.428

# Define domain map extent (optional - if not provided, defaults to full domain)
lon_min = radar_lon - 6.0
lon_max = radar_lon + 6.0
lat_min = radar_lat - 4
lat_max = radar_lat + 4

# Get start year from start_date
start_year = str(start_date.split('-')[0])

# Tracking config file
config_file = f"/global/homes/f/feng045/program/scream/config/config_celltracking_3km_SCREAMv1-Cess2_CONUS.yaml"

# Execution parameters
parallel_mode = 1
n_workers = 128
figsize_x = 10  # Width in inches (height is auto-calculated to maintain aspect ratio)

# Tracking pixel-level file time format
time_format = 'yyyymodd_hhmmss'

# Variable name for plotting in pixel files
varname = 'dbz_comp'

# Title prefix for plots (added before date/time, can be an empty string)
title_prefix = 'SCREAM Cess2'

# Control features to draw
draw_land = True  # Draw land features
draw_border = True  # Draw country borders
draw_state = True    # Draw state/province borders
show_rangecircle = False  # Draw radar range circles
show_azimuth = False      # Draw azimuth lines
show_tracks = True        # Show tracks
show_paths = True         # Show track path lines (None = auto from show_tracks)
show_symbols = False       # Show track centroid symbols (None = auto from show_tracks)
# Font sizes
fontsize = 13              # Main font size for labels and text
fontsize_tracks = 6        # Font size for track numbers (None = auto-calculate as fontsize*0.8)
# Map resolution for Natural Earth features (Cartopy options: '10m', '50m', '110m')
# '10m' = highest detail (slowest), '50m' = medium detail, '110m' = lowest detail (fastest)
map_resolution = '50m'
# Figure parameters
subset = 1  # Subset data before plotting (0:no, 1:yes)
fig_basename = "cell_tracks_"

# Output directories
figdir = f"/global/cfs/cdirs/m1657/zfeng/SCREAMv1-cess2/cell_conus/quicklooks/"
animation_dir = f"/global/cfs/cdirs/m1657/zfeng/SCREAMv1-cess2/cell_conus/animations/"

# Execution control options
run_plotting = True   # Set to False to skip plotting and use existing PNG files
run_ffmpeg = True     # Set to False to skip animation creation (plotting only)

# FFmpeg animation parameters
input_framerate = 2    # (frames per second) - how fast to transition between frames
output_framerate = 10  # (frames per second) - video playback speed (lower values = smaller file size, e.g., 24 fps is cinema)
video_quality = 20     # CRF value (lower = better quality, range 0-51, 18-28 is good)
output_width = 1920    # Output video width in pixels (height auto-calculated to maintain aspect ratio, set to None to keep original size)

# Animation parameters
start_date_str = start_date[:13]  # Extract YYYY-MM-DDTHH
end_date_str = end_date[:13]      # Extract YYYY-MM-DDTHH
animation_filename = f"{animation_dir}{fig_basename}{start_date_str}_{end_date_str}.mp4"

###############################################################################################
# Main execution
###############################################################################################

print("Make cell tracking animation")
print(f"Plotting script: plot_subset_cell_tracks_demo.py")
print(f"Date range: {start_date} to {end_date}")
print(f"Radar location: ({radar_lat}, {radar_lon})")
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
        'python', 'plot_subset_cell_tracks_demo.py',
        '--start', start_date,
        '--end', end_date,
        '--config', config_file,
    ]
    
    # Add radar location if provided
    if radar_lat is not None:
        cmd.extend(['--radar_lat', str(radar_lat)])
    if radar_lon is not None:
        cmd.extend(['--radar_lon', str(radar_lon)])
    
    # Add extent if all values are provided
    if all(v is not None for v in [lon_min, lon_max, lat_min, lat_max]):
        cmd.extend(['--extent', str(lon_min), str(lon_max), str(lat_min), str(lat_max)])
    
    cmd.extend([
        '--subset', str(subset),
        '--time_format', str(time_format),
        '--parallel', str(parallel_mode),
        '--workers', str(n_workers),
        '--output', figdir,
        '--figsize_x', str(figsize_x),
        '--figbasename', fig_basename,
        '--varname', varname,
        '--draw_land', str(int(draw_land)),
        '--draw_border', str(int(draw_border)),
        '--draw_state', str(int(draw_state)),
        '--show_rangecircle', str(int(show_rangecircle)),
        '--show_azimuth', str(int(show_azimuth)),
        '--show_tracks', str(int(show_tracks)),
        '--fontsize', str(fontsize),
        '--map_resolution', map_resolution,
        '--title_prefix', title_prefix,
    ])
    
    # Add optional show_paths and show_symbols if specified (not None)
    if show_paths is not None:
        cmd.extend(['--show_paths', str(int(show_paths))])
    if show_symbols is not None:
        cmd.extend(['--show_symbols', str(int(show_symbols))])
    if fontsize_tracks is not None:
        cmd.extend(['--fontsize_tracks', str(fontsize_tracks)])

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
