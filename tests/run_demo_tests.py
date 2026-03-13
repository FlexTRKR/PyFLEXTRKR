#!/usr/bin/env python
"""
Automated demo test runner for PyFLEXTRKR.

Downloads sample data, runs end-to-end tracking demos, and validates outputs.
Replaces the manual workflow of editing dir_demo in each shell script,
running demos one-by-one, and visually inspecting results.

Usage examples
--------------
  # List available demos
  python tests/run_demo_tests.py --list

  # Run specific demos with 4 workers
  python tests/run_demo_tests.py --demos demo_mcs_imerg demo_cell_nexrad -n 4

  # Run all portable demos
  python tests/run_demo_tests.py --all -n 8

  # Validate outputs without re-running (e.g. after a previous run)
  python tests/run_demo_tests.py --demos demo_mcs_imerg --validate-only

  # Custom data root directory
  python tests/run_demo_tests.py --demos demo_mcs_imerg --data-root /tmp/demo -n 4

  # Include quicklook plots and animation
  python tests/run_demo_tests.py --demos demo_mcs_imerg --with-plots -n 4

  # Force re-download of input data
  python tests/run_demo_tests.py --demos demo_mcs_imerg --fresh -n 4

  # Back up existing outputs instead of deleting
  python tests/run_demo_tests.py --demos demo_mcs_imerg --backup -n 4

Workflow
--------
For each selected demo, the runner:
  1. Cleans previous output directories (keeps input/ unless --fresh)
  2. Downloads & extracts sample data (skipped if input/ already has files)
  3. Generates a tracking config from the template (overrides nprocesses)
  4. Runs the tracking pipeline
  5. (Optional) Runs quicklook plotting & ffmpeg animation
  6. Validates outputs (stats files, pixel files, lat/lon ranges)
  7. Prints a summary table

The runner also sets PYFLEXTRKR_TEST_DATA so that pytest-based regression
tests (test_regression_local.py) can validate the same outputs afterward:

  export PYFLEXTRKR_TEST_DATA=~/data/demo
  python -m pytest tests/test_regression_local.py -v
"""

import argparse
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_DATA_ROOT = Path.home() / "data" / "demo"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("demo_tests")


# ---------------------------------------------------------------------------
# Demo registry
# ---------------------------------------------------------------------------
@dataclass
class DemoInfo:
    """Metadata for a single end-to-end demo."""
    name: str
    description: str
    # Data
    data_subdir: str          # path under data_root (e.g. 'mcs_tbpf/imerg')
    data_url: str             # wget URL for sample tar.gz
    config_template: str      # YAML template filename in config/
    runscript: str            # Python runscript path relative to repo root
    # Extra config substitutions beyond INPUT_DIR / TRACK_DIR
    extra_subs: Dict[str, str] = field(default_factory=dict)
    # Validation metadata
    stats_pattern: str = "trackstats_*.nc"
    pixel_path_name: str = "mcstracking"
    pixel_filebase: str = "mcstracking_"
    min_tracks: int = 1
    lat_var: str = "meanlat"
    lon_var: str = "meanlon"
    lat_range: Tuple[float, float] = (-90, 90)
    lon_range: Tuple[float, float] = (-180, 360)
    # Plotting (optional)
    plot_script: str = ""
    plot_args: str = ""
    quicklook_subdir: str = "quicklooks_trackpaths"
    ffmpeg_framerate: int = 2
    animation_name: str = "quicklook_animation.mp4"


DEMOS: Dict[str, DemoInfo] = {
    # ── MCS Tb+PF demos ─────────────────────────────────────────────
    "demo_mcs_imerg": DemoInfo(
        name="demo_mcs_imerg",
        description="MCS tracking – GPM IMERG Tb+Precip",
        data_subdir="mcs_tbpf/imerg",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/gpm_tb_imerg.tar.gz",
        config_template="config_imerg_mcs_tbpf_example.yml",
        runscript="runscripts/run_mcs_tbpf.py",
        stats_pattern="mcs_tracks_final_*.nc",
        pixel_filebase="mcstrack_",
        lat_range=(-60, 20),
        lon_range=(-80, 10),
        plot_script="Analysis/plot_subset_tbpf_mcs_tracks_demo.py",
        plot_args="-s '2019-01-25T00' -e '2019-01-27T00' -o vertical -p 1 --figsize 10 8",
    ),
    "demo_mcs_wrf_tbpf": DemoInfo(
        name="demo_mcs_wrf_tbpf",
        description="MCS tracking – WRF Tb+Precip",
        data_subdir="mcs_tbpf/wrf",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/wrf_tbpcp.tar.gz",
        config_template="config_wrf4km_mcs_tbpf_example.yml",
        runscript="runscripts/run_mcs_tbpf.py",
        stats_pattern="mcs_tracks_final_*.nc",
        pixel_filebase="mcstrack_",
        lat_range=(-20, 5),
        lon_range=(-80, -40),
        plot_script="Analysis/plot_subset_tbpf_mcs_tracks_demo.py",
        plot_args="-s '2014-03-19T00' -e '2014-03-21T00' -o horizontal -p 1 --figsize 9 10",
    ),
    "demo_mcs_model25km": DemoInfo(
        name="demo_mcs_model25km",
        description="MCS tracking – E3SM OLR+Precip (25 km)",
        data_subdir="mcs_tbpf/e3sm",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/e3sm_tbpcp.tar.gz",
        config_template="config_model25km_mcs_tbpf_example.yml",
        runscript="runscripts/run_mcs_tbpf.py",
        stats_pattern="mcs_tracks_final_*.nc",
        pixel_filebase="mcstrack_",
        lat_range=(-60, 60),
        lon_range=(-180, 360),
        quicklook_subdir="quicklooks_robust",
        plot_script="Analysis/plot_subset_tbpf_mcs_tracks_demo.py",
        plot_args="-s '2007-05-07T00' -e '2007-05-12T00' -o horizontal -p 1 --figsize 10 10",
        animation_name="mcs_robust_animation.mp4",
    ),
    "demo_mcs_tbpf_idealized": DemoInfo(
        name="demo_mcs_tbpf_idealized",
        description="MCS tracking – idealized Tb+Precip",
        data_subdir="mcs_tbpf/idealized/test4",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/idealized_tbpcp.tar.gz",
        config_template="config_mcs_idealized.yml",
        runscript="runscripts/run_mcs_tbpf.py",
        extra_subs={"BASENAME": "MCS-test-4_"},
        stats_pattern="mcs_tracks_final_*.nc",
        pixel_filebase="mcstrack_",
        lat_range=(-90, 90),
        lon_range=(-180, 360),
        plot_script="Analysis/plot_subset_tbpf_mcs_tracks_demo.py",
        plot_args="-s '2020-01-01T00' -e '2020-01-03T00' -o vertical -p 1 --figsize 10 8",
    ),
    # ── MCS Tb+Radar demos ──────────────────────────────────────────
    "demo_mcs_wrf_tbradar": DemoInfo(
        name="demo_mcs_wrf_tbradar",
        description="MCS tracking – WRF Tb+Radar",
        data_subdir="mcs_tbpfradar3d/wrf",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_radar/wrf_tbradar.tar.gz",
        config_template="config_wrf_mcs_tbradar_example.yml",
        runscript="runscripts/run_mcs_tbpfradar3d_wrf.py",
        stats_pattern="mcs_tracks_final_*.nc",
        pixel_filebase="mcstrack_",
        lat_range=(25, 55),
        lon_range=(-115, -75),
        plot_script="Analysis/plot_subset_tbze_mcs_tracks_demo.py",
        plot_args="-s '2015-05-06T00' -e '2015-05-10T00' -o horizontal -p 1 --figsize 8 12",
    ),
    "demo_mcs_gridrad": DemoInfo(
        name="demo_mcs_gridrad",
        description="MCS tracking – GridRad Tb+Radar",
        data_subdir="mcs_tbpfradar3d/gridrad",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_radar/gridrad_tbradar.tar.gz",
        config_template="config_gridrad_mcs_example.yml",
        runscript="runscripts/run_mcs_tbpfradar3d_wrf.py",
        stats_pattern="mcs_tracks_final_*.nc",
        pixel_filebase="mcstrack_",
        lat_range=(20, 55),
        lon_range=(-130, -60),
        plot_script="Analysis/plot_subset_tbze_mcs_tracks_demo.py",
        plot_args="-s '2020-08-10T00' -e '2020-08-13T00' -o horizontal -p 1 --figsize 10 13",
    ),
    # ── MCS Tb-only demo ────────────────────────────────────────────
    "demo_mcs_himawari": DemoInfo(
        name="demo_mcs_himawari",
        description="MCS tracking – Himawari Tb-only",
        data_subdir="mcs_tbpf/himawari",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/tb_pcp/himawari_tb.tar.gz",
        config_template="config_himawari_mcs_example.yml",
        runscript="runscripts/run_mcs_tb.py",
        stats_pattern="mcs_tracks_*.nc",
        pixel_filebase="mcstrack_",
        pixel_path_name="mcstracking_tb",
        lat_range=(-60, 60),
        lon_range=(80, 200),
        plot_script="Analysis/plot_subset_tb_mcs_tracks_demo.py",
        plot_args="-s '2021-10-24T00' -e '2021-10-25T00' -p 1 --figsize 10 10",
    ),
    # ── Cell tracking demos ─────────────────────────────────────────
    "demo_cell_nexrad": DemoInfo(
        name="demo_cell_nexrad",
        description="Cell tracking – NEXRAD radar (KHGX)",
        data_subdir="cell_radar/nexrad",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/radar/nexrad_reflectivity1.tar.gz",
        config_template="config_nexrad500m_example.yml",
        runscript="runscripts/run_celltracking.py",
        stats_pattern="trackstats_*.nc",
        pixel_path_name="celltracking",
        pixel_filebase="celltracks_",
        lat_var="cell_meanlat",
        lon_var="cell_meanlon",
        lat_range=(28, 32),
        lon_range=(-97, -93),
        plot_script="Analysis/plot_subset_cell_tracks_demo.py",
        plot_args="-s '2014-08-07T12' -e '2014-08-07T15' --radar_lat 29.4719 --radar_lon -95.0787 -p 1 --figsize 8 7",
        ffmpeg_framerate=4,
    ),
    "demo_cell_csapr": DemoInfo(
        name="demo_cell_csapr",
        description="Cell tracking – CACTI CSAPR2 radar",
        data_subdir="cell_radar/csapr",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/radar/taranis_corcsapr2.tar.gz",
        config_template="config_csapr500m_example.yml",
        runscript="runscripts/run_celltracking.py",
        stats_pattern="trackstats_*.nc",
        pixel_path_name="celltracking",
        pixel_filebase="celltracks_",
        lat_var="cell_meanlat",
        lon_var="cell_meanlon",
        lat_range=(-35, -30),
        lon_range=(-67, -62),
        plot_script="Analysis/plot_subset_cell_tracks_demo.py",
        plot_args="-s '2019-01-25T12' -e '2019-01-26T00' --radar_lat -32.12641 --radar_lon -64.72837 -p 1 --figsize 8 7",
    ),
    # ── Generic tracking demo ───────────────────────────────────────
    "demo_generic_tracking": DemoInfo(
        name="demo_generic_tracking",
        description="Generic feature tracking – ERA5 Z500 anomaly",
        data_subdir="general_tracking/z500_blocking",
        data_url="https://portal.nersc.gov/project/m1867/PyFLEXTRKR/sample_data/generic/ERA5_z500_anom.tar.gz",
        config_template="config_era5_z500_example.yml",
        runscript="runscripts/run_generic_tracking.py",
        stats_pattern="trackstats_*.nc",
        pixel_path_name="z500tracking",
        pixel_filebase="z500tracks_",
        lat_range=(-90, 90),
        lon_range=(-180, 360),
        plot_script="Analysis/plot_subset_generic_tracks_demo.py",
        plot_args="-s '1979-06-01' -e '1979-08-31' --figsize 8 4 -p 1",
        ffmpeg_framerate=4,
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_stats_file(stats_dir: Path, pattern: str) -> Optional[Path]:
    """Find the best stats file matching pattern, excluding sparse files."""
    files = sorted(stats_dir.glob(pattern))
    files = [f for f in files if "sparse" not in f.name]
    return files[-1] if files else None


def clean_output_dirs(dir_demo: Path, backup: bool = False):
    """Remove all output subdirectories in dir_demo, preserving input/."""
    if not dir_demo.exists():
        return
    for entry in sorted(dir_demo.iterdir()):
        if entry.is_dir() and entry.name != "input":
            if backup:
                bak = entry.with_name(entry.name + ".bak")
                if bak.exists():
                    shutil.rmtree(bak)
                entry.rename(bak)
                logger.info(f"  Backed up {entry.name}/ -> {bak.name}/")
            else:
                shutil.rmtree(entry)
                logger.info(f"  Removed {entry.name}/")


def download_and_extract(data_url: str, dir_input: Path):
    """Download a tar.gz and extract it into dir_input."""
    dir_input.mkdir(parents=True, exist_ok=True)
    tar_name = Path(data_url).name
    tar_path = dir_input / tar_name

    logger.info(f"  Downloading {tar_name} ...")
    result = subprocess.run(
        ["wget", "-q", "--show-progress", data_url, "-O", str(tar_path)],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed (exit {result.returncode}): {data_url}")

    logger.info(f"  Extracting ...")
    subprocess.run(
        ["tar", "-xzf", str(tar_path), "-C", str(dir_input)],
        check=True,
    )
    tar_path.unlink()
    logger.info(f"  Input data ready in {dir_input}")


def generate_config(
    demo: DemoInfo,
    dir_input: Path,
    dir_demo: Path,
    nworkers: int,
) -> Path:
    """Create a tracking config from the template with correct paths."""
    template_path = CONFIG_DIR / demo.config_template
    text = template_path.read_text()

    # Standard substitutions (match what the bash scripts do)
    text = text.replace("INPUT_DIR", str(dir_input) + "/")
    text = text.replace("TRACK_DIR", str(dir_demo) + "/")

    # Extra substitutions (e.g. BASENAME for idealized demo)
    for key, val in demo.extra_subs.items():
        text = text.replace(key, val)

    # Override nprocesses
    text = re.sub(
        r"^(nprocesses\s*:\s*).*$",
        rf"\g<1>{nworkers}  # auto-set by run_demo_tests",
        text,
        flags=re.MULTILINE,
    )

    config_out = CONFIG_DIR / f"config_autotest_{demo.name}.yml"
    config_out.write_text(text)
    logger.info(f"  Config: {config_out.name} (nprocesses={nworkers})")
    return config_out


def run_tracking(demo: DemoInfo, config_path: Path) -> subprocess.CompletedProcess:
    """Run the tracking pipeline."""
    runscript = REPO_ROOT / demo.runscript
    cmd = [sys.executable, str(runscript), str(config_path)]
    logger.info(f"  Running: {Path(demo.runscript).name} ...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(CONFIG_DIR))
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info(f"  Tracking completed in {elapsed:.0f}s")
    else:
        logger.error(f"  Tracking FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
    return result


def run_plots_and_animation(demo: DemoInfo, dir_demo: Path, config_path: Path):
    """Run quicklook plotting and ffmpeg animation."""
    quicklook_dir = dir_demo / demo.quicklook_subdir
    quicklook_dir.mkdir(parents=True, exist_ok=True)

    if demo.plot_script:
        plot_script = REPO_ROOT / demo.plot_script
        # Plot scripts concatenate output path + filename directly,
        # so the path must end with '/'
        quicklook_str = str(quicklook_dir) + "/"
        cmd = (
            f"{sys.executable} {plot_script} {demo.plot_args}"
            f" -c {config_path} --output {quicklook_str}"
        )
        logger.info(f"  Plotting: {Path(demo.plot_script).name}")
        subprocess.run(cmd, shell=True, cwd=str(CONFIG_DIR))

    # ffmpeg animation
    pngs = sorted(quicklook_dir.glob("*.png"))
    if pngs:
        anim_path = quicklook_dir / demo.animation_name
        cmd = [
            "ffmpeg", "-framerate", str(demo.ffmpeg_framerate),
            "-pattern_type", "glob", "-i", str(quicklook_dir / "*.png"),
            "-c:v", "libx264", "-r", "10", "-crf", "20",
            "-pix_fmt", "yuv420p", "-y", str(anim_path),
        ]
        logger.info(f"  Creating animation ({len(pngs)} frames) ...")
        subprocess.run(cmd, capture_output=True)
        if anim_path.exists():
            logger.info(f"  Animation: {anim_path.name}")
        else:
            logger.warning("  ffmpeg animation failed (is ffmpeg installed?)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating one demo's outputs."""
    demo_name: str
    passed: bool = True
    n_tracks: int = 0
    messages: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def fail(self, msg: str):
        self.passed = False
        self.messages.append(f"FAIL: {msg}")

    def ok(self, msg: str):
        self.messages.append(f"  OK: {msg}")


def validate_demo(
    demo: DemoInfo, dir_demo: Path, check_plots: bool = False,
) -> ValidationResult:
    """Validate outputs of a completed demo run."""
    result = ValidationResult(demo_name=demo.name)

    if not HAS_XARRAY:
        result.fail("xarray not installed — cannot validate NetCDF outputs")
        return result

    stats_dir = dir_demo / "stats"

    # ── 1. Stats file ──────────────────────────────────────────────
    stat_file = find_stats_file(stats_dir, demo.stats_pattern)
    if stat_file is None:
        # Fallback: any trackstats file
        stat_file = find_stats_file(stats_dir, "trackstats_*.nc")
    if stat_file is None:
        result.fail(f"No stats file matching '{demo.stats_pattern}' in {stats_dir}")
        return result

    try:
        with xr.open_dataset(stat_file) as ds:
            # Track count
            if "tracks" not in ds.dims:
                result.fail(f"No 'tracks' dimension in {stat_file.name}")
                return result
            n_tracks = ds.sizes["tracks"]
            result.n_tracks = n_tracks
            if n_tracks < demo.min_tracks:
                result.fail(f"Only {n_tracks} tracks (need >= {demo.min_tracks})")
            else:
                result.ok(f"{n_tracks} tracks in {stat_file.name}")

            # Lat/lon ranges
            for var_name, expected_range, label in [
                (demo.lat_var, demo.lat_range, "latitude"),
                (demo.lon_var, demo.lon_range, "longitude"),
            ]:
                if var_name in ds:
                    vals = ds[var_name].values.ravel()
                    valid = vals[np.isfinite(vals)]
                    if len(valid) == 0:
                        result.fail(f"{var_name} has no finite values")
                    else:
                        lo, hi = expected_range
                        if np.all(valid >= lo) and np.all(valid <= hi):
                            result.ok(
                                f"{label}: {valid.min():.1f} to {valid.max():.1f} "
                                f"(within [{lo}, {hi}])"
                            )
                        else:
                            result.fail(
                                f"{label} out of range: "
                                f"{valid.min():.1f}..{valid.max():.1f} "
                                f"expected [{lo}, {hi}]"
                            )

            # Track durations
            dur_var = "track_duration"
            if dur_var in ds:
                dur = ds[dur_var].values.ravel()
                valid_dur = dur[np.isfinite(dur)]
                if len(valid_dur) > 0 and np.all(valid_dur >= 0):
                    result.ok("track durations non-negative")
                elif len(valid_dur) > 0:
                    result.fail("negative track durations found")
    except Exception as e:
        result.fail(f"Error reading stats file: {e}")
        return result

    # ── 2. Pixel-level files ───────────────────────────────────────
    pixel_dir = dir_demo / demo.pixel_path_name
    pattern = str(pixel_dir / "**" / f"{demo.pixel_filebase}*.nc")
    pixel_files = sorted(glob.glob(pattern, recursive=True))
    if pixel_files:
        result.ok(f"{len(pixel_files)} pixel-level files in {demo.pixel_path_name}/")
        # Spot-check first file for expected variable
        try:
            with xr.open_dataset(pixel_files[0]) as ds:
                if "tracknumber" in ds:
                    result.ok("tracknumber variable present")
                elif "pcptracknumber" in ds:
                    result.ok("pcptracknumber variable present")
        except Exception as e:
            result.fail(f"Error reading pixel file: {e}")
    else:
        result.fail(f"No pixel files found in {demo.pixel_path_name}/")

    # ── 3. Plots / animation (optional) ───────────────────────────
    if check_plots:
        ql_dir = dir_demo / demo.quicklook_subdir
        pngs = list(ql_dir.glob("*.png")) if ql_dir.exists() else []
        if pngs:
            result.ok(f"{len(pngs)} quicklook PNGs")
        else:
            result.fail(f"No quicklook PNGs in {demo.quicklook_subdir}/")

        anim = ql_dir / demo.animation_name if ql_dir.exists() else None
        if anim and anim.exists():
            result.ok(f"Animation: {demo.animation_name}")
        else:
            result.fail(f"Animation not found: {demo.animation_name}")

    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_single_demo(
    demo: DemoInfo,
    data_root: Path,
    nworkers: int,
    with_plots: bool = False,
    backup: bool = False,
    fresh: bool = False,
    validate_only: bool = False,
) -> ValidationResult:
    """Run one demo end-to-end and validate."""
    dir_demo = data_root / demo.data_subdir
    dir_input = dir_demo / "input"
    t0 = time.time()

    header = f"\n{'='*64}\n  {demo.name}: {demo.description}\n{'='*64}"
    logger.info(header)

    if not validate_only:
        # Clean outputs
        logger.info("Cleaning previous outputs ...")
        clean_output_dirs(dir_demo, backup=backup)

        # Download (unless input exists and not --fresh)
        input_has_files = dir_input.exists() and any(dir_input.iterdir())
        if fresh or not input_has_files:
            try:
                download_and_extract(demo.data_url, dir_input)
            except RuntimeError as e:
                result = ValidationResult(
                    demo_name=demo.name, elapsed_sec=time.time() - t0,
                )
                result.fail(str(e))
                return result
        else:
            n_items = len(list(dir_input.iterdir()))
            logger.info(f"  Input data exists ({n_items} items), skipping download")

        # Generate config
        config_path = generate_config(demo, dir_input, dir_demo, nworkers)

        # Run tracking
        proc = run_tracking(demo, config_path)
        if proc.returncode != 0:
            result = ValidationResult(
                demo_name=demo.name, elapsed_sec=time.time() - t0,
            )
            result.fail(f"Tracking failed with exit code {proc.returncode}")
            return result

        # Plots & animation (optional)
        if with_plots:
            run_plots_and_animation(demo, dir_demo, config_path)

    # Validate
    logger.info("Validating outputs ...")
    result = validate_demo(demo, dir_demo, check_plots=with_plots)
    result.elapsed_sec = time.time() - t0

    status = "PASSED" if result.passed else "FAILED"
    logger.info(f"  Result: {status}")
    for msg in result.messages:
        logger.info(f"    {msg}")

    return result


def print_summary(results: List[ValidationResult]):
    """Print a summary table of all results."""
    print("\n" + "=" * 72)
    print("  DEMO TEST SUMMARY")
    print("=" * 72)

    name_w = max(len(r.demo_name) for r in results) + 2
    print(f"  {'Demo':<{name_w}} {'Status':<10} {'Tracks':>7} {'Time':>10}  Notes")
    print(f"  {'-'*name_w} {'-'*10} {'-'*7} {'-'*10}  {'-'*30}")

    for r in results:
        status = "PASSED" if r.passed else "FAILED"
        elapsed = f"{r.elapsed_sec:.0f}s"
        fails = [m for m in r.messages if m.startswith("FAIL")]
        note = fails[0].replace("FAIL: ", "") if fails else ""
        # Truncate long notes
        if len(note) > 50:
            note = note[:47] + "..."
        print(
            f"  {r.demo_name:<{name_w}} {status:<10} {r.n_tracks:>7} "
            f"{elapsed:>10}  {note}"
        )

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\n  Total: {len(results)} demos — {n_pass} passed, {n_fail} failed")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Run PyFLEXTRKR end-to-end demo tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list
  %(prog)s --demos demo_mcs_imerg demo_cell_nexrad -n 4
  %(prog)s --all -n 8
  %(prog)s --demos demo_mcs_imerg --validate-only
  %(prog)s --demos demo_mcs_imerg --data-root /tmp/demo -n 4
  %(prog)s --demos demo_mcs_imerg --with-plots -n 4
""",
    )
    parser.add_argument(
        "--demos", nargs="+", metavar="NAME",
        help="Demo names to run (e.g. demo_mcs_imerg demo_cell_nexrad)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all available demos",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available demos and exit",
    )
    parser.add_argument(
        "-n", "--nworkers", type=int,
        default=min(os.cpu_count() or 4, 8),
        help="Number of parallel workers (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help=f"Root directory for demo data (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing outputs (skip download/tracking)",
    )
    parser.add_argument(
        "--with-plots", action="store_true",
        help="Run quicklook plots and create animations",
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Back up existing outputs to *.bak/ instead of deleting",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Re-download input data even if it already exists",
    )
    args = parser.parse_args()

    # ── List mode ──────────────────────────────────────────────────
    if args.list:
        print("\nAvailable demos:")
        print(f"  {'Name':<30} Description")
        print(f"  {'-'*30} {'-'*40}")
        for name, demo in DEMOS.items():
            print(f"  {name:<30} {demo.description}")
        print(f"\n  Total: {len(DEMOS)} demos\n")
        return

    # ── Select demos ───────────────────────────────────────────────
    if args.all:
        selected = list(DEMOS.keys())
    elif args.demos:
        for name in args.demos:
            if name not in DEMOS:
                print(f"Error: Unknown demo '{name}'")
                print(f"Available: {', '.join(DEMOS.keys())}")
                sys.exit(1)
        selected = args.demos
    else:
        parser.print_help()
        sys.exit(1)

    # ── Run ────────────────────────────────────────────────────────
    logger.info(f"Running {len(selected)} demo(s) with {args.nworkers} workers")
    logger.info(f"Data root: {args.data_root}")

    # Set PYFLEXTRKR_TEST_DATA so pytest regression tests can find the data
    os.environ["PYFLEXTRKR_TEST_DATA"] = str(args.data_root)

    results = []
    for name in selected:
        demo = DEMOS[name]
        try:
            r = run_single_demo(
                demo=demo,
                data_root=args.data_root,
                nworkers=args.nworkers,
                with_plots=args.with_plots,
                backup=args.backup,
                fresh=args.fresh,
                validate_only=args.validate_only,
            )
        except Exception as e:
            r = ValidationResult(demo_name=name)
            r.fail(f"Exception: {e}")
            logger.error(f"  Exception running {name}: {e}", exc_info=True)
        results.append(r)

    print_summary(results)

    # Exit with non-zero if any demo failed
    if any(not r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
