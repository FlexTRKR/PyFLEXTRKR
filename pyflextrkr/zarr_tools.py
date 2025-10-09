"""
Dask & Zarr Tools

Author: Zhe Feng | zhe.feng@pnnl.gov
"""

import os
import shutil
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from dask.distributed import progress
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

def setup_dask_client(parallel, n_workers, threads_per_worker, memory_limit=None, 
                      dashboard_address=':8787', scheduler_port=0, processes=True,
                      memory_target_fraction=0.8, memory_spill_fraction=0.85, 
                      memory_pause_fraction=0.9, memory_terminate_fraction=0.95,
                      tcp_timeout='300s', client_heartbeat='10s', logger=None):
    """
    Set up a Dask client optimized for HPC hardware with comprehensive configuration options
    
    Args:
        parallel: bool
            Whether to use parallel processing
        n_workers: int
            Number of workers for the Dask cluster
        threads_per_worker: int
            Number of threads per worker
        memory_limit: str, optional
            Memory limit per worker (e.g., "60GB"). If None, auto-calculated from system memory
        dashboard_address: str, optional
            Dashboard address (default: ':8787')
        scheduler_port: int, optional
            Scheduler port (default: 0 for auto-selection)
        processes: bool, optional
            Use processes for better memory isolation (default: True)
        memory_target_fraction: float, optional
            Target memory fraction before spilling to disk (default: 0.8)
        memory_spill_fraction: float, optional
            Memory fraction at which to start spilling to disk (default: 0.85)
        memory_pause_fraction: float, optional
            Memory fraction at which to pause worker (default: 0.9)
        memory_terminate_fraction: float, optional
            Memory fraction at which to terminate worker (default: 0.95)
        tcp_timeout: str, optional
            TCP timeout for distributed communication (default: '300s')
        client_heartbeat: str, optional
            Client heartbeat interval (default: '10s')
        logger: logging.Logger, optional
            Logger for status messages
            
    Returns:
        dask.distributed.Client or None: Dask client if parallel is True, None otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if not parallel:
        logger.info("Running in sequential mode (parallel=False)")
        return None
    
    try:
        from dask.distributed import Client, LocalCluster
        import dask
        import psutil
        
        # Auto-calculate memory limit per worker if not provided
        if memory_limit is None:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)  # Convert to GB
            # Leave 20% for system overhead, divide remaining by number of workers
            usable_memory_gb = total_memory_gb * 0.8
            memory_per_worker_gb = usable_memory_gb / n_workers
            memory_limit = f"{memory_per_worker_gb:.1f}GB"
            logger.info(f"Auto-calculated memory per worker: {memory_limit} "
                       f"(from {total_memory_gb:.1f}GB total system memory)")
        
        logger.info(f"Setting up Dask cluster optimized for HPC hardware")
        logger.info(f"Workers: {n_workers}, Threads per worker: {threads_per_worker}")
        logger.info(f"Memory per worker: {memory_limit}")
        logger.info(f"Memory management: target={memory_target_fraction}, "
                   f"spill={memory_spill_fraction}, pause={memory_pause_fraction}")
        
        # Set up environment variables for NUMA-aware computation (commented out by default)
        # import os
        # os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
        # os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
        # os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
        # os.environ['NUMBA_NUM_THREADS'] = str(threads_per_worker)
        
        # Configure Dask settings using the new config-based approach
        dask.config.set({
            'distributed.worker.memory.target': memory_target_fraction,
            'distributed.worker.memory.spill': memory_spill_fraction,
            'distributed.worker.memory.pause': memory_pause_fraction,
            'distributed.worker.memory.terminate': memory_terminate_fraction,
            'distributed.comm.timeouts.tcp': tcp_timeout,
            'distributed.client.heartbeat': client_heartbeat,
            'distributed.worker.daemon': False,
        })
        
        # Create cluster without deprecated memory parameters
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=processes,  # Use processes for better memory isolation
            scheduler_port=scheduler_port,
            dashboard_address=dashboard_address,
            silence_logs=False,  # Keep logs for debugging
        )
        client = Client(cluster)
        
        logger.info(f"Dask dashboard: {client.dashboard_link}")
        
        return client
        
    except ImportError:
        logger.warning("Dask not available, falling back to sequential processing")
        return None


def write_zarr(ds, out_zarr, client=None, logger=None, chunksize_time=24, chunksize_cell=None):
    """
    Write dataset to Zarr with optimized chunking for HEALPix grid.
    
    Args:
        ds: xarray.Dataset
            Dataset to write
        out_zarr: str
            Output Zarr store path
        client: dask.distributed.Client, optional
            Dask client for distributed computation
        logger: logging.Logger, optional
            Logger for status messages
        chunksize_time: int, optional
            Time chunk size (default: 24)
        chunksize_cell: int, optional
            Cell chunk size. If None, calculated from HEALPix nside
            
    Returns:
        None
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Remove existing zarr store if it exists
    if os.path.exists(out_zarr):
        logger.info(f"Removing existing zarr store: {out_zarr}")
        shutil.rmtree(out_zarr)

    # Optimize cell chunking for HEALPix grid if not provided
    if chunksize_cell is None:
        if hasattr(ds, 'crs') and 'healpix_nside' in ds.crs.attrs:
            zoom_level = zoom_level_from_nside(ds.crs.attrs['healpix_nside'])
            chunksize_cell = 12 * 4**zoom_level
        else:
            # Default cell chunking for non-HEALPix data
            chunksize_cell = min(10000, ds.sizes.get('cell', ds.sizes.get('ncol', 1000)))
    
    # Make time chunks more even if needed
    if isinstance(chunksize_time, (int, float)) and chunksize_time != 'auto':
        total_times = ds.sizes['time']
        chunks = total_times // chunksize_time
        if chunks * chunksize_time < total_times:
            # We have a remainder - try to make chunks more even
            if total_times % chunks == 0:
                chunksize_time = total_times // chunks
            elif total_times % (chunks + 1) == 0:
                chunksize_time = total_times // (chunks + 1)
    
    # Determine spatial dimension name
    spatial_dim = None
    for dim_name in ['cell', 'ncol']:
        if dim_name in ds.dims:
            spatial_dim = dim_name
            break
    
    if spatial_dim is None:
        raise ValueError("Could not find spatial dimension ('cell' or 'ncol') in dataset")
    
    # Clear any existing encoding that might conflict with new chunking
    # This is critical when rechunking data that was previously written to zarr
    for var in ds.data_vars:
        if 'chunks' in ds[var].encoding:
            del ds[var].encoding['chunks']
        if 'preferred_chunks' in ds[var].encoding:
            del ds[var].encoding['preferred_chunks']
    
    # Set proper chunking
    chunk_dict = {
        "time": chunksize_time, 
        spatial_dim: chunksize_cell, 
    }
    chunked_ds = ds.chunk(chunk_dict)
    
    # Report dataset size and chunking info
    logger.info(f"Output dataset dimensions: {dict(chunked_ds.sizes)}")
    logger.info(f"Output chunking scheme: time={chunksize_time}, {spatial_dim}={chunksize_cell}")

    # ---------- WRITE ZARR OUTPUT ----------
    logger.info(f"Starting Zarr write to: {out_zarr}")
    
    # Create a delayed task for Zarr writing
    write_task = chunked_ds.to_zarr(
        out_zarr,
        mode="w",
        consolidated=True,  # Enable for better performance when reading
        compute=False      # Create a delayed task
    )
    
    # Compute the task, with progress reporting
    if client:
        try:
            # Temporarily suppress distributed.shuffle logs during progress display
            shuffle_logger = logging.getLogger('distributed.shuffle')
            original_level = shuffle_logger.level
            shuffle_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
            
            # Get cluster state information before processing
            try:
                if PSUTIL_AVAILABLE:
                    memory_usage = client.run(lambda: psutil.Process().memory_info().rss / 1e9)
                    logger.info(f"Current memory usage across workers (GB): {memory_usage}")
            except Exception as e:
                logger.warning(f"Could not get memory usage: {e}")
                   
            try:
                # Compute with progress tracking
                future = client.compute(write_task)
                logger.info("Writing Zarr (this may take a while)...")
                if DASK_AVAILABLE:
                    progress(future)  # Shows a progress bar in notebooks or detailed progress in terminals

                result = future.result()
                logger.info("Zarr write completed successfully")
            except Exception as e:
                logger.error(f"Zarr write failed: {str(e)}")
                raise
            finally:
                # Restore original log level
                shuffle_logger.setLevel(original_level)
        except Exception as e:
            if not DASK_AVAILABLE:
                logger.warning("Dask distributed components not available, computing locally")
            else:
                logger.error(f"Error during distributed computation: {e}")
            write_task.compute()
    else:
        # Compute locally if no client
        logger.info("Computing zarr write locally...")
        write_task.compute()

    logger.info(f"Zarr file complete: {out_zarr}")


def write_zarr_simple(ds, out_zarr, logger=None):
    """
    Simple zarr write function without Dask optimization.
    
    Args:
        ds: xarray.Dataset
            Dataset to write
        out_zarr: str
            Output Zarr store path
        logger: logging.Logger, optional
            Logger for status messages
            
    Returns:
        None
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Remove existing zarr store if it exists
    if os.path.exists(out_zarr):
        logger.info(f"Removing existing zarr store: {out_zarr}")
        shutil.rmtree(out_zarr)
    
    logger.info(f"Writing zarr to: {out_zarr}")
    ds.to_zarr(out_zarr, mode='w', consolidated=True)
    logger.info(f"Zarr write completed: {out_zarr}")


