import calendar
import logging
import os
import yaml

from pytz import utc


class WorkflowManager(object):
    """ Workflow manager for FlexTRKR workflows. This class handles registering of the processing
    steps, tracking its config files, and coordinating the various processing steps. """


    def __init__(self, config_filename):

        self.workflow = {}
        self.datasets = {}
        self.config = load_config_and_paths(config_file=config_filename)

        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
        self.logger = logging.getLogger(__name__)


        # First we register our various datasets. We don't register an input dataset, but all intermediate defaults are registered.
        # 1. Output of step 1
        # 2. Stats file
        # 3. idv files
        # 4.

        # Need to register each of the processing steps
        # We'll register a default set of processing steps based on config file




    def register_processing_step(self, input_dataset_name, step_number=-1, enabled=True):
        """ Register a processing step for the workflow

        Parameters:
        -----------
        input_dataset_name: string
            Input dataset needed for this processing step
        step_number: float, -1 default
            Position to insert step at. Steps are run in numerical order. To insert a step between
            1 and 2, one can use 1.5, or 1.2 for instance. -1 adds after last step.

        Note: All processing steps are given the config dictionary and so if more esoteric processing is needed
        that can be handled within the function.
        """
        pass

    def register_dataset(self, dataset_name, dataset_path, time_conversion_function=None):
        """ Register a dataset for availability to processing steps.

        Parameters:
        -----------
        dataset_name: string
            Name to register dataset under.
        dataset_path: string
            Path to dataset
        time_conversion_function: function
            Function that maps dataset filenames to times (can be used to filter out files). In the case of statistics
            files this can just be an idempotent mapping.
        """
        # Steps of processing
        # 1. test if single file or filelist
        # 2. If single file, add to list of datasets
        # 3. If directory then pass list of files through time_conversion function, check they are within limits and sort them then store.
        pass

    def unregister_processing_step(self, step_number):
        pass

    def run_workflow(self):
        pass

    def run_next_step(self):
        pass

    def run_step(self, step_number):
        pass

    def change_enabled_state_of_processing_step(self, step_number, new_state):
        """ Given a processing step_number, enable or disable it."""


    def __repr__(self):
        pass

def load_config_and_paths(config_file = None):
    """ Load configuration file and set paths to the various files we will use. The preferred
    method for using this to set the environment variables, but you can pass a workflow config file as well.

    config_file: path
        Path to a config file. `None` uses environment variable FLEXTRKR_CONFIG_FILE which is preferred.
    """
    logger = logging.getLogger(__name__)
    if config_file is None:
        config_path = os.environ.get("FLEXTRKR_CONFIG_FILE", "config/")
        config_filename = os.environ.get("FLEXTRKR_CONFIG_FILE", "config/global_gpm_mcs_workflow_config.yml")
    else:
        config_filename = config_file
        config_path = os.path.dirname(os.path.realpath(config_file))

    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)

    # Load the processing configuration file and append it to our dictionary.
    processing_config = yaml.load(open(config_path + config['processing_config']), Loader=yaml.FullLoader)
    config.update(processing_config)

    # Set up paths

    root_path = os.environ['FLEXTRKR_BASE_DATA_PATH']
    logger.info(f'ROOT DATA PATH IS {root_path}')


    config['clouddata_path'] = root_path + config['input_data_directory']
    config['pfdata_path'] = root_path + config['input_data_directory']

    # Specify additional file locations
    config['tracking_outpath'] = root_path + "tracking/"  # Data on individual features being tracked

    config['stats_outpath'] = root_path + "stats/"  # Data on track statistics
    config['mcstracking_outpath'] = root_path + "mcstracking/" + startdate + "_" + enddate + "/"# Pixel level data for MCSs
    logger.debug(
        f"Tracking path is {tracking_outpath}, {stats_outpath}, {mcstracking_outpath}"
    )

    logger.debug(f'Clouddatapath: {clouddata_path}, pfdata_path: {pfdata_path}')

    config['rainaccumulation_path'] = pfdata_path
    config['landmask_file'] = root_path + 'map/' + config['landmask_filename']
    # landmask_file = root_path + "map_data/IMERG_landmask_saag.nc"

    # TODO: JOE: Move this to the register dataset portion
    if not os.path.exists(config['tracking_outpath']):
        os.makedirs(config['tracking_outpath'])

    if not os.path.exists(config['stats_outpath']):
        os.makedirs(config['stats_outpath'])

    config['cloudtb_threshs'] = np.hstack(
        (config['cloudtb_core'], config['cloudtb_cold'], config['cloudtb_warm'], config['cloudtb_cloud'])
    )

    if 'absolutetb_threshs' not in config:
        config['absolute_tb_threshs'] =  np.array([160, 330])



    # Process time entries

    temp_starttime = datetime.datetime(
        int(config['startdate'][0:4]),
        int(config['startdate'][4:6]),
        int(config['startdate'][6:8]),
        int(config['startdate'][9:11]),
        int(config['startdate'][11:13]),
        0,
        tzinfo=utc,
    )
    config['start_basetime'] = calendar.timegm(temp_starttime.timetuple())

    temp_endtime = datetime.datetime(
        int(config['enddate'][0:4]),
        int(config['enddate'][4:6]),
        int(config['enddate'][6:8]),
        int(config['enddate'][9:11]),
        int(config['enddate'][11:13]),
        0,
        tzinfo=utc,
    )
    config['end_basetime'] = calendar.timegm(temp_endtime.timetuple()

    return config