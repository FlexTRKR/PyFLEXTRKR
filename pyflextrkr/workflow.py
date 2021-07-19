import calendar
import os, yaml
import logging

class WorkflowManager(object):
    """ Workflow manager for FlexTRKR workflows. This class handles registering of the processing
    steps, tracking its config files, and coordinating the various processing steps. """


    def __init__(self, config_filename):

        self.workflow = {}
        self.datasets = {}
        self.config = load_config_and_paths(config_file=config_filename)

        # Need to register each of the processing steps

        # Set up the various lists of files

        return self

    def register_processing_step(self, input_dataset_name, step_number=-1):
        """ Register a processing step for the workflow

        Parameters:
        -----------
        input_dataset_name: string
            Input dataset needed for this processing step
        step_number: float
            Position to insert step at. Steps are run in numerical order. To insert a step between
            1 and 2, one can use 1.5, or 1.2 for instance.
        """
        pass

    def register_dataset(self, dataset_name, dataset_path, time_conversion_function):
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
        pass

    def unregister_processing_step(self, step_number):
        pass

    def run_workflow(self):
        pass

    def run_next_step(self):
        pass

    def run_step(self, step_number):
        pass


    def __repr__(self):
        pass

def load_config_and_paths(config_file = None):
    """ Load configuration file and set paths to the various files we will use. The preferred
    method for using this to set the environment variables, but you can pass a workflow config file as well.

    config_file: path
        Path to a config file. `None` uses environment variable FLEXTRKR_CONFIG_FILE which is preferred.
    """
    logger = logging.getLogger(__name__)
    if config_file_path == None:
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


    # TODO: JOE:  Move these into config reading part as well
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

    if not os.path.exists(config['tracking_outpath']):
        os.makedirs(config['tracking_outpath'])

    if not os.path.exists(config['stats_outpath']):
        os.makedirs(config['stats_outpath'])


    # Process time entries

    TEMP_starttime = datetime.datetime(
        int(config['startdate'][0:4]),
        int(config['startdate'][4:6]),
        int(config['startdate'][6:8]),
        int(config['startdate'][9:11]),
        int(config['startdate'][11:13]),
        0,
        tzinfo=utc,
    )
    config['start_basetime'] = calendar.timegm(TEMP_starttime.timetuple())

    TEMP_endtime = datetime.datetime(
        int(config['enddate'][0:4]),
        int(config['enddate'][4:6]),
        int(config['enddate'][6:8]),
        int(config['enddate'][9:11]),
        int(config['enddate'][11:13]),
        0,
        tzinfo=utc,
    )
    config['end_basetime'] = calendar.timegm(TEMP_endtime.timetuple()

    return config