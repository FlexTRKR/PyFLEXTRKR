from pyflextrkr import workflow_manager
import calendar
import datetime
from pytz import utc


def parse_raw_filenames(filename, data_basename='merg_'):
    """ Parse raw GPM filenames into a datetime object."""

    # Loop through files, identifying files within the startdate - enddate interval
    nleadingchar = len(data_basename)
    fnmatch.filter(os.listdir(clouddata_path), databasename + "*")

    #TODO: JOE: This is ugly and should be redone with strptime.
    filetime = datetime.datetime(
        int(filename[nleadingchar: nleadingchar + 4]),
        int(filename[nleadingchar + 4: nleadingchar + 6]),
        int(filename[nleadingchar + 6: nleadingchar + 8]),
        int(filename[nleadingchar + 8: nleadingchar + 10]),
        0,
        0,
        tzinfo=utc,
    )
        return calendar.timegm(filetime.timetuple())


if __name__ == '__main__':
    workflow = workflow_manager.WorkflowManager(config_filename='../config/global_gpm_mcs_workflow_config.yml')

    # Let's register our starting dataset. Other intermediate ones are registered by default.
    workflow.register_dataset("raw_clouddata", config['clouddata_path'], time_conversion_function=parse_raw_filenames)


    print(workflow)
    workflow.run_workflow()