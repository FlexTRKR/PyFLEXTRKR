# Configuration file for ipengine.

#------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
#------------------------------------------------------------------------------

## This is an application.

## The date format used by logging formatters for %(asctime)s
c.Application.log_datefmt = '%Y-%m-%d %H:%M:%S'

# The Logging format template
c.Application.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
c.Application.log_level = 100

#------------------------------------------------------------------------------
# BaseIPythonApplication(Application) configuration
#------------------------------------------------------------------------------

## IPython: an enhanced interactive Python shell.

## Whether to create profile dir if it doesn't exist
#c.BaseIPythonApplication.auto_create = False

## Whether to install the default config files into the profile dir. If a new
#  profile is being created, and IPython contains config files for that profile,
#  then they will be staged into the new directory.  Otherwise, default config
#  files will be automatically generated.
#c.BaseIPythonApplication.copy_config_files = False

## Path to an extra config file to load.
#  
#  If specified, load this config file in addition to any other IPython config.
#c.BaseIPythonApplication.extra_config_file = u''

## The name of the IPython directory. This directory is used for logging
#  configuration (through profiles), history storage, etc. The default is usually
#  $HOME/.ipython. This option can also be specified through the environment
#  variable IPYTHONDIR.
#c.BaseIPythonApplication.ipython_dir = u''

## Whether to overwrite existing config files when copying
#c.BaseIPythonApplication.overwrite = False

## The IPython profile to use.
#c.BaseIPythonApplication.profile = u'default'

## Create a massive crash report when IPython encounters what may be an internal
#  error.  The default is to append a short message to the usual traceback
#c.BaseIPythonApplication.verbose_crash = False

#------------------------------------------------------------------------------
# BaseParallelApplication(BaseIPythonApplication) configuration
#------------------------------------------------------------------------------

## IPython: an enhanced interactive Python shell.

## whether to cleanup old logfiles before starting
#c.BaseParallelApplication.clean_logs = False

## String id to add to runtime files, to prevent name collisions when using
#  multiple clusters with a single profile simultaneously.
#  
#  When set, files will be named like: 'ipcontroller-<cluster_id>-engine.json'
#  
#  Since this is text inserted into filenames, typical recommendations apply:
#  Simple character strings are ideal, and spaces are not recommended (but should
#  generally work).
#c.BaseParallelApplication.cluster_id = ''

## whether to log to a file
#c.BaseParallelApplication.log_to_file = False

## The ZMQ URL of the iplogger to aggregate logging.
#c.BaseParallelApplication.log_url = ''

## Set the working dir for the process.
#c.BaseParallelApplication.work_dir = u'/global/u1/h/hcbarnes'

#------------------------------------------------------------------------------
# IPEngineApp(BaseParallelApplication) configuration
#------------------------------------------------------------------------------

## The URL for the iploggerapp instance, for forwarding logging to a central
#  location.
#c.IPEngineApp.log_url = ''

## specify a command to be run at startup
#c.IPEngineApp.startup_command = ''

## specify a script to be run at startup
#c.IPEngineApp.startup_script = u''

## The full location of the file containing the connection information for the
#  controller. If this is not given, the file must be in the security directory
#  of the cluster directory.  This location is resolved using the `profile` or
#  `profile_dir` options.
#c.IPEngineApp.url_file = u''

## 
#c.IPEngineApp.url_file_name = u'ipcontroller-engine.json'

## The maximum number of seconds to wait for url_file to exist. This is useful
#  for batch-systems and shared-filesystems where the controller and engine are
#  started at the same time and it may take a moment for the controller to write
#  the connector files.
#c.IPEngineApp.wait_for_url_file = 10

#------------------------------------------------------------------------------
# InteractiveShell(SingletonConfigurable) configuration
#------------------------------------------------------------------------------

## An enhanced, interactive shell for Python.

## 'all', 'last', 'last_expr' or 'none', specifying which nodes should be run
#  interactively (displaying output from expressions).
#c.InteractiveShell.ast_node_interactivity = 'last_expr'

## A list of ast.NodeTransformer subclass instances, which will be applied to
#  user input before code is run.
#c.InteractiveShell.ast_transformers = []

## Make IPython automatically call any callable object even if you didn't type
#  explicit parentheses. For example, 'str 43' becomes 'str(43)' automatically.
#  The value can be '0' to disable the feature, '1' for 'smart' autocall, where
#  it is not applied if there are no more arguments on the line, and '2' for
#  'full' autocall, where all callable objects are automatically called (even if
#  no arguments are present).
#c.InteractiveShell.autocall = 0

## Autoindent IPython code entered interactively.
#c.InteractiveShell.autoindent = True

## Enable magic commands to be called without the leading %.
#c.InteractiveShell.automagic = True

## The part of the banner to be printed before the profile
#c.InteractiveShell.banner1 = 'Python 2.7.13 |Continuum Analytics, Inc.| (default, Dec 20 2016, 23:09:15) \nType "copyright", "credits" or "license" for more information.\n\nIPython 5.3.0 -- An enhanced Interactive Python.\n?         -> Introduction and overview of IPython\'s features.\n%quickref -> Quick reference.\nhelp      -> Python\'s own help system.\nobject?   -> Details about \'object\', use \'object??\' for extra details.\n'

## The part of the banner to be printed after the profile
#c.InteractiveShell.banner2 = ''

## Set the size of the output cache.  The default is 1000, you can change it
#  permanently in your config file.  Setting it to 0 completely disables the
#  caching system, and the minimum value accepted is 20 (if you provide a value
#  less than 20, it is reset to 0 and a warning is issued).  This limit is
#  defined because otherwise you'll spend more time re-flushing a too small cache
#  than working
#c.InteractiveShell.cache_size = 1000

## Use colors for displaying information about objects. Because this information
#  is passed through a pager (like 'less'), and some pagers get confused with
#  color codes, this capability can be turned off.
#c.InteractiveShell.color_info = True

## Set the color scheme (NoColor, Neutral, Linux, or LightBG).
#c.InteractiveShell.colors = 'Neutral'

## 
#c.InteractiveShell.debug = False

## **Deprecated**
#  
#  Will be removed in IPython 6.0
#  
#  Enable deep (recursive) reloading by default. IPython can use the deep_reload
#  module which reloads changes in modules recursively (it replaces the reload()
#  function, so you don't need to change anything to use it). `deep_reload`
#  forces a full reload of modules whose code may have changed, which the default
#  reload() function does not.  When deep_reload is off, IPython will use the
#  normal reload(), but deep_reload will still be available as dreload().
#c.InteractiveShell.deep_reload = False

## Don't call post-execute functions that have failed in the past.
#c.InteractiveShell.disable_failing_post_execute = False

## If True, anything that would be passed to the pager will be displayed as
#  regular output instead.
#c.InteractiveShell.display_page = False

## (Provisional API) enables html representation in mime bundles sent to pagers.
#c.InteractiveShell.enable_html_pager = False

## Total length of command history
#c.InteractiveShell.history_length = 10000

## The number of saved history entries to be loaded into the history buffer at
#  startup.
#c.InteractiveShell.history_load_length = 1000

## 
#c.InteractiveShell.ipython_dir = ''

## Start logging to the given file in append mode. Use `logfile` to specify a log
#  file to **overwrite** logs to.
#c.InteractiveShell.logappend = ''

## The name of the logfile to use.
#c.InteractiveShell.logfile = ''

## Start logging to the default log file in overwrite mode. Use `logappend` to
#  specify a log file to **append** logs to.
#c.InteractiveShell.logstart = False

## 
#c.InteractiveShell.object_info_string_level = 0

## Automatically call the pdb debugger after every exception.
#c.InteractiveShell.pdb = False

## Deprecated since IPython 4.0 and ignored since 5.0, set
#  TerminalInteractiveShell.prompts object directly.
#c.InteractiveShell.prompt_in1 = 'In [\\#]: '

## Deprecated since IPython 4.0 and ignored since 5.0, set
#  TerminalInteractiveShell.prompts object directly.
#c.InteractiveShell.prompt_in2 = '   .\\D.: '

## Deprecated since IPython 4.0 and ignored since 5.0, set
#  TerminalInteractiveShell.prompts object directly.
#c.InteractiveShell.prompt_out = 'Out[\\#]: '

## Deprecated since IPython 4.0 and ignored since 5.0, set
#  TerminalInteractiveShell.prompts object directly.
#c.InteractiveShell.prompts_pad_left = True

## 
#c.InteractiveShell.quiet = False

## 
#c.InteractiveShell.separate_in = '\n'

## 
#c.InteractiveShell.separate_out = ''

## 
#c.InteractiveShell.separate_out2 = ''

## Show rewritten input, e.g. for autocall.
#c.InteractiveShell.show_rewritten_input = True

## Enables rich html representation of docstrings. (This requires the docrepr
#  module).
#c.InteractiveShell.sphinxify_docstring = False

## 
#c.InteractiveShell.wildcards_case_sensitive = True

## 
#c.InteractiveShell.xmode = 'Context'

#------------------------------------------------------------------------------
# ZMQInteractiveShell(InteractiveShell) configuration
#------------------------------------------------------------------------------

## A subclass of InteractiveShell for ZMQ.

#------------------------------------------------------------------------------
# ProfileDir(LoggingConfigurable) configuration
#------------------------------------------------------------------------------

## An object to manage the profile directory and its resources.
#  
#  The profile directory is used by all IPython applications, to manage
#  configuration, logging and security.
#  
#  This object knows how to find, create and manage these directories. This
#  should be used by any code that wants to handle profiles.

## Set the profile location directly. This overrides the logic used by the
#  `profile` option.
#c.ProfileDir.location = u''

#------------------------------------------------------------------------------
# Session(Configurable) configuration
#------------------------------------------------------------------------------

## Object for handling serialization and sending of messages.
#  
#  The Session object handles building messages and sending them with ZMQ sockets
#  or ZMQStream objects.  Objects can communicate with each other over the
#  network via Session objects, and only need to work with the dict-based IPython
#  message spec. The Session will handle serialization/deserialization, security,
#  and metadata.
#  
#  Sessions support configurable serialization via packer/unpacker traits, and
#  signing with HMAC digests via the key/keyfile traits.
#  
#  Parameters ----------
#  
#  debug : bool
#      whether to trigger extra debugging statements
#  packer/unpacker : str : 'json', 'pickle' or import_string
#      importstrings for methods to serialize message parts.  If just
#      'json' or 'pickle', predefined JSON and pickle packers will be used.
#      Otherwise, the entire importstring must be used.
#  
#      The functions must accept at least valid JSON input, and output *bytes*.
#  
#      For example, to use msgpack:
#      packer = 'msgpack.packb', unpacker='msgpack.unpackb'
#  pack/unpack : callables
#      You can also set the pack/unpack callables for serialization directly.
#  session : bytes
#      the ID of this Session object.  The default is to generate a new UUID.
#  username : unicode
#      username added to message headers.  The default is to ask the OS.
#  key : bytes
#      The key used to initialize an HMAC signature.  If unset, messages
#      will not be signed or checked.
#  keyfile : filepath
#      The file containing a key.  If this is set, `key` will be initialized
#      to the contents of the file.

## Threshold (in bytes) beyond which an object's buffer should be extracted to
#  avoid pickling.
#c.Session.buffer_threshold = 1024

## Whether to check PID to protect against calls after fork.
#  
#  This check can be disabled if fork-safety is handled elsewhere.
#c.Session.check_pid = True

## Threshold (in bytes) beyond which a buffer should be sent without copying.
#c.Session.copy_threshold = 65536

## Debug output in the Session
#c.Session.debug = False

## The maximum number of digests to remember.
#  
#  The digest history will be culled when it exceeds this value.
#c.Session.digest_history_size = 65536

## The maximum number of items for a container to be introspected for custom
#  serialization. Containers larger than this are pickled outright.
#c.Session.item_threshold = 64

## execution key, for signing messages.
#c.Session.key = ''

## path to file containing execution key.
#c.Session.keyfile = ''

## Metadata dictionary, which serves as the default top-level metadata dict for
#  each message.
#c.Session.metadata = {}

## The name of the packer for serializing messages. Should be one of 'json',
#  'pickle', or an import name for a custom callable serializer.
#c.Session.packer = 'json'

## The UUID identifying this session.
#c.Session.session = u''

## The digest scheme used to construct the message signatures. Must have the form
#  'hmac-HASH'.
#c.Session.signature_scheme = 'hmac-sha256'

## The name of the unpacker for unserializing messages. Only used with custom
#  functions for `packer`.
#c.Session.unpacker = 'json'

## Username for the Session. Default is your system username.
#c.Session.username = u'hcbarnes'

#------------------------------------------------------------------------------
# RegistrationFactory(SessionFactory) configuration
#------------------------------------------------------------------------------

## The Base Configurable for objects that involve registration.

## The IP address for registration.  This is generally either '127.0.0.1' for
#  loopback only or '*' for all interfaces.
#c.RegistrationFactory.ip = u''

## The port on which the Hub listens for registration.
#c.RegistrationFactory.regport = 0

## The 0MQ transport for communications.  This will likely be the default of
#  'tcp', but other values include 'ipc', 'epgm', 'inproc'.
#c.RegistrationFactory.transport = 'tcp'

## The 0MQ url used for registration. This sets transport, ip, and port in one
#  variable. For example: url='tcp://127.0.0.1:12345' or url='epgm://*:90210'
#c.RegistrationFactory.url = ''

#------------------------------------------------------------------------------
# EngineFactory(RegistrationFactory) configuration
#------------------------------------------------------------------------------

## IPython engine

## The class for handling displayhook. Typically
#  'ipykernel.displayhook.ZMQDisplayHook'
#c.EngineFactory.display_hook_factory = 'ipykernel.displayhook.ZMQDisplayHook'

## The location (an IP address) of the controller.  This is used for
#  disambiguating URLs, to determine whether loopback should be used to connect
#  or the public address.
#c.EngineFactory.location = u''

## The maximum number of times a check for the heartbeat ping of a  controller
#  can be missed before shutting down the engine.
#  
#  If set to 0, the check is disabled.
#c.EngineFactory.max_heartbeat_misses = 50

## The OutStream for handling stdout/err. Typically
#  'ipykernel.iostream.OutStream'
#c.EngineFactory.out_stream_factory = 'ipykernel.iostream.OutStream'

## Whether to use paramiko instead of openssh for tunnels.
#c.EngineFactory.paramiko = False

## The SSH private key file to use when tunneling connections to the Controller.
#c.EngineFactory.sshkey = u''

## The SSH server to use for tunneling connections to the Controller.
#c.EngineFactory.sshserver = u''

## The time (in seconds) to wait for the Controller to respond to registration
#  requests before giving up.
#c.EngineFactory.timeout = 5.0

#------------------------------------------------------------------------------
# Kernel(SingletonConfigurable) configuration
#------------------------------------------------------------------------------

## Whether to use appnope for compatiblity with OS X App Nap.
#  
#  Only affects OS X >= 10.9.
#c.Kernel._darwin_app_nap = True

## 
#c.Kernel._execute_sleep = 0.0005

## 
#c.Kernel._poll_interval = 0.05

#------------------------------------------------------------------------------
# IPythonKernel(Kernel) configuration
#------------------------------------------------------------------------------

## 
#c.IPythonKernel.help_links = [{'url': 'http://docs.python.org/2.7', 'text': 'Python'}, {'url': 'http://ipython.org/documentation.html', 'text': 'IPython'}, {'url': 'http://docs.scipy.org/doc/numpy/reference/', 'text': 'NumPy'}, {'url': 'http://docs.scipy.org/doc/scipy/reference/', 'text': 'SciPy'}, {'url': 'http://matplotlib.org/contents.html', 'text': 'Matplotlib'}, {'url': 'http://docs.sympy.org/latest/index.html', 'text': 'SymPy'}, {'url': 'http://pandas.pydata.org/pandas-docs/stable/', 'text': 'pandas'}]

#------------------------------------------------------------------------------
# MPI(Configurable) configuration
#------------------------------------------------------------------------------

## Configurable for MPI initialization

## 
#c.MPI.default_inits = {'pytrilinos': 'from PyTrilinos import Epetra\nclass SimpleStruct:\npass\nmpi = SimpleStruct()\nmpi.rank = 0\nmpi.size = 0\n', 'mpi4py': 'from mpi4py import MPI as mpi\nmpi.size = mpi.COMM_WORLD.Get_size()\nmpi.rank = mpi.COMM_WORLD.Get_rank()\n'}

## Initialization code for MPI
#c.MPI.init_script = ''

## How to enable MPI (mpi4py, pytrilinos, or empty string to disable).
#c.MPI.use = ''
