from .config import get_key_path, init_config

init_config()

from .mjconstants import *
from .mjcore import MjModel, register_license
from .mjviewer import MjViewer
from .platname_targdir import targdir

register_license(get_key_path())
