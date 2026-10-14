"""The code behind the ComfyUI instance on Colab notebooks.

The notebook's cell 2 writes this package to the Colab machine and imports it; the names below
are what the cells call. Everything runs as root in the notebook's kernel except what the
application user runs: ComfyUI, pip and custom nodes (commands.app_command) and the file
operations in its tree (app_tree.app_action).

Modules, roughly in the order a runtime uses them:
  paths         where things are kept (BASE: the application user's; PRIVATE: root only)
  messages      t(): the texts shown, in the notebook's language
  filesystem    small file helpers
  commands      running commands, as root or as the application user
  app_tree      the application user's tree and the file operations in it
  dependencies  apt, the virtual environment, Python packages, SageAttention
  drive         Google Drive and its models folder
  environment   saving the environment to Drive in parts, and restoring it
  installation  setup (cell 3) and updating ComfyUI (cell 11)
  downloads     model downloads (cell 4)
  cloudflare    cloudflared, the named tunnel and Cloudflare Access (6-3)
  pairing       pairing codes for the public URL (6-2)
  proxy         the proxy in front of ComfyUI
  stop_button   the stop button on ComfyUI's page
  service       starting and stopping (cells 6 and 10)
  diagnosis     connection diagnosis (cell 7)
  __main__      the processes the package starts of itself (python -m)

The modules refer to each other as module.name (paths.BASE, commands.run): the app action and
the tests replace such names on their module, which reaches every caller.
"""

# Every module, so that runtime.<module> is there for whoever has the package.
from . import (app_tree, cloudflare, commands, dependencies, diagnosis, downloads, drive, environment,  # noqa: F401
               filesystem, installation, messages, pairing, paths, proxy, service, stop_button)
from .downloads import fetch_model, model_token_name
from .drive import hide_drive_models, sync_models
from .environment import checkpoint
from .installation import setup, update_comfyui
from .pairing import issue_pairing_code
from .service import load_config, redact, run_diagnosis, start, start_with_access, stop, zip_outputs

__all__ = ["checkpoint", "fetch_model", "hide_drive_models", "issue_pairing_code", "load_config", "model_token_name",
           "redact", "run_diagnosis", "setup", "start", "start_with_access", "stop", "sync_models", "update_comfyui",
           "zip_outputs"]
