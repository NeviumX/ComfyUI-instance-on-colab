"""Linux root check: unprivileged application children retain shared-library lookup settings.

Uses a tiny temporary native library; no GPU, Google Drive, or network is used.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime

if sys.platform != "linux" or os.geteuid() != 0:
    raise SystemExit("Run this check as Linux root.")
compiler = shutil.which("gcc")
if compiler is None:
    raise SystemExit("gcc is required for this native-library check.")

with tempfile.TemporaryDirectory(prefix="comfy-gpu-env-check-") as temporary:
    root = Path(temporary)
    root.chmod(0o755)  # The application user must reach the library.
    runtime.commands.ensure_app_user()
    runtime.paths.BASE = root / "session"
    (runtime.paths.BASE / "home").mkdir(parents=True)
    libraries = root / "driver-libraries"
    libraries.mkdir()
    source = root / "driver_probe.c"
    source.write_text("int colab_driver_probe(void) { return 42; }\n")
    subprocess.run([compiler, "-shared", "-fPIC", "-o",
                    str(libraries / "libcolab_driver_probe.so"), str(source)], check=True)
    os.environ["LD_LIBRARY_PATH"] = str(libraries)
    os.environ["HF_TOKEN"] = "test-only-credential"
    os.environ["TBE_EPHEM_CREDS_ADDR"] = "test-only-address"
    child = '''import ctypes, os
assert os.geteuid() != 0
assert 'HF_TOKEN' not in os.environ and 'TBE_EPHEM_CREDS_ADDR' not in os.environ
library = ctypes.CDLL('libcolab_driver_probe.so')
assert library.colab_driver_probe() == 42
print('unprivileged driver-library=42')
'''
    previous = runtime.commands.application_environment()
    previous.pop("LD_LIBRARY_PATH", None)
    failure = subprocess.run(runtime.commands.app_command([sys.executable, "-c", child]), env=previous,
                             capture_output=True, text=True)
    assert failure.returncode != 0 and "libcolab_driver_probe.so" in failure.stderr
    success = runtime.commands.run([sys.executable, "-c", child], app=True, capture=True)
    assert "unprivileged driver-library=42" in success
    print("PASS: removing LD_LIBRARY_PATH reproduces library lookup failure")
    print("PASS: the unprivileged application loads the library with inherited GPU lookup settings")
    print("PASS: notebook credential variables are not passed to the application")