"""Build the Japanese and English Colab notebooks from the runtime package and translations.json.

translations.json holds every text a reader sees, in both languages:
  "notebook"  the text of each cell. A code cell is section -> key -> {"JP": ..., "EN": ...}
              (its title, form descriptions and printed messages); a markdown cell is
              section -> {"JP": [lines], "EN": [lines]}.
  "runtime"   the messages of the runtime package, as section -> key -> {"JP": ..., "EN": ...}.
This file holds the code of the cells and refers to their texts by "section.key".
The runtime package does the same with t("section.key", value=...); the copy embedded in a
notebook carries the messages of that notebook's language (in messages.py), so it needs no
other file. Cell 2 writes the package to PACKAGE on the Colab machine and imports it.
"""
import ast
import json
from pathlib import Path, PurePosixPath
import re
import string
import textwrap

HERE = Path(__file__).resolve().parent
NAME = "ComfyUI_instance_on_Colab"
TRANSLATIONS = HERE / "translations.json"
LANGUAGES = ("JP", "EN")
JAPANESE = re.compile(r"[　-ヿ一-鿿＀-￯]")
MESSAGES_LINE = re.compile(r"(?m)^MESSAGES = None\b.*$")
RUNTIME = HERE / "runtime"
# Where cell 2 puts the package. Its parent must be a folder only root can write: the processes
# the package starts of itself run there (commands.own_command).
PACKAGE = "/content/comfy_colab_runtime"
LOAD_CONFIG = "config = runtime.load_config()"


def translations():
    return json.loads(TRANSLATIONS.read_text(encoding="utf-8"))


def runtime_messages(language, sections=None):
    """The messages of the runtime package in one language, as {"section.key": text}."""
    sections = translations()["runtime"] if sections is None else sections
    return {f"{section}.{name}": entry[language]
            for section, entries in sections.items() for name, entry in entries.items()}


def shown_values(text):
    """(name, format, conversion) of every value a message shows."""
    return {(name, spec, conversion) for _, name, spec, conversion in string.Formatter().parse(text)
            if name is not None}


def runtime_sources():
    """The files of the runtime package, as {file name: source}."""
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(RUNTIME.glob("*.py"))}


def check_runtime(sources, sections):
    """Refuse a runtime and messages that do not fit each other. sources: {file name: source}."""
    tables = {language: runtime_messages(language, sections) for language in LANGUAGES}
    japanese, english = (tables[language] for language in LANGUAGES)
    used = set()
    for file, source in sources.items():
        where = f"runtime/{file}"
        if JAPANESE.search(source):
            raise ValueError(f"{where} holds Japanese text: put it in {TRANSLATIONS.name}")
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t"):
                continue
            if not (len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str) and all(keyword.arg for keyword in node.keywords)):
                raise ValueError(f"{where} line {node.lineno}: t() takes a literal key and named values")
            key = node.args[0].value
            if key not in japanese:
                raise KeyError(f"{TRANSLATIONS.name} has no runtime message {key} ({where} line {node.lineno})")
            used.add(key)
            given = {keyword.arg for keyword in node.keywords}
            if {name for name, _, _ in shown_values(japanese[key])} != given:
                raise ValueError(f"runtime message {key} does not show exactly the values given to it: "
                                 + ", ".join(sorted(given)) + f" ({where} line {node.lineno})")
    for key in japanese:
        if key not in used:
            raise ValueError(f"{TRANSLATIONS.name} has a runtime message the runtime package does not use: {key}")
        # A sentence may order its values differently in another language, but it must
        # show the same values, formatted the same way.
        if shown_values(japanese[key]) != shown_values(english[key]):
            raise ValueError(f"runtime message {key} shows different values in the two languages")
        if JAPANESE.search(english[key]):
            raise ValueError(f"runtime message {key} has Japanese in its English text")


def runtime_files(language):
    """The runtime package as a notebook embeds it: with the messages of one language in messages.py."""
    sources = runtime_sources()
    sections = translations()["runtime"]
    check_runtime(sources, sections)
    table = "".join(f"    {key!r}: {text!r},\n" for key, text in runtime_messages(language, sections).items())
    embedded, count = MESSAGES_LINE.subn(lambda _: "MESSAGES = {\n" + table + "}", sources["messages.py"])
    if count != 1:
        raise ValueError("runtime/messages.py must have exactly one line that starts with MESSAGES = None")
    return {**sources, "messages.py": embedded}


def build(language):
    """The notebook as a dict. language is "JP" or "EN"."""
    texts = translations()["notebook"]
    cells = []
    used = set()

    def t(key):
        """One line of a code cell, by "section.key": a title, a form description or a message."""
        section, name = key.split(".")
        text = texts[section][name][language]
        if "\n" in text:
            raise ValueError(f"{TRANSLATIONS.name}: {key} must be a single line")
        used.add(key)
        return text

    def said(key):
        """The same text as a string literal, for a message the cell prints."""
        return json.dumps(t(key), ensure_ascii=False)

    def markdown(section):
        used.add(section)
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": "\n".join(texts[section][language]) + "\n"})

    def code(section, text):
        # Form view: Colab shows the fields and checkboxes and hides the code.
        cells.append({"cell_type": "code", "execution_count": None, "outputs": [],
                      "metadata": {"cellView": "form"},
                      "source": "#@title " + t(f"{section}.title") + "\n" + textwrap.dedent(text).strip() + "\n"})

    markdown("intro")
    code("check", f'''
    #@markdown {t("check.use_latest")}
    USE_LATEST_COMFYUI = False #@param {{type:"boolean"}}
    import re, subprocess
    found = subprocess.run(["git", "ls-remote", "--tags", "--refs", "https://github.com/Comfy-Org/ComfyUI.git", "v*"],
                           capture_output=True, text=True, timeout=60)
    releases = [tuple(map(int, tag.groups())) for tag in
                (re.search(r"refs/tags/v(\\d+)\\.(\\d+)\\.(\\d+)$", line) for line in found.stdout.splitlines()) if tag]
    if found.returncode or not releases:
        raise RuntimeError({said("check.failed")})
    LATEST_COMFY_REF = "v" + ".".join(map(str, max(releases)))
    print({said("check.latest")}, LATEST_COMFY_REF)
    # Cell 1 sets COMFY_REF from its form, so it takes the choice made here from this name.
    COMFY_REF_OVERRIDE = LATEST_COMFY_REF if USE_LATEST_COMFYUI else ""
    if USE_LATEST_COMFYUI:
        COMFY_REF = LATEST_COMFY_REF
        print({said("check.using")})
    else:
        print({said("check.not_using")})
    ''')
    code("settings", f'''
    #@markdown {t("settings.drive_folder")}
    DRIVE_FOLDER_NAME = "ComfyUI-Colab" #@param {{type:"string"}}
    #@markdown {t("settings.comfy_ref")}
    COMFY_REF = "v0.39.0" #@param {{type:"string"}}
    #@markdown {t("settings.checkpoint_seconds")}
    CHECKPOINT_SECONDS = 120 #@param {{type:"integer"}}
    #@markdown {t("settings.restore")}
    RESTORE_SAVED_ENVIRONMENT = True #@param {{type:"boolean"}}
    #@markdown {t("settings.apt_packages")}
    APT_PACKAGES = "" #@param {{type:"string"}}
    #@markdown {t("settings.model_access")}
    MODEL_ACCESS = "direct" #@param ["direct", "copy"]
    #@markdown {t("settings.sage_attention")}
    USE_SAGE_ATTENTION = True #@param {{type:"boolean"}}
    #@markdown {t("settings.fp16_accumulation")}
    USE_FP16_ACCUMULATION = False #@param {{type:"boolean"}}
    if globals().get("COMFY_REF_OVERRIDE"):
        COMFY_REF = COMFY_REF_OVERRIDE
    print("COMFY_REF:", COMFY_REF)
    ''')
    package = PurePosixPath(PACKAGE)
    code("unpack", '''
import importlib, shutil, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiohttp>=3.11,<4", "requests", "PyJWT[crypto]>=2.8,<3"], check=True)
''' + "FILES = " + repr(runtime_files(language)) + "\n" + f'''
package = Path({str(package)!r})
shutil.rmtree(package, ignore_errors=True)
package.mkdir()
for name, text in FILES.items():
    (package / name).write_text(text, encoding="utf-8")
Path({str(package) + ".py"!r}).unlink(missing_ok=True)  # Written by earlier versions of this notebook.
# Run again: import the code just written, not the one imported before.
for name in [name for name in sys.modules if name.split(".")[0] == {package.name!r}]:
    del sys.modules[name]
if {str(package.parent)!r} not in sys.path:
    sys.path.insert(0, {str(package.parent)!r})
importlib.invalidate_caches()
import {package.name} as runtime
print({said("unpack.done")})
''')
    code("setup", f'''
    #@markdown {t("setup.note")}
    from IPython.display import Javascript, display
    display(Javascript("google.colab.output.setIframeHeight(0, true, {{maxHeight: 300}})"))
    runtime.setup(root_name=DRIVE_FOLDER_NAME, comfy_ref=COMFY_REF,
                  checkpoint_seconds=CHECKPOINT_SECONDS, apt_packages=APT_PACKAGES.split(),
                  restore=RESTORE_SAVED_ENVIRONMENT, model_access=MODEL_ACCESS,
                  sage_attention=USE_SAGE_ATTENTION, fp16_accumulation=USE_FP16_ACCUMULATION)
    print({said("setup.models_location")} + DRIVE_FOLDER_NAME + "/models/")
    ''')
    markdown("models")
    code("download", f'''
    #@markdown {t("download.url")}
    MODEL_URL = "" #@param {{type:"string"}}
    #@markdown {t("download.folder")}
    MODEL_FOLDER = "checkpoints" #@param ["checkpoints", "diffusion_models", "text_encoders", "vae", "loras", "controlnet", "clip_vision", "upscale_models", "embeddings"] {{allow-input: true}}
    #@markdown {t("download.filename")}
    MODEL_FILENAME = "" #@param {{type:"string"}}
    #@markdown {t("download.sha256")}
    MODEL_SHA256 = "" #@param {{type:"string"}}
    #@markdown {t("download.token")}
    USE_DOWNLOAD_TOKEN = False #@param {{type:"boolean"}}
    #@markdown {t("download.runtime_only")}
    RUNTIME_ONLY = False #@param {{type:"boolean"}}
    DOWNLOAD_NOW = False #@param {{type:"boolean"}}
    if DOWNLOAD_NOW:
        {LOAD_CONFIG}
        token = ""
        try:
            if USE_DOWNLOAD_TOKEN:
                from google.colab import userdata
                token = userdata.get(runtime.model_token_name(MODEL_URL))
            runtime.fetch_model(config, MODEL_URL, MODEL_FOLDER, MODEL_FILENAME, MODEL_SHA256, token,
                                runtime_only=RUNTIME_ONLY)
        finally:
            token = ""
    else:
        print({said("download.skipped")})
    ''')
    code("refresh", f'''
    REFRESH_MODELS = False #@param {{type:"boolean"}}
    if REFRESH_MODELS:
        {LOAD_CONFIG}
        runtime.sync_models(config)
        print({said("refresh.done")})
    else:
        print({said("refresh.skipped")})
    ''')
    markdown("connecting")
    code("start_colab", f'''
    {LOAD_CONFIG}
    runtime.start(config, "colab")
    ''')
    code("start_tunnel", f'''
    #@markdown {t("start_tunnel.new_code")}
    NEW_PAIRING_CODE = False #@param {{type:"boolean"}}
    {LOAD_CONFIG}
    if NEW_PAIRING_CODE:
        runtime.issue_pairing_code(config)
    else:
        runtime.start(config, "tunnel")
    ''')
    code("start_access", f'''
    #@markdown {t("start_access.setup")}
    #@markdown
    #@markdown {t("start_access.hostname")}
    ACCESS_HOSTNAME = "" #@param {{type:"string"}}
    {LOAD_CONFIG}
    from google.colab import userdata
    token, audience = userdata.get("CF_TUNNEL_TOKEN"), userdata.get("CF_ACCESS_AUD")
    try:
        runtime.start_with_access(config, ACCESS_HOSTNAME, audience, token)
    finally:
        token = audience = ""
    ''')
    code("status", f'''
    #@markdown {t("status.diagnosis")}
    RUN_CONNECTION_DIAGNOSIS = False #@param {{type:"boolean"}}
    if RUN_CONNECTION_DIAGNOSIS:
        runtime.run_diagnosis()
    path = runtime.paths.PRIVATE / "checkpoint-status.json"
    print(path.read_text() if path.exists() else {said("status.no_status")})
    for name in ("comfy.log", "supervisor.log", "proxy.log", "tunnel.log"):
        path = runtime.paths.PRIVATE / name
        if path.exists():
            print("---", name)
            print(runtime.redact(path.read_text(errors="replace")[-5000:]))
    ''')
    markdown("saving")
    code("save", f'''
    SAVE_NOW = False #@param {{type:"boolean"}}
    if SAVE_NOW:
        {LOAD_CONFIG}
        runtime.checkpoint(config, force=True)
        print({said("save.done")})
    else:
        print({said("save.skipped")})
    ''')
    code("zip", f'''
    DOWNLOAD_OUTPUT_ZIP = False #@param {{type:"boolean"}}
    if DOWNLOAD_OUTPUT_ZIP:
        from google.colab import files
        files.download(runtime.zip_outputs())
    else:
        print({said("zip.skipped")})
    ''')
    code("stop", f'''
    STOP_NOW = False #@param {{type:"boolean"}}
    #@markdown {t("stop.unmount")}
    FLUSH_AND_UNMOUNT = True #@param {{type:"boolean"}}
    if STOP_NOW:
        {LOAD_CONFIG}
        runtime.stop(config, save=True)
        if FLUSH_AND_UNMOUNT:
            from google.colab import drive
            runtime.hide_drive_models()
            drive.flush_and_unmount()
            print({said("stop.unmounted")})
        else:
            print({said("stop.still_mounted")})
    else:
        print({said("stop.skipped")})
    ''')
    markdown("updating")
    code("update", f'''
    #@markdown {t("update.note")}
    UPDATE_COMFYUI_NOW = False #@param {{type:"boolean"}}
    if UPDATE_COMFYUI_NOW:
        {LOAD_CONFIG}
        runtime.update_comfyui(config, COMFY_REF)
    else:
        print({said("update.skipped")})
    ''')
    markdown("security")

    # A markdown section holds its lines per language; a code section holds one entry per text.
    defined = {section if language in entry else f"{section}.{name}"
               for section, entry in texts.items() for name in entry}
    if defined != used:
        raise ValueError(f"{TRANSLATIONS.name} has texts no cell uses: " + ", ".join(sorted(defined - used)))
    for index, cell in enumerate(cells):
        cell["id"] = f"comfy-colab-{index:02d}"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": f"{NAME}_{language}.ipynb", "provenance": []},
        "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}}, "cells": cells}


if __name__ == "__main__":
    for language in LANGUAGES:
        target = HERE / f"{NAME}_{language}.ipynb"
        target.write_text(json.dumps(build(language), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(target)
