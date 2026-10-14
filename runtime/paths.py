"""Where the runtime keeps things on the Colab machine.

Other modules read these as paths.NAME: the app action and the tests point them elsewhere."""

from pathlib import Path


BASE = Path("/content/comfy-session")  # The application user's tree: ComfyUI, its environment and models.
PRIVATE = Path("/content/.comfy-private")  # Root only: Drive, keys, logs and the session settings.
DRIVE_MOUNT = PRIVATE / "drive"
CONFIG = PRIVATE / "session.json"
DRIVE_MODELS = Path("/content/comfy-drive-models")  # The read-only view of Drive's models folder.
MODEL_FOLDERS = ("checkpoints", "diffusion_models", "unet", "text_encoders", "clip", "clip_vision", "vae",
                 "loras", "controlnet", "upscale_models", "embeddings", "audio_encoders", "model_patches")
