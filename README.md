English | [日本語](README_JP.md)

# ComfyUI instance on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeviumX/ComfyUI-instance-on-colab/blob/main/ComfyUI_instance_on_Colab_EN.ipynb)

A notebook that runs ComfyUI on a Google Colaboratory GPU, for use from your browser or from the **Remote Connection** of Comfy Desktop.

- Uses the models you put on Google Drive, and downloads models from Hugging Face and Civitai to Drive.
- Saves the ComfyUI environment, custom nodes and settings to Google Drive and restores them on the next runtime.
- Offers three ways to connect: a private URL, a public URL with a pairing code, or a URL on your own domain.

## Requirements

- Google Colab with a GPU. A paid plan and a compute unit balance are assumed.
- Google Drive with room for your models and the environment, and free space on Colab for the models you use.
- [Comfy Desktop](https://github.com/Comfy-Org/Comfy-Desktop), if you want to use Remote Connection. A regular browser is enough otherwise.

## How to use

1. Click **Open in Colab** above (it opens the English notebook; the button in the [Japanese README](README_JP.md) opens the Japanese one). You can also download [ComfyUI_instance_on_Colab_EN.ipynb](ComfyUI_instance_on_Colab_EN.ipynb) or [ComfyUI_instance_on_Colab_JP.ipynb](ComfyUI_instance_on_Colab_JP.ipynb) and upload it to [Google Colab](https://colab.research.google.com/). The two are the same except for the language of the text and messages.
2. Choose a GPU runtime and run cells 1 to 3 (settings, unpacking the code, setup). Allow access to Drive in the Google dialog that appears. Each cell is operated through its fields and checkboxes; an action with a checkbox only happens when you run the cell with the box checked.
3. Put your models into the folder for their type under `ComfyUI-Colab/models/` on Drive and run cell 5, or paste a Hugging Face or Civitai URL into cell 4. Check `RUNTIME_ONLY` there to use a model on this runtime only, without saving it to Drive.
4. Run one of cells 6-1 to 6-3 to start ComfyUI, and open the URL it shows (see "Ways to connect" below).
5. When you finish, make sure your outputs are on your PC, run cell 10, and wait for it to finish before you delete the runtime.

## Ways to connect

| Cell | Use it for | Who can connect |
|---|---|---|
| **6-1 Private URL** | The browser the notebook is open in | A browser signed in to the Google account that runs the notebook. Comfy Desktop cannot use it. |
| **6-2 Public URL** | Comfy Desktop, or another device | A device that opens the URL and enters the pairing code shown in the notebook |
| **6-3 Custom domain URL** | A fixed URL on your own domain | Accounts you allowed, signed in through Cloudflare Access |

- To switch to another way, stop with cell 10 first (`FLUSH_AND_UNMOUNT` off).
- **6-2:** a pairing code works once, for 10 minutes. To add another device, run 6-2 with `NEW_PAIRING_CODE` checked. Stopping ends the URL and every connected device.
- **Comfy Desktop:** enter the URL in Remote Connection, and turn on **Auto-download outputs** to save generated files to your PC.
- **Stop button:** **Stop** at the top of the ComfyUI page saves the environment and disconnects. You can choose to delete the runtime as well; that loses any output you have not downloaded to your PC.

### Setting up the custom domain URL (6-3)

You need a Cloudflare account, a domain managed by Cloudflare DNS, and Cloudflare Zero Trust (the free plan is enough). Do this once. Menu names in the Cloudflare dashboard may differ slightly.

1. In Zero Trust, create an **Access application** (self-hosted) for the hostname you will use, with a policy that allows only your own email address. Choose a sign-in method; the one-time PIN by email also works inside Comfy Desktop. Create this before the tunnel.
2. Create a **tunnel** (connector type: cloudflared) and copy its token, the long string that starts with `eyJ`. You do not need to install the connector anywhere. Add a public hostname to the tunnel: your hostname, service type HTTP, URL `127.0.0.1:8189`.
3. Copy the **Application Audience (AUD) Tag** of the Access application.
4. In Colab, open Secrets (the key icon), add `CF_TUNNEL_TOKEN` with the token and `CF_ACCESS_AUD` with the AUD tag as their values, and allow this notebook to read them.
5. Enter the hostname in cell 6-3 and run it. Open the URL it shows and sign in.

- Keep the tunnel token secret. Store it only in Colab Secrets, and delete the tunnel in Cloudflare if it may have leaked.
- Cell 6-3 only connects when the hostname asks for the Access sign-in and the tunnel points to `127.0.0.1:8189`. Right after changing the settings in Cloudflare, wait a few minutes and run it again.

## Notes

### Security

- Connecting Drive grants access to your whole Drive. Use a dedicated Google account that holds no personal data.
- Custom nodes run their own code on the runtime. Install only custom nodes and models you trust. The notebook keeps Drive and the connection secrets away from them, but this is not complete isolation.
- The environment saved on Drive is restored on the next runtime, and its custom nodes run again. Their names are shown during setup; if one looks suspicious, turn off `RESTORE_SAVED_ENVIRONMENT` in cell 1 and start with a fresh environment.
- Do not share pairing codes or tokens, and clear the cell outputs before you share the notebook. Register a download token in Colab Secrets only when a download needs one.

### Saving

- The environment is saved to Drive automatically while ComfyUI runs, but changes made just before a sudden disconnection may not be restored. Use cell 8 to save right away.
- Input and output files are not saved to Drive. Previews in Desktop and outputs generated while Desktop was not connected are not downloaded automatically; collect them as a ZIP with cell 9.

### Updating ComfyUI

- A restored environment keeps the ComfyUI version it was saved with.
- To update ComfyUI while keeping your custom nodes, settings and workflows, run cell 0 with its box checked (it selects the latest release) or enter the version as `COMFY_REF` in cell 1, then run 1 → 2 → 3 → 11 → one of 6-1 to 6-3. If ComfyUI is running, stop it with cell 10 first (`FLUSH_AND_UNMOUNT` off).
- If the update fails, ComfyUI is put back to its previous version and nothing on Drive changes.
- This notebook was tested with the default `COMFY_REF`.

### Faster generation

- **SageAttention (on by default):** [SageAttention](https://github.com/thu-ml/SageAttention) makes attention faster. The gain shows mostly with video models. Setup tries it on the GPU of the runtime and starts ComfyUI with the standard attention if it cannot be used. The output of cell 3 says which of the two it is.
- **fp16 accumulation (off by default):** turn on `USE_FP16_ACCUMULATION` in cell 1 to start ComfyUI with `--fast fp16_accumulation`. ComfyUI calls it an experimental feature that may lower quality.
- Both take effect during setup in cell 3. If your results come out wrong, turn off `USE_SAGE_ATTENTION` or `USE_FP16_ACCUMULATION` in cell 1 and run again on a new runtime.

### When the UI does not appear

- The first load can take around a minute, depending on your connection.
- If it stays on the logo, run cell 7 with `RUN_CONNECTION_DIAGNOSIS` checked. It shows where the connection stops. If it finds no problem, check your PC (ad blockers, security software, browser extensions).
- With 6-1, check that the browser is signed in to the same Google account as the notebook.

### Terms

- Follow the [Colab terms](https://research.google.com/colaboratory/faq.html) and the [limits of Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
