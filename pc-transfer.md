# PC Transfer — GPU Setup

Steps to clone and run this project on a Windows PC with an NVIDIA GPU.
The code already auto-detects CUDA at startup (`main.py`), so no source changes
are needed — only the install of a CUDA-enabled PyTorch wheel.

## Prerequisites on PC

1. **NVIDIA driver** — install Game Ready / Studio driver from nvidia.com.
   Verify in PowerShell:
   ```powershell
   nvidia-smi
   ```
   Should print GPU model + driver version.

2. **Python 3.12** — install from python.org. Tick "Add to PATH".
   (3.13 / 3.14 may not have CUDA torch wheels yet. 3.12 is safest.)

3. **Node.js LTS** — install from nodejs.org.

4. **Git** — install from git-scm.com.

5. *(Conditional)* **Microsoft Visual C++ Build Tools** — only needed if SAM-2's
   pip install fails with `vcvarsall.bat not found`. Install Visual Studio 2022
   Build Tools and select the "Desktop development with C++" workload.

## Repo prep on laptop (before pushing)

The laptop's `requirements.txt` pins `torch==2.9.0+cpu` (or similar). The PC
needs the CUDA wheel instead. Edit `backend/requirements.txt` on laptop and
remove the `torch` and `torchvision` lines so torch is installed separately
with a CUDA index URL on the PC. Push to GitHub.

`.gitignore` already excludes `backend/.venv/`, `frontend/node_modules/`,
`backend/jobs/`, `backend/weights/` — none of those go to GitHub.

## Setup on PC

Open PowerShell.

```powershell
git clone <your-github-url>
cd "NST FontMaker"

# backend venv
cd backend
python -m venv .venv
.venv\Scripts\activate

# torch with CUDA (12.4 wheels — pick the index URL shown current at pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# rest of dependencies
pip install -r requirements.txt
```

If the `pip install -r requirements.txt` step pulls in `torch` again from PyPI,
double-check `requirements.txt` no longer mentions torch — re-edit and run
`pip install -r requirements.txt --upgrade --force-reinstall` only on the
remaining packages.

## SAM2 checkpoint

```powershell
mkdir weights
curl.exe -L -o weights\sam2_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

(309 MB.)

## Run

Two PowerShell windows.

**Backend:**
```powershell
cd backend
.venv\Scripts\activate
uvicorn main:app --reload
```

Expected startup logs:
```
INFO device: cuda
INFO vgg-19 loaded
INFO sam2 loaded
```

**Frontend:**
```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Status bar should show "running on cuda".

## Verification

```powershell
curl http://localhost:8000/health
```
Returns:
```json
{"status":"ok","device":"cuda","vgg_loaded":true,"sam2_loaded":true}
```

## Expected performance vs laptop

768² NST, 500 iterations:
- Laptop (Iris Xe iGPU, CPU torch) — ~30 min
- RTX 30/40-series (≥8 GB VRAM) — ~30–60 s

50–60× faster.

## Troubleshooting

- **`torch.cuda.is_available()` returns False.** Driver too old or wrong torch
  wheel installed. Run `nvidia-smi` (driver), `pip show torch` (must be
  `2.x.x+cu124` or similar — NOT `+cpu`).
- **OOM during NST.** Drop `TARGET_SHORT_SIDE` in `backend/preprocessing.py`
  from 768 to 512. Or unload SAM2 between segment and stylize.
- **SAM-2 pip install fails on Windows.** Install Visual Studio 2022 Build
  Tools (C++ workload), reopen PowerShell, retry.
- **Long-path errors during pip install.** Enable Win32 long paths
  (Group Policy → Computer Configuration → Administrative Templates →
  System → Filesystem → Enable Win32 long paths).
- **Port 5173 / 8000 already in use.** Kill the prior process or change ports
  (Vite: `npm run dev -- --port 5174`; uvicorn: `uvicorn main:app --port 8001`,
  then update `BASE` in `frontend/src/api.js`).
