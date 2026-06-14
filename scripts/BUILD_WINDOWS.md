# Windows build (`stocks-predictor.exe`)

PyInstaller **cannot** produce a Windows `.exe` from Linux. Use one of these options.

## Option A — Build on a Windows PC

1. Install [Python 3.12](https://www.python.org/downloads/) (check “Add to PATH”).
2. Copy the whole `stocks` project folder (including `saved_models/latest/` from training).
3. In **Command Prompt** or **PowerShell**:

```bat
cd path\to\stocks
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
scripts\build_predictor.bat
```

4. Output:
   - `dist\stocks-predictor.exe`
   - `dist\stocks-predictor-bundle\` (exe + bundled models)

Run:

```bat
cd dist\stocks-predictor-bundle
stocks-predictor.exe --strategy combined --top 20
```

## Option B — GitHub Actions (from Linux, no Windows machine)

1. Push the repo to GitHub.
2. **Actions** → **Build Windows predictor** → **Run workflow**.
3. Download artifact **stocks-predictor-windows** (zip with exe + bundle).

If models are not in the repo, copy your local `saved_models/latest/` into
`dist\stocks-predictor-bundle\saved_models\latest\` after download.

## Notes

- First build may take 10–20 minutes (torch + PyInstaller).
- Antivirus may flag large one-file exes; allow the folder or use `--onedir`:
  `python scripts\build_predictor.py --onedir`
