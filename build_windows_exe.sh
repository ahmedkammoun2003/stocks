#!/usr/bin/env bash
# =============================================================================
#  build_windows_exe.sh
#  Cross-compiles main.py → stocks-windows.exe on a Linux host.
#  The resulting .exe is fully self-contained: no Python, no pip, no installs
#  needed on the target Windows machine.
#
#  Strategy
#  --------
#  Option A (default) — Docker + Windows Python via official wine-based image
#  Option B           — bare Wine install (fallback, best-effort)
#  Option C           — native build inside WSL / Windows host (instructions)
#
#  Usage:
#    chmod +x build_windows_exe.sh
#    ./build_windows_exe.sh            # Docker method (recommended)
#    ./build_windows_exe.sh --wine     # bare Wine method
#    ./build_windows_exe.sh --info     # print info only
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[BUILD]${RESET} $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
err()  { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
sep()  { echo -e "${BOLD}──────────────────────────────────────────────────${RESET}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/dist"
EXE_NAME="stocks-windows.exe"
SPEC_FILE="${SCRIPT_DIR}/stocks-windows.spec"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

# ── Parse args ────────────────────────────────────────────────────────────────
METHOD="docker"
for arg in "$@"; do
    case "$arg" in
        --wine)  METHOD="wine"  ;;
        --info)  METHOD="info"  ;;
        --help|-h)
            echo "Usage: $0 [--docker|--wine|--info]"
            echo "  (default) Docker cross-compile (recommended)"
            echo "  --wine    Use local Wine installation"
            echo "  --info    Show what would be built, then exit"
            exit 0 ;;
    esac
done

sep
echo -e "${BOLD}  Stocks Windows EXE Builder${RESET}"
echo -e "  Source : ${SCRIPT_DIR}/main.py"
echo -e "  Output : ${OUT_DIR}/${EXE_NAME}"
echo -e "  Method : ${METHOD}"
sep

# ── Info only ─────────────────────────────────────────────────────────────────
if [[ "$METHOD" == "info" ]]; then
    log "Files that will be bundled:"
    echo "  main.py, ga_optimizer.py, backtest.py, data_loader.py,"
    echo "  features.py, memory_manager.py, models/*, tunisian_stocks_30y.csv"
    log "All Python dependencies from requirements.txt will be embedded."
    log "Run without --info to start the build."
    exit 0
fi

mkdir -p "${OUT_DIR}"

# =============================================================================
#  METHOD A — Docker (recommended, most reliable)
#  Uses cdrx/pyinstaller-windows which ships a Wine + Python + PyInstaller
#  environment specifically for cross-compiling to Windows.
# =============================================================================
build_docker() {
    log "Checking Docker..."
    if ! command -v docker &>/dev/null; then
        err "Docker not found. Install Docker or use --wine."
        exit 1
    fi

    # Pull the cross-compile image (cached after first run)
    DOCKER_IMAGE="cdrx/pyinstaller-windows:python3"
    log "Pulling Docker image: ${DOCKER_IMAGE} ..."
    docker pull "${DOCKER_IMAGE}"

    sep
    log "Starting cross-compilation inside Docker container..."
    log "This will be extremely fast as heavy ML libraries are deferred to first-run setup!"
    sep

    docker run --rm \
        -v "${SCRIPT_DIR}:/src" \
        -w /src \
        --entrypoint bash \
        "${DOCKER_IMAGE}" \
        -c "
            set -e
            echo '>>> Installing PyInstaller into Wine Python...'
            wine python -m pip install pyinstaller --quiet

            echo '>>> Running PyInstaller...'
            wine pyinstaller \
                --clean \
                --noconfirm \
                stocks-windows.spec

            echo '>>> Build complete.'
        "

    # The container writes to /src/dist which is our mounted SCRIPT_DIR/dist
    BUILT="${OUT_DIR}/stocks-windows/${EXE_NAME}"
    if [[ ! -f "${BUILT}" ]]; then
        # PyInstaller one-file mode puts it directly in dist/
        BUILT="${OUT_DIR}/${EXE_NAME}"
    fi

    if [[ -f "${BUILT}" ]]; then
        SIZE=$(du -sh "${BUILT}" | cut -f1)
        ok "EXE built successfully!"
        ok "Location : ${BUILT}"
        ok "Size     : ${SIZE}"
    else
        err "Build finished but EXE not found in ${OUT_DIR}."
        err "Check the Docker output above for errors."
        exit 1
    fi
}

# =============================================================================
#  METHOD B — Bare Wine (best-effort, no Docker required)
#  Requires: wine, wine64, winetricks (or manual Python for Windows install)
# =============================================================================
build_wine() {
    log "Checking Wine..."
    if ! command -v wine &>/dev/null; then
        err "Wine not found. Install with: sudo apt install wine winetricks"
        exit 1
    fi

    WINE_PYTHON_VER="3.11.9"
    WINE_PYTHON_URL="https://www.python.org/ftp/python/${WINE_PYTHON_VER}/python-${WINE_PYTHON_VER}-amd64.exe"
    WINE_PREFIX="${HOME}/.wine_stocks_build"
    WINE_PYTHON="${WINE_PREFIX}/drive_c/Python311/python.exe"

    export WINEPREFIX="${WINE_PREFIX}"
    export WINEARCH=win64

    # ── Install Python for Windows under Wine ─────────────────────────────────
    if [[ ! -f "${WINE_PYTHON}" ]]; then
        log "Installing Python ${WINE_PYTHON_VER} for Windows under Wine..."
        log "Downloading Python installer..."
        INSTALLER="/tmp/python-win-installer.exe"
        curl -fsSL "${WINE_PYTHON_URL}" -o "${INSTALLER}"

        log "Running Python installer under Wine (silent)..."
        wine "${INSTALLER}" \
            /quiet \
            InstallAllUsers=1 \
            PrependPath=1 \
            TargetDir="C:\\Python311" \
            2>/dev/null || true

        if [[ ! -f "${WINE_PYTHON}" ]]; then
            err "Python for Windows installation failed."
            err "Try the Docker method: ./build_windows_exe.sh"
            exit 1
        fi
        ok "Python for Windows installed."
    else
        ok "Python for Windows already present."
    fi

    # ── Install pip dependencies ───────────────────────────────────────────────
    log "Installing project dependencies under Wine Python..."
    wine "${WINE_PYTHON}" -m pip install --upgrade pip --quiet
    wine "${WINE_PYTHON}" -m pip install pyinstaller --quiet
    wine "${WINE_PYTHON}" -m pip install \
        psutil==5.9.5 \
        pandas numpy xgboost torch hmmlearn scipy \
        stable-baselines3 gymnasium scikit-learn matplotlib \
        beautifulsoup4 lxml tqdm \
        --quiet
    ok "Dependencies installed."

    # ── Run PyInstaller ───────────────────────────────────────────────────────
    sep
    log "Running PyInstaller under Wine..."
    cd "${SCRIPT_DIR}"
    wine "${WINE_PYTHON}" -m PyInstaller \
        --clean \
        --noconfirm \
        "${SPEC_FILE}"

    BUILT="${OUT_DIR}/${EXE_NAME}"
    if [[ -f "${BUILT}" ]]; then
        SIZE=$(du -sh "${BUILT}" | cut -f1)
        ok "EXE built successfully!"
        ok "Location : ${BUILT}"
        ok "Size     : ${SIZE}"
    else
        err "Build finished but EXE not found. Check output above."
        exit 1
    fi
}

# =============================================================================
#  Dispatch
# =============================================================================
case "$METHOD" in
    docker) build_docker ;;
    wine)   build_wine   ;;
    *)      err "Unknown method: ${METHOD}"; exit 1 ;;
esac

sep
echo -e "${BOLD}  DONE — Transfer dist/${EXE_NAME} to your Windows machine and run it.${RESET}"
echo -e "  No Python installation required on the target."
sep
