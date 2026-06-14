"""
launcher.py — Standalone launcher and dependency manager for Stocks Algorithmic Trading System.
Bypasses PyInstaller size limitations by installing binary packages natively on first run.
"""

import os
import sys
import shutil

# Print a beautiful ASCI art banner
def print_banner():
    banner = r"""
======================================================================
  STOCKS ALGORITHMIC TRADING SYSTEM — WINDOWS STANDALONE LAUNCHER
======================================================================
"""
    print(banner)

def main():
    print_banner()

    # Determine execution directories
    if getattr(sys, 'frozen', False):
        # Running inside PyInstaller executable bundle
        exe_dir = os.path.dirname(sys.executable)
        src_dir = sys._MEIPASS
    else:
        # Running as standard script
        exe_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = exe_dir

    # Target folder next to the EXE to store the installed libraries
    env_dir = os.path.join(exe_dir, "trading_env")
    
    # Add env_dir and src_dir to the sys.path so we can import our modules and libraries
    sys.path.insert(0, env_dir)
    sys.path.insert(0, src_dir)

    # Check if the environment has already been set up and packages are importable
    environment_ready = False
    if os.path.exists(env_dir):
        try:
            import pandas
            import numpy
            import xgboost
            import scipy
            import sklearn
            import torch
            environment_ready = True
        except ImportError:
            pass

    if not environment_ready:
        print(">>> INITIALIZING NATIVE WINDOWS TRADING ENVIRONMENT...")
        print("This first-time setup downloads and optimizes high-performance")
        print("machine learning libraries (PyTorch, XGBoost, SciPy, stable-baselines3, etc.)")
        print("specifically for your Windows hardware.")
        print("This runs only once and ensures maximum execution speed.\n")
        
        if os.path.exists(env_dir):
            shutil.rmtree(env_dir)
        os.makedirs(env_dir)

        print(f"Installing dependencies to: {env_dir}")
        print("Please wait, this may take 2-4 minutes depending on your internet connection...")
        
        try:
            # We import pip programmatically to install libraries inside the frozen EXE context
            from pip._internal import main as pipmain
        except ImportError:
            try:
                from pip import main as pipmain
            except ImportError:
                print("\n[ERROR] Bundled pip was not found inside the executable.")
                input("Press Enter to exit...")
                sys.exit(1)

        # Define the exact packages from requirements.txt
        packages = [
            "pandas",
            "numpy",
            "xgboost",
            "hmmlearn",
            "scipy",
            "stable-baselines3",
            "gymnasium",
            "scikit-learn",
            "matplotlib",
            "beautifulsoup4",
            "lxml",
            "tqdm",
            "psutil"
        ]

        # Use CPU-only version of PyTorch for faster downloading and lighter RAM footprint
        pip_args = [
            "install",
            "--target", env_dir,
            "--no-cache-dir",
            "--extra-index-url", "https://download.pytorch.org/whl/cpu",
            "torch==1.13.1+cpu"
        ] + packages

        # Run pip installation programmatically
        exit_code = pipmain(pip_args)
        
        if exit_code != 0:
            print("\n[ERROR] Environment installation failed. Please check your internet connection.")
            input("Press Enter to exit...")
            sys.exit(exit_code)

        print("\n" + "=" * 70)
        print("  SUCCESS: High-performance trading environment is ready!")
        print("=" * 70 + "\n")

    # Launch the main application
    print(">>> Starting Algorithmic Trading System...\n")
    try:
        import main
        main.main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 70)
        print("  CRITICAL RUNTIME ERROR OCCURRED")
        print("=" * 70)
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == '__main__':
    main()
