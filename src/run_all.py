import subprocess
import sys

scripts = [
    "src/01_explore.py",
    "src/02_preprocess.py",
    "src/03_visualize.py",
    "src/04_model.py",
    "src/05_by_position.py",
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"Running {script}...")
    print('='*50)
    subprocess.run([sys.executable, script], check=True)