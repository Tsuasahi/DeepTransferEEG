import subprocess
import sys

methods = [
    "tl/bn-adapt.py",
    "tl/tent.py",
    "tl/pl.py",
    "tl/t3a.py",
    "tl/cotta.py",
    "tl/sar.py",
    "tl/ttime.py",
    "tl/isfda.py",
    "tl/delta.py"
]

for method in methods:
    try:
        subprocess.run([sys.executable, method, "0"], check=True)
    except:
        pass
