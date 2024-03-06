import subprocess
import sys

requirements_file = 'requirements.txt'

with open(requirements_file, 'r') as file:
    for line in file:
        # Skip empty lines and comments
        line = line.strip()
        if line and not line.startswith('#'):
            try:
                # Use subprocess to call pip and install the packages
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', line])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {line}. Error: {e}")
