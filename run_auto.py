"""
Automated runner for the 2D blueprint to 3D model project
This script runs the project with default settings automatically
"""
import subprocess
import sys

# Prepare input responses for the interactive prompts
# Response order:
# 1. Blender path (default - empty)
# 2. StackingFile or ConfigFile (default ConfigFile - empty)
# 3. Config file path (default - empty)
# 4. Set custom images? (y)
# 5. Image path for config file (Images/Examples/example.png - has doors and furniture)
# 6. Continue? (empty = yes)
# 7. Clear cached data? (default = yes - empty)
inputs = """


y
Images/Examples/example.png


"""

# Run main.py with the inputs
process = subprocess.Popen(
    [sys.executable, "main.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send the inputs and get the output
stdout, stderr = process.communicate(input=inputs)

# Print the output
print(stdout)
if stderr:
    print("STDERR:", stderr, file=sys.stderr)

# Exit with the same code as the subprocess
sys.exit(process.returncode)
