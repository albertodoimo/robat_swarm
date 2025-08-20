import subprocess
import time

try:
    while True:
        # Run the date command and capture output
        result = subprocess.run(['date', '+%Y%m%d_%H_%M_%S_%N'], capture_output=True, text=True)
        
        # Print the output, stripping the newline
        print(result.stdout.strip())

        # Wait a short moment (e.g., 0.1 seconds) before next update
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped.")
