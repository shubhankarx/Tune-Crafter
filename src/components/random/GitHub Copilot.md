GitHub Copilot
The error message Ports are not available: exposing port TCP 0.0.0.0:5000 -> 0.0.0.0:0: listen tcp 0.0.0.0:5000: bind: address already in use indicates that the port 5000 is already in use by another process on your machine. To resolve this issue, you can either stop the process that is using port 5000 or run your Docker container on a different port.

Steps to Resolve the Issue:
Identify the Process Using Port 5000:

You can use the lsof command to identify the process using port 5000.
Stop the Process Using Port 5000:

Once you identify the process, you can stop it. For example, if the process ID (PID) is 1234, you can stop it using the kill command.
Run Docker on a Different Port:

If you do not want to stop the process using port 5000, you can run your Docker container on a different port. Modify your interactive.sh script to use a different port, such as 5001.
Example: Modify interactive.sh to Use a Different Port
If your interactive.sh script contains a command to run the Docker container, modify the port mapping to use a different port.

Example: Run Docker Container with a Different Port
If you are running the Docker container directly from the command line, specify a different port mapping.

Verify the Changes
Run the Modified Script:

Run the modified interactive.sh script.
Check Docker Container:

Verify that the Docker container is running and accessible on the new port (5001 in this example).
By following these steps, you should be able to resolve the port conflict and run your Docker container successfully.