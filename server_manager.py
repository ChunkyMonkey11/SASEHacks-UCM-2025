import os
import sys
import psutil
import time
import signal
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def kill_process_on_port(port):
    """Kill any process using the specified port."""
    try:
        # For Windows
        if os.name == 'nt':
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if f':{port}' in line:
                    pid = line.split()[-1]
                    try:
                        subprocess.run(['taskkill', '/F', '/PID', pid])
                        logger.info(f"Killed process {pid} on port {port}")
                    except:
                        pass
        # For Unix-like systems
        else:
            result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line:
                    pid = line.split()[1]
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logger.info(f"Killed process {pid} on port {port}")
                    except:
                        pass
        
        time.sleep(1)  # Wait for ports to be freed
        return True
    except Exception as e:
        logger.error(f"Error killing processes: {str(e)}")
        return False

def cleanup_environment():
    """Clean up the environment before starting the server."""
    # Kill processes on common development ports
    ports = [8080, 8081, 8082, 8083, 5000]
    for port in ports:
        kill_process_on_port(port)
    
    # Clean up any temporary files
    if os.path.exists('uploads'):
        try:
            for file in os.listdir('uploads'):
                os.remove(os.path.join('uploads', file))
        except Exception as e:
            logger.error(f"Error cleaning uploads: {str(e)}")
    
    # Clean up log files
    if os.path.exists('app.log'):
        try:
            os.remove('app.log')
        except Exception as e:
            logger.error(f"Error cleaning log file: {str(e)}")

def start_server():
    """Start the Flask server."""
    try:
        # Clean up first
        cleanup_environment()
        
        # Start the server
        logger.info("Starting Flask server...")
        server_process = subprocess.Popen([sys.executable, 'app.py'])
        
        # Wait a bit to check if server starts successfully
        time.sleep(2)
        if server_process.poll() is None:
            logger.info("Server started successfully")
            return server_process
        else:
            logger.error("Server failed to start")
            return None
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        return None

if __name__ == '__main__':
    server_process = start_server()
    if server_process:
        try:
            server_process.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
            cleanup_environment()
            sys.exit(0) 