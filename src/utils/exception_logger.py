"""Lightweight exception logging helpers."""

import datetime
import os
import traceback

class ExceptionLogger:
    """
    A class for logging exceptions to a file instead of printing to screen.
    """
    
    def __init__(self, log_file="exceptions.log", overwrite=False):
        """
        Initialize the exception logger with a specified log file.
        
        Args:
            log_file (str): Path to the log file. Defaults to "exceptions.log".
            overwrite (bool): If True, the log file will be overwritten if it exists.
                              If False, logs will be appended to the existing file.
        """
        self.log_file = log_file
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Clear the existing log file if overwrite is True
        if overwrite and os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write(f"Log file created at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*50}\n")
    
    def log_exception(self, exception, additional_info=None):
        """
        Log an exception to the file.
        
        Args:
            exception: The exception object to log
            additional_info (str, optional): Any additional information to include
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, "a") as log:
            log.write(f"\n{'='*50}\n")
            log.write(f"TIMESTAMP: {timestamp}\n")
            log.write(f"EXCEPTION TYPE: {type(exception).__name__}\n")
            log.write(f"EXCEPTION MESSAGE: {str(exception)}\n")
            
            if additional_info:
                log.write(f"ADDITIONAL INFO: {additional_info}\n")
            
            log.write("TRACEBACK:\n")
            traceback.print_exception(type(exception), exception, exception.__traceback__, file=log)
            log.write(f"{'='*50}\n")
    