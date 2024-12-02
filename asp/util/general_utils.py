import sys
sys.path.append(".")

import time, os
from functools import wraps
import argparse
import inspect
import traceback

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
        return result
    return wrapper

def try_wrapper(function, filename, log_path):
    try:
        return function()
    except Exception as e:
        error_trace = traceback.format_exc()

        with open(log_path, 'a') as log_file:
            log_file.write(f"{filename}: {error_trace}\n")
        print(f"Error in {filename}:\n{error_trace}")

def parse_args(main_function):
    parser = argparse.ArgumentParser()

    used_short_versions = set("h")

    signature = inspect.signature(main_function)
    for param_name, param in signature.parameters.items():
        short_version = param_name[0]
        if short_version in used_short_versions or not short_version.isalpha():
            for char in param_name[1:]:
                short_version = char
                if char.isalpha() and short_version not in used_short_versions:
                    break
            else:
                short_version = None
        
        if short_version:
            used_short_versions.add(short_version)
            param_call = (f'-{short_version}', f'--{param_name}')
        else:
            param_call = (f'--{param_name}',)

        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                param_type = type(param.default)
            else:
                param_type = str
            parser.add_argument(*param_call, type=param_type, default=param.default,
                                help=f"Automatically detected argument: {param_name}, default: {param.default}")
        else:
            parser.add_argument(*param_call, required=True,
                                help=f"Required argument: {param_name}")

    args = parser.parse_args()

    return args

def assert_file_exist(*args):
    path = os.path.join(*args)
    if not os.path.exists(path):
        raise Exception(f"File {path} does not exist")
    
    return path