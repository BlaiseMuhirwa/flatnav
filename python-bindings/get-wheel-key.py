

import platform
import sys

def get_wheel_key():
    """
    Generate the wheel key based on current Python version.
    """
    try:
        version = platform.python_version_tuple()
        key = "".join(version[0:2]) 
        
        system = platform.system().lower()
        
        if system == "linux":
            platform_tag = "manylinux_x86_64"
        elif system == "darwin":
            platform_tag = "macosx_x86_64"
        else:
            raise ValueError(f"Unsupported platform: {system}")
            
        return f"cp{key}-{platform_tag}"
    except Exception as e:
        print(f"Error generating wheel key: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    print(get_wheel_key())