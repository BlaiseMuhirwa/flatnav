

import platform

if __name__ == "__main__":
    version = platform.python_version_tuple()
    key = "".join(version[0:-1])
    print(f"cp{key}-manylinux_x86_64")