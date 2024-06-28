#!/bin/bash 

set -ex 

./build_wheel.sh

WHEEL_FILE=$(ls dist/*.whl)

# Check that the wheel file exists
if [ ! -f "$WHEEL_FILE" ]; then
    echo "Failed to build wheel file"
    exit 1
fi


# Install the wheel using pip 
# This is not optimal because we are not caching the wheel file, but it's good 
# because it avoids headaches with pip refusing to install a new wheel without 
# changing the version number
poetry run pip install $WHEEL_FILE --force-reinstall --no-cache-dir

echo "Installation of wheel completed"

#Testing the wheel 
poetry run python -c "import flatnav"
echo "Successfully installed flatnav"

