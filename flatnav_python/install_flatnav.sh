#!/bin/bash 

set -ex 

# Activate the poetry environment 
POETRY_ENV=$(poetry env info --path)

# Generate wheel file
$POETRY_ENV/bin/python setup.py bdist_wheel 

# Assuming the build only produces one wheel file in the dist directory
WHEEL_FILE=$(ls dist/*.whl)


# Install the wheel using pip 
$POETRY_ENV/bin/pip install $WHEEL_FILE

echo "Installation of wheel completed"

#Testing the wheel 
$POETRY_ENV/bin/python -c "import flatnav"

