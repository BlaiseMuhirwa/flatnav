#!/bin/bash 

set -ex 

function check_poetry_installed() {
    if ! command -v poetry &> /dev/null; then 
        echo "Poetry not found. Installing it now..."

        curl -sSL https://install.python-poetry.org | python3 -

        # Check the shell and append to poetry to PATH 
        SHELL_NAME=$(basename "$SHELL")
        # For newer poetry versions, this might be different. 
        # On ubuntu x86-64, for instance, I found this to be instead
        # $HOME/.local/share/pypoetry/venv/bin 
        POETRY_PATH="$HOME/.poetry/bin"

        if [[ "$SHELL_NAME" == "zsh" ]]; then 
            echo "Detected zsh shell."
            echo "export PATH=\"$POETRY_PATH:\$PATH\"" >> $HOME/.zshrc
            source $HOME/.zshrc

        elif [[ "$SHELL_NAME" == "bash" ]]; then 
            echo "Detected bash shell."
            echo "export PATH=\"$POETRY_PATH:\$PATH\"" >> $HOME/.bashrc
            source $HOME/.bashrc 

        else 
            echo "Unsupported shell for poetry installation. $SHELL_NAME"
            exit 1
        fi 
    fi
}


# Make sure we are in this directory 
cd "$(dirname "$0")"

# Install poetry if not yet installed
check_poetry_installed

poetry lock && poetry install --no-root

# Activate the poetry environment 
POETRY_ENV=$(poetry env info --path)

# Generate wheel file
$POETRY_ENV/bin/python setup.py bdist_wheel

# Assuming the build only produces one wheel file in the dist directory
WHEEL_FILE=$(ls dist/*.whl)


# Install the wheel using pip 
$POETRY_ENV/bin/pip install $WHEEL_FILE --force-reinstall

echo "Installation of wheel completed"

#Testing the wheel 
$POETRY_ENV/bin/python -c "import flatnav"
