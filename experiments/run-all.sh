#!/bin/bash 

# Ensure we are in the experiments directory
cd "$(dirname "$0")"


# Name of the tmux session
SESSION_NAME="benchmarks"

# Check if the tmux session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

# If the session doesn't exist, create it
if [ $? != 0 ]; then
  echo "Tmux session '$SESSION_NAME' does not exist. Creating it."
  tmux new-session -d -s $SESSION_NAME "make sift && make gist && make glove-100 && make glove-200"

  echo "Tmux session '$SESSION_NAME' created. Attaching to it."

  # Attach to the tmux session
  tmux attach -t $SESSION_NAME
else
  echo "Tmux session '$SESSION_NAME' already exists. Attaching to it."
  tmux attach -t $SESSION_NAME
fi


