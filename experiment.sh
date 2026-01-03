#!/bin/bash
# Copyright 2024 Shot-wise contributors
# Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

# Function to commit changes and push to remote repository
commit_and_push() {
    ./compress.sh
    git add .
    if [ -z "$1" ]; then
        git commit -m "Auto commit at $(date)"
    else
        git commit -m "$1 - Auto commit at $(date)"
    fi
    git push
}

# Pull changes from remote repository
git pull

# Remove experiment.log if it exists
if [ -f experiment.log ]; then
    rm experiment.log
fi

# Pass an undefined number of arguments to the script
python3 -u run.py exp "$@" | tee -a experiment.log

# Check if the experiment was successful and push changes to remote repository
if [[ $? = 0 ]]; then
    commit_and_push "Successful Experiment"
    echo "Successful Experiment"
else
    commit_and_push "Failed Experiment"
    echo "Failed Experiment"
fi
