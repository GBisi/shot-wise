#!/bin/bash

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
./decompress.sh

# Remove exps.log if it exists
if [ -f exps.log ]; then
    rm exps.log
fi

# Run experiments for all configurations by order
for file in $(ls configs); do
    echo "Running experiment for configuration $file"
    
    # Remove config.ini if it exists and copy the configuration file
    if [ -f config.ini ]; then
        cp config.ini config.ini.bak
        rm config.ini
    fi
    cp configs/$file config.ini

    # Run the experiment
    ./experiment.sh $@ | tee -a exps.log

    # Restore the original config.ini file
    if [ -f config.ini.bak ]; then
        rm config.ini
        mv config.ini.bak config.ini
    fi

    commit_and_push "Completed experiment for configuration $file"

    echo "Finished running experiment for configuration $file"

done

commit_and_push "Experiments Completed"
