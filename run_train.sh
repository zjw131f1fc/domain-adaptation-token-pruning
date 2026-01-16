#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate token-pruning

while true; do
    echo "$(date): Starting main.py..."
    python main.py

    if [ $? -eq 0 ]; then
        echo "$(date): main.py completed successfully!"
        break
    else
        echo "$(date): main.py failed, retrying in 5 seconds..."
        sleep 5
    fi
done
