#!/bin/bash

declare -i k0=1 #start counter of folders
declare -i k=$k0

SESNAME="sphere"
PARNAME="param_alan"

echo "Starting command group"
{
echo "Running plot script.."
sleep 3
# run plot script
echo "Making video.."
} > $SESNAME$k'_plot.out' &

echo "End"


















