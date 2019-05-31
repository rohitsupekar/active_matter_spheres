#!/bin/bash

SESNAME="sphere"
RUNSCRIPT="get_omega_coeffs_phase.py"

### SIMULATION PARAMETERS ###

export OMP_NUM_THREADS=1

IND=( 38 39 40 41 42 43 )
FIELD="om_coeffs"


for (( l = 0 ; l < ${#IND[@]} ; l++ )) ; do

        echo -e 'Starting:'$SESNAME$''${IND[$l]}
        python3 $RUNSCRIPT ${IND[$l]} > nohup_sphere${IND[$l]}_coeffs.out 2>&1
        wait

        if [ -f videos/sphere${IND[$l]}_$FIELD'.mp4' ]
        then
            echo "remove"
            rm  videos/sphere${IND[$l]}_$FIELD'.mp4'
        fi
        echo 'Running ffmepg'
        ffmpeg -r 10 -f image2 -s 1920x1080 -i images/sphere${IND[$l]}/$FIELD'_%05d.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/sphere${IND[$l]}_$FIELD'.mp4' > video.out 2>&1
        wait

done

echo "All videos done"
