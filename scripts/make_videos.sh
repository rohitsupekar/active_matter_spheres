#!/bin/bash

SESNAME="sphere"

### SIMULATION PARAMETERS ###

IND=( 30 )
FIELD="om_coeffs"

for (( l = 0 ; l < ${#IND[@]} ; l++ )) ; do

        if [ -f videos/sphere${IND[$l]}_$FIELD'.mp4' ]
        then
            echo "remove"
            rm  videos/sphere${IND[$l]}_$FIELD'.mp4'
        fi

        echo "starting movie sphere${IND[$l]}"

        ffmpeg -r 15 -f image2 -s 1920x1080 -i images/sphere${IND[$l]}/$FIELD'_%05d.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/sphere${IND[$l]}_$FIELD'.mp4'

        wait

done

echo "All videos done"
