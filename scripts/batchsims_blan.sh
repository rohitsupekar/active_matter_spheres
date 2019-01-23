#!/bin/bash

declare -i k0=4 #start counter of folders
declare -i k=$k0

SESNAME="sphere"
PARNAME="param"

### SIMULATION PARAMETERS ###

LL=( 4 )
kk=( 0.5 1 )
ff=( 10 100 )

#number of cores to be used
declare -i ncores=15

### Make files and folders if they don't exist ###

if [ -f "$PARNAME.txt" ]
then
    echo "Parameter file already exists... appending to it."
else
    echo "Creating new parameter file $PARNAME.txt"
    echo -e 'Session Name''\t''Lmid''\t''Kappa''\t''f' > $PARNAME.txt
fi

if [ -d "data" ]
then
    echo "Data folder exists."
else
    mkdir data
    echo "Folder data/ created"
fi

if [ -d "images" ]
then
    echo "Image folder exists."
else
    mkdir images
    echo "Image folder images/ created"
fi

if [ -d "videos" ]
then
    echo "Video folder exists."
else
    mkdir videos
    echo "Video folder videos/ created"
fi

#check if dedalus environment is activated, else exit
if [[ $CONDA_DEFAULT_ENV != 'dedalus' ]]; then
    echo "Dedalus not activated. Exiting.."
    exit
fi


for (( l = 0 ; l < ${#LL[@]} ; l++ )) ; do
    for (( m = 0 ; m < ${#kk[@]} ; m++ )) ; do
        for (( n = 0 ; n < ${#ff[@]} ; n++ )) ; do

        # Create subdirectory for specific session
        mkdir data/$SESNAME$k

        echo -e $SESNAME$k'\t\t'${LL[$l]}'\t'${kk[$m]}'\t'${ff[$n]} >> $PARNAME.txt

        # copy script
        cp runningSimulation_pars.py data/$SESNAME$k/runscript.py

        echo -e 'Started: '$SESNAME$k'\t''L='${LL[$l]}'\t''kappa='${kk[$m]}'\t''f='${ff[$n]}
        # run Dedalus script
        mpiexec -n $ncores python3 runningSimulation_pars.py ${LL[$l]} ${kk[$m]} ${ff[$n]} data/$SESNAME$k > $SESNAME$k.out

        # wait for processes to be done
        #wait

        echo "Running plot script.."
        # run plot script
        python3 plot_output.py data/$SESNAME$k images/$SESNAME$k
        wait

        #remove files if they already exist
        rm videos/$SESNAME$k'_v_ph.mp4'
        rm videos/$SESNAME$k'_om.mp4'

        echo "Making video.."
        # make video
        ffmpeg -r 15 -f image2 -s 1920x1080 -i images/$SESNAME$k/v_ph_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/$SESNAME$k'_v_ph.mp4'
        ffmpeg -r 15 -f image2 -s 1920x1080 -i images/$SESNAME$k/om_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/$SESNAME$k'_om.mp4'

        # increment k
        k=$k+1

        done
    done
done

k=$k-1
echo 'All Simulations Done. Last simulation: '$SESNAME$k















