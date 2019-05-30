#!/bin/bash

declare -i k0=74 #start counter of folders
declare -i k=$k0

export OMP_NUM_THREADS=1

SESNAME="sphere"
PARNAME="param_alan"
RUNSCRIPT="runningSimulation_pars_alan.py"
RUNSCRIPTCOEFFS="get_omega_coeffs_phase.py"
FIELD="om_coeffs"

FRATE=10 #frame rate for making the video

### SIMULATION PARAMETERS ###

LL=( 8 )
kk=( 0.5 1 1.5 )
ff=( 1000 )
facfac=( 0.1 )
NOTE="" #add anything special about these simulations, else leave empty

#number of cores to be used
declare -i ncores=8

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
          for (( o = 0 ; o < ${#facfac[@]} ; o++ )) ; do

          # Create subdirectory for specific session
          mkdir data/$SESNAME$k

          echo -e $SESNAME$k'\t'${LL[$l]}'\t'${kk[$m]}'\t'${ff[$n]}'\t'${facfac[$o]}'\t'$NOTE >> $PARNAME.txt

          # copy script
          cp $RUNSCRIPT data/$SESNAME$k/runscript.py

          echo -e 'Started: '$SESNAME$k'\t''L='${LL[$l]}'\t''kappa='${kk[$m]}'\t''f='${ff[$n]}'\t''factor='${facfac[$o]}
          # run Dedalus script
          mpiexec -n $ncores python3 $RUNSCRIPT ${LL[$l]} ${kk[$m]} ${ff[$n]} ${facfac[$o]} data/$SESNAME$k > $SESNAME$k'.out' 2>&1

          # wait for processes to be done
          wait

          echo "Starting plot/video command group in background.."
          {
          echo "Running plot script.."
          # run plot script
          python3 plot_output.py data/$SESNAME$k images/$SESNAME$k
          wait
          echo "Running coeffs script.."

          python3 $RUNSCRIPTCOEFFS $k > nohup_sphere$k'_coeffs.out' 2>&1
          wait

          echo "Making video.."

          #remove files if they already exist
          rm videos/$SESNAME$k'_v_ph.mp4'
          rm videos/$SESNAME$k'_om.mp4'
          rm videos/$SESNAME$k'_'$FIELD'.mp4'

          # make videos
          ffmpeg -r $FRATE -f image2 -s 1920x1080 -i images/$SESNAME$k/v_ph_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/$SESNAME$k'_v_ph.mp4' > video.out 2>&1
          ffmpeg -r $FRATE -f image2 -s 1920x1080 -i images/$SESNAME$k/om_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/$SESNAME$k'_om.mp4' > video.out 2>&1
          ffmpeg -r $FRATE -f image2 -s 1920x1080 -i images/$SESNAME$k/$FIELD'_%05d.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p videos/$SESNAME$k'_'$FIELD'.mp4' > video.out 2>&1

        } > $SESNAME$k'_plot.out' 2>&1 &

          # increment k
          k=$k+1

        done
      done
    done
done

k=$k-1
echo 'All Simulations Done. Last simulation: '$SESNAME$k
