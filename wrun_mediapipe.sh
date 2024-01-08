#!/usr/bin/bash
# Usage    : Used by wsubmit.sh to process 1 video file
# Parameter: 1) Submitted video file name (e.g AR Gait_P_H019_Free walk_09-12-2022_15-24-52_noAR.sub)
#            2) left leg length in mm
#            3) right leg length in mm
# Output   : Pack of files in $OPENPOSE_ROOT/output/ARGait.../*.tar.gz

echo Started at `date`

OPENPOSE_ROOT=/data/openpose
WEB_ROOT=/data/pdlogger
WEB_PUBLIC=$WEB_ROOT/public/PD

videoFile="$OPENPOSE_ROOT/input/$1"
if [ ! -f "$videoFile" ]; then
 echo ERROR: Input file $1 not found
 exit 1
fi
videoName="${1%.sub}_mp"

# if output directory exist, clear the content else create directory
if [ -d "$OPENPOSE_ROOT/output/$videoName" ]
then
 rm -f "$OPENPOSE_ROOT/output/$videoName/*"
else
 mkdir "$OPENPOSE_ROOT/output/$videoName"
fi

# set env variables for python scripts
export PYTHONPATH=.:$OPENPOSE_ROOT/gait
export MODEL_DIR=$OPENPOSE_ROOT

# make a copy of video to output folder
cd "$OPENPOSE_ROOT/output/$videoName"
cp -p $videoFile .
videoFile="$OPENPOSE_ROOT/output/$videoName/$1"
python3 -m mediapipe_video Side.RIGHT "$videoFile" "$OPENPOSE_ROOT/output/$videoName" $2 $3
if [ $? -gt 0 ]; then
 echo ERROR: Video processing failed
 exit 1
fi

# gather gait analysis data
# cbtaName="${videoName}_cbta.csv"
# savName="${videoName}_sav.csv"
# gaitName="${videoName}_gait.csv"
# kineName="${videoName}_kine.csv"
# imuName="${videoName}_imu.csv"
# frontName="${videoName}_front.csv"
# echo cbta > "$gaitName"
# if [ -f "$cbtaName" ]; then
#  awk -f $OPENPOSE_ROOT/gait/pickcsv.awk "$cbtaName" >> "$gaitName"
# else
#  echo No result >> "$gaitName"
# fi
# echo >> "$gaitName"
# echo sav >> "$gaitName"
# if [ -f "$savName" ]; then
#  awk -f $OPENPOSE_ROOT/gait/pickcsv.awk "$savName" >> "$gaitName"
# else
#  echo No result >> "$gaitName"
# fi
# echo >> "$gaitName"
# echo front >> "$gaitName"
# if [ -f "$frontName" ]; then
#  awk -f $OPENPOSE_ROOT/gait/pickcsv.awk "$frontName" >> "$gaitName"
# else
#  echo No result >> "$gaitName"
# fi
# echo >> "$gaitName"
# echo IMU >> "$gaitName"
# if [ -f "$imuName" ]; then
#  awk -f $OPENPOSE_ROOT/gait/pickcsv.awk "$imuName" >> "$gaitName"
# else
#  echo No result >> "$gaitName"
# fi
# echo >> "$gaitName"
# echo kine >> "$gaitName"
# if [ -f "$kineName" ]; then
#  cat "$kineName" >> "$gaitName"
# else
#  echo No result >> "$gaitName"
# fi
# if [ $(grep -c "No result" *gait.csv) -eq 5 ]; then
#  rm "$gaitName"
# fi

# convert video to x.264 format for browser playback
aviName="${videoName}.avi"
mp4Name="${videoName}.mp4"
#HandBrakeCLI -e x264 -a none -i "$aviName" -o "$mp4Name"
deface -o "$mp4Name" --replacewith blur --ffmpeg-config '{"macro_block_size":1, "codec": "libx264"}' --execution-provider CUDAExecutionProvider "$aviName"  2>/dev/null

# copy the files to Web public PD folder
cp -f *.pkl "$WEB_PUBLIC/${videoName%_noAR*}"
cp -f *.mp4 "$WEB_PUBLIC/${videoName%_noAR*}"
# cp -f *_gait.csv "$WEB_PUBLIC/${videoName%_noAR*}"
# cp -f *_kine_?.csv "$WEB_PUBLIC/${videoName%_noAR*}"
# cp -f *_cbta_?.csv "$WEB_PUBLIC/${videoName%_noAR*}"

# update pd_assessment table with gait parameters
# if [ -f "$gaitName" ]; then
#  fieldSets=`awk -F',' -f $WEB_ROOT/supporting.files/update_pda.awk "$gaitName"`
#  php $WEB_ROOT/artisan pda:update "${videoName%_noAR}" "$fieldSets"
# fi
cd -

# clean up
rm -rf "$OPENPOSE_ROOT/output/$videoName"

echo
echo Completed at `date`
