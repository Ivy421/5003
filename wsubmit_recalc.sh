#!/usr/bin/bash
# Usage    : For web app to submit 1 PKL file for processing
# Parameter: 1) File name (e.g ARGait_P_H019_Free walk_09-12-2022_15-24-52)
#            2) Left Leg Length in mm
#            3) Right Leg Length in mm
#            4) params
#            5) override left stride (0 = default midpoint)
#            6) override right stride (0 = default midpoint)

OPENPOSE_ROOT=/data/openpose
WEB_ROOT=/data/pdlogger
pkl="${WEB_ROOT}/storage/app/public/${1}/${1}_noAR.pkl"
recalcType=`echo "$4" | awk '{patsplit($0,arr,/[[:alnum:]]+/); if (arr[1]=="adjustOpenpose") print arr[2]}'`
# recalcType: 0 for raw, 1 for OPA, 2 for RJN, 3 for Mediapipe, 4 for Manual, 5 for Mediapipe OPA, 6 for Mediapipe RJN
if [[ $recalcType -eq 3 ]] || [[ $recalcType -eq 5 ]] || [[ $recalcType -eq 6 ]]; then
 pkl="${WEB_ROOT}/storage/app/public/${1}/${1}_noAR_mp.pkl"
fi
# check if there is input file, exit if none
if [ ! -f "$pkl" ]; then
 echo ERROR: $pkl file not found
 exit 1
fi

LOGFILE=$OPENPOSE_ROOT/log/recalc_${1}-`date +%Y%m%d%H%M%S`.log
echo singularity exec --nv -bind=/data $OPENPOSE_ROOT/opp20.sif $OPENPOSE_ROOT/gait/wrun_recalc.sh "$pkl" $2 $3 "$4" $5 $6
echo $LOGFILE
singularity exec --nv --bind=/data $OPENPOSE_ROOT/opp20.sif $OPENPOSE_ROOT/gait/wrun_recalc.sh "$pkl" $2 $3 "$4" $5 $6 >& "$LOGFILE"
exit 0
