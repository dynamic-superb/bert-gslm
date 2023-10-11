model_dir=$1
save_dir=$2
test_data_dir=$3
cd ..
set -e
for task in ReverberationDetectionmediumroom_VCTKRirsNoises SpeakerVerification_VCTK SpeakerCounting_LibriTTSTestClean SpeechCommandRecognition_GoogleSpeechCommandsV1 SpeechDetection_LibriSpeechTestClean SpeechDetection_LibriSpeechTestOther SpeechDetection_LJSpeech SpeechTextMatching_LibriSpeechTestClean SpeechTextMatching_LibriSpeechTestOther SpeechTextMatching_LJSpeech SpokenTermDetection_LibriSpeechTestClean SpokenTermDetection_LibriSpeechTestOther SpokenTermDetection_LJSpeech SpoofDetection_ASVspoof2015 SpoofDetection_ASVspoof2017 StressDetection_MIRSD
do 
    echo "Sample 4 $task"
    CUDA_VISIBLE_DEVICES=0 python sample.py --model_dir ${model_dir}/ \
    --data_dir ${test_data_dir}/$task \
    --exp_name universal_task \
    --save_dir ${save_dir}/$task
done