model_dir=$1
save_dir=$2
test_data_dir=$3
cd ..
set -e
for task in SarcasmDetection_Mustard ReverberationDetectionmediumroom_LJSpeechRirsNoises NoiseSNRLevelPredictionmusic_VCTKMusan NoiseSNRLevelPredictionnoise_VCTKMusan NoiseSNRLevelPredictionspeech_VCTKMusan ReverberationDetectionlargeroom_LJSpeechRirsNoises ReverberationDetectionlargeroom_VCTKRirsNoises ReverberationDetectionmediumroom_LJSpeechRirsNoises ReverberationDetectionmediumroom_VCTKRirsNoises ReverberationDetectionsmallroom_LJSpeechRirsNoises ReverberationDetectionsmallroom_VCTKRirsNoises SarcasmDetection_Mustard
do 
    echo "Sample 3 $task"
    CUDA_VISIBLE_DEVICES=0 python sample.py --model_dir ${model_dir}/ \
    --data_dir ${test_data_dir}/$task \
    --exp_name universal_task \
    --save_dir $save_dir/$task
done

