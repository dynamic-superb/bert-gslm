model_dir=$1
save_dir=$2
test_data_dir=$3
cd ..
set -e
set -u
for task in MultiSpeakerDetection_VCTK ReverberationDetectionsmallroom_VCTKRirsNoises ReverberationDetectionlargeroom_VCTKRirsNoises SpeakerVerification_LibriSpeechTestClean Intent_Classification_FluentSpeechCommands_Action Intent_Classification_FluentSpeechCommands_Location Intent_Classification_FluentSpeechCommands_Object LanguageIdentification_VoxForge NoiseDetectiongaussian_LJSpeechMusan NoiseDetectiongaussian_VCTKMusan NoiseDetectionmusic_LJSpeechMusan NoiseDetectionmusic_VCTKMusan NoiseDetectionnoise_LJSpeechMusan NoiseDetectionnoise_VCTKMusan NoiseDetectionspeech_LJSpeechMusan NoiseDetectionspeech_VCTKMusan NoiseSNRLevelPredictiongaussian_VCTKMusan  
do 
    echo "Sample 2 $task"
    CUDA_VISIBLE_DEVICES=0 python sample.py --model_dir ${model_dir}/ckpts/ \
    --data_dir ${test_data_dir}/$task \
    --exp_name universal_task \
    --save_dir $save_dir/$task
done