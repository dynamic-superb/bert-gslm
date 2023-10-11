model_dir=$1
save_dir=$2
test_data_dir=$3
cd ..
set -e
set -u
for task in MultiSpeakerDetection_LibriSpeechTestClean ReverberationDetectionsmallroom_LJSpeechRirsNoises ReverberationDetectionlargeroom_LJSpeechRirsNoises DialogueActPairing_DailyTalk AccentClassification_AccentdbExtended BirdSoundDetection_Warblrb10k ChordClassification_AcousticGuitarAndPiano DialogueActClassification_DailyTalk DialogueEmotionClassification_DailyTalk EmotionRecognition_MultimodalEmotionlinesDataset EnhancementDetection_LibrittsTestCleanWham EnvironmentalSoundClassification_AnimalsESC50 EnvironmentalSoundClassification_ExteriorAndUrbanNoisesESC50 EnvironmentalSoundClassification_HumanAndNonSpeechSoundsESC50 EnvironmentalSoundClassification_InteriorAndDomesticSoundsESC50 EnvironmentalSoundClassification_NaturalSoundscapesAndWaterSoundsESC50 HowFarAreYou_3DSpeaker
do 
    echo "Sample 1 $task "
    CUDA_VISIBLE_DEVICES=0 python sample.py --model_dir ${model_dir}/ \
    --data_dir ${test_data_dir}/$task \
    --exp_name universal_task \
    --save_dir $save_dir/$task
done