# pgrep -u stan python -d ' ' | xargs kill -9
model_dir=$1
save_dir=$2
test_data_dir=$3
set -e
for i in {1..4}
do 
    bash run_sample${i}.sh ${model_dir} ${save_dir} ${test_data_dir} &
done