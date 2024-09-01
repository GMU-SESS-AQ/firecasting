#!/bin/bash

# Specify the name of the script you want to submit
SCRIPT_NAME="fc_model_creation_train_slurm_generated.sh"
echo "write the slurm script into ${SCRIPT_NAME}"
cat > ${SCRIPT_NAME} << EOF
#!/bin/bash
#SBATCH -J fc_model_creation_train_slurm       # Job name
#SBATCH --account=qtong
#SBATCH --qos=qtong             #
#SBATCH --partition=contrib     # partition (queue): debug, interactive, contrib, normal, orc-test
#SBATCH --time=24:00:00         # walltime
#SBATCH --nodes=1               # Number of nodes I want to use, max is 15 for lin-group, each node has 48 cores
#SBATCH --ntasks-per-node=8    # Number of MPI tasks, multiply number of nodes with cores per node. 2*48=96
#SBATCH --mail-user=zsun@gmu.edu    #Email account
#SBATCH --mail-type=FAIL           #When to email
#SBATCH --mem=150G
#SBATCH --cores-per-socket=4
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

# Activate your customized virtual environment
source /home/zsun/anaconda3/bin/activate

python -u << INNER_EOF

from fc_model_creation import WildfireModelTrainer, chosen_input_columns, model_type, get_model_paths, folder_path, training_data_folder

model_type = "lightgbm"  # Can be 'lightgbm' or 'tabnet'
model_paths = get_model_paths(model_type)
print(f"training model: {model_type}")
trainer = WildfireModelTrainer(
    model_type=model_type,
    training_data_folder=training_data_folder, 
    chosen_input_columns=chosen_input_columns
)

start_date_str = "20160110"
end_date_str = "20191231"

trainer.train_model(
    start_date_str, end_date_str, 
    folder_path, model_paths, 
    fire_size_threshold=1, region_dividing_longitude=-100
)

# trainer.train_model_on_one_file(
#     start_date_str=start_date_str,
#     end_date_str=end_date_str,
#     training_csv_path=f"{training_data_folder}/giant_few_shot_samples/stratified_{start_date_str}_{end_date_str}.csv",
#     model_paths=model_paths,
#     fire_size_threshold=300,
#     region_dividing_longitude=-100,
# )

# trainer.stratified_sampling(
#     start_date_str,
#     end_date_str,
#     folder_path,
#     model_paths,
#     giant_output_file=f"{training_data_folder}/giant_few_shot_samples/stratified_{start_date_str}_{end_date_str}.csv",
#     fire_size_threshold=1,
#     region_dividing_longitude=-100,
# )

print(f"Training completed and models saved to {model_paths[model_type]}")

INNER_EOF

EOF

# Submit the Slurm job and wait for it to finish
echo "sbatch ${SCRIPT_NAME}"


# Submit the Slurm job
job_id=$(sbatch ${SCRIPT_NAME} | awk '{print $4}')
echo "job_id="${job_id}

if [ -z "${job_id}" ]; then
    echo "job id is empty. something wrong with the slurm job submission."
    exit 1
fi

# Wait for the Slurm job to finish
file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
previous_content=$(<"${file_name}")
exit_code=0
while true; do
    # Capture the current content
    file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
    current_content=$(<"${file_name}")

    # Compare current content with previous content
    diff_result=$(diff <(echo "$previous_content") <(echo "$current_content"))
    # Check if there is new content
    if [ -n "$diff_result" ]; then
        echo "$diff_result"
    fi
    # Update previous content
    previous_content="$current_content"

    job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
    if [[ $job_status == *"COMPLETED"* || $job_status == *"CANCELLED"* || $job_status == *"FAILED"* || $job_status == *"TIMEOUT"* || $job_status == *"NODE_FAIL"* || $job_status == *"PREEMPTED"* || $job_status == *"OUT_OF_MEMORY"* ]]; then
        echo "Job $job_id has finished with state: $job_status"
        exit_code=1;
        break;
    fi
    sleep 100  # Adjust the sleep interval as needed
done

echo "Slurm job ($job_id) has finished."

echo "Print the job's output logs"
sacct --format=JobID,JobName,State,ExitCode,MaxRSS,Start,End -j $job_id
#find /scratch/zsun/ -type f -name "*${job_id}.out" -exec cat {} \;

echo "All slurm job for ${SCRIPT_NAME} finishes."

job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
if [[ $job_status != *"COMPLETED"* ]]; then
    exit 1
fi

exit $exit_code

