#!/bin/bash

echo "start to run fc_model_predict_2weeks.sh"
pwd

# clean up the old log
> /home/zsun/fc_model_predict_2weeks.out
> /home/zsun/fc_model_predict_2weeks.err

# Specify the name of the script you want to submit
SCRIPT_NAME="fc_model_predict_2weeks_slurm_generated.sh"
echo "write the slurm script into ${SCRIPT_NAME}"
cat > ${SCRIPT_NAME} << EOF
#!/bin/bash
#SBATCH -J fc_model_predict_2weeks       # Job name
#SBATCH --account=qtong
#SBATCH --qos=qtong             #
#SBATCH --partition=contrib     # partition (queue): debug, interactive, contrib, normal, orc-test
#SBATCH --time=22:00:00         # walltime
#SBATCH --nodes=1               # Number of nodes I want to use, max is 15 for lin-group, each node has 48 cores
#SBATCH --ntasks-per-node=4    # Number of MPI tasks, multiply number of nodes with cores per node. 2*48=96
#SBATCH --mail-user=zsun@gmu.edu    #Email account
#SBATCH --mail-type=FAIL           #When to email
#SBATCH --mem=20G
#SBATCH --cores-per-socket=4
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`


# Activate your customized virtual environment
source /home/zsun/anaconda3/bin/activate

# Call the Python script using process substitution
python -u << INNER_EOF

from fc_model_predict_2weeks import  WildfireEmissionPredictor
from fc_model_creation import get_model_paths, chosen_input_columns, model_type, TabNetHandler, LightGBMHandler

start_date = "20210701"
end_date = "20210831"

model_type = "lightgbm"
model_paths = get_model_paths(model_type)
predictor = WildfireEmissionPredictor(
    "/groups/ESS3/zsun/firecasting/model/fc_lightgbm_single_giant.pkl_20160110_20191231_20243108220215.pkl",
    chosen_input_columns, 
    model_type)
predictor.predict_2weeks(start_date, end_date)

INNER_EOF

EOF

# Submit the Slurm job and wait for it to finish
echo "sbatch ${SCRIPT_NAME}"
# should have another check. if there is another job running, should cancel it before submitting a new job.

# Find and cancel existing running jobs with the same script name
#existing_jobs=$(squeue -h -o "%A %j" -u $(whoami) | awk -v script="$SCRIPT_NAME" '$2 == script {print $1}')

# if [ -n "$existing_jobs" ]; then
#     echo "Canceling existing jobs with the script name '$SCRIPT_NAME'..."
#     for job_id in $existing_jobs; do
#         scancel $job_id
#     done
# else
#     echo "No existing jobs with the script name '$SCRIPT_NAME' found."
# fi

# Submit the Slurm job
job_id=$(sbatch ${SCRIPT_NAME} | awk '{print $4}')
echo "job_id="${job_id}

if [ -z "${job_id}" ]; then
    echo "job id is empty. something wrong with the slurm job submission."
    exit 1
fi

# Wait for the Slurm job to finish
file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
previous_content=$(cat file_name)
exit_code=0
while true; do
    # Capture the current content\
    #echo ${job_id}
    file_name=$(find /scratch/zsun -name '*'${job_id}'.out' -print -quit)
    #echo "file_name="$file_name
    current_content=$(<"${file_name}")
    #echo "current_content = "$current_content

    # Compare current content with previous content
    diff_result=$(diff <(echo "$previous_content") <(echo "$current_content"))
    # Check if there is new content
    if [ -n "$diff_result" ]; then
        # Print the newly added content
        #echo "New content added:"
        echo "$diff_result"
        #echo "---------------------"
    fi
    # Update previous content
    previous_content="$current_content"


    job_status=$(scontrol show job ${job_id} | awk '/JobState=/{print $1}')
    #echo "job_status "$job_status
    #if [[ $job_status == "JobState=COMPLETED" ]]; then
    #    break
    #fi
    if [[ $job_status == *"COMPLETED"* ]]; then
        echo "Job $job_id has finished with state: $job_status"
        break;
    elif [[ $job_status == *"CANCELLED"* || $job_status == *"FAILED"* || $job_status == *"TIMEOUT"* || $job_status == *"NODE_FAIL"* || $job_status == *"PREEMPTED"* || $job_status == *"OUT_OF_MEMORY"* ]]; then
        echo "Job $job_id has finished with state: $job_status"
        exit_code=1
        break;
    fi
    sleep 10  # Adjust the sleep interval as needed
done

echo "Slurm job ($job_id) has finished."

echo "Print the job's output logs"
sacct --format=JobID,JobName,State,ExitCode,MaxRSS,Start,End -j $job_id
find /scratch/zsun/ -type f -name "*${job_id}.out" -exec cat {} \;
cat /scratch/zsun/test_data_slurm-*-$job_id.out

echo "All slurm job for ${SCRIPT_NAME} finishes."

exit $exit_code

