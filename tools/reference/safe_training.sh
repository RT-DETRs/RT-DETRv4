#!/bin/bash

# Function to display the menu for selecting model size
select_model_size() {
    echo "Select model size:"
    select size in s m l x; do
        case $size in
            s|m|l|x)
                echo "You selected model size: $size"
                MODEL_SIZE=$size
                break
                ;;
            *)
                echo "Invalid selection. Please try again."
                    ;;
        esac
    done
}

# Function to display the menu for selecting task
select_task() {
    echo "Select task:"
    select task in obj365 obj2coco coco; do
        case $task in
            obj365|obj2coco|coco)
                echo "You selected task: $task"
                TASK=$task
                break
                ;;
            *)
                echo "Invalid selection. Please try again."
                ;;
        esac
    done
}

# Function to ask if the user wants to save logs to a txt file
ask_save_logs() {
    while true; do
        read -p "Do you want to save logs to a txt file? (y/n): " yn
        case $yn in
            [Yy]* )
                SAVE_LOGS=true
                break
                ;;
            [Nn]* )
                SAVE_LOGS=false
                break
                ;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Call the functions to let the user select
select_model_size
select_task
ask_save_logs

# Set config file and output directory based on selection
if [ "$TASK" = "coco" ]; then
    CONFIG_FILE="configs/dfine/dfine_hgnetv2_${MODEL_SIZE}_${TASK}.yml"
else
    CONFIG_FILE="configs/dfine/objects365/dfine_hgnetv2_${MODEL_SIZE}_${TASK}.yml"
fi

OUTPUT_DIR="output/${MODEL_SIZE}_${TASK}"

# Construct the training command
TRAIN_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c $CONFIG_FILE --use-amp --seed=0 --output-dir $OUTPUT_DIR"

# Append log redirection if SAVE_LOGS is true
if [ "$SAVE_LOGS" = true ]; then
    LOG_FILE="${MODEL_SIZE}_${TASK}.txt"
    TRAIN_CMD="$TRAIN_CMD &> \"$LOG_FILE\" 2>&1 &"
else
    TRAIN_CMD="$TRAIN_CMD &"
fi

# Run the training command
eval $TRAIN_CMD
if [ $? -ne 0 ]; then
    echo "First training failed, restarting with resume option..."
    while true; do
        RESUME_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c $CONFIG_FILE --use-amp --seed=0 --output-dir $OUTPUT_DIR -r ${OUTPUT_DIR}/last.pth"
        if [ "$SAVE_LOGS" = true ]; then
            LOG_FILE="${MODEL_SIZE}_${TASK}_2.txt"
            RESUME_CMD="$RESUME_CMD &> \"$LOG_FILE\" 2>&1 &"
        else
            RESUME_CMD="$RESUME_CMD &"
        fi
        eval $RESUME_CMD
        if [ $? -eq 0 ]; then
            break
        fi
    done
fi
