#!/bin/bash

max_iterations="$1"
num_eval_rays="$2"
wandb_selection="3"
api_key="586d8fb90e6cbf9f7a859f854a056fe53f60499e"



# switch to neuralangelo branch
neuralangelo_branch_name="neuralangelo"
git checkout "$neuralangelo_branch_name"

echo "Switched to branch '$neuralangelo_branch_name'."


# run training
#expect << EOF
ns-train neuralangelo --data data/dtu/scan105 --vis 'viewer+wandb' --max-num-iterations $max_iterations --pipeline.model.eval-num-rays-per-chunk $num_eval_rays --viewer.quit-on-train-completion True

#	expect "Wandb verification:"
#	send "$wandb_selection\r"

#	expect "Authorize:"
#	send "$api_key\r"
	
	# Wait for the script to finish (adjust this based on the script behavior)
#	expect eof
#EOF

## get the config file

# Get the latest modified folder name to get the config of the model

target_dir="outputs/scan105/neuralangelo/"
latest_folder=$(find "$target_dir" -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d ' ' -f 2)

if [ -n "$latest_folder/config.yml" ]; then
        echo "Latest modified folder in $target_dir: $latest_folder"
    else
        echo "Config file not found in $latest_folder"
    fi


## export mesh
ns-export tsdf --load-config "$latest_folder/config.yml"  --output-dir ../outputs/scan105/neuralangelo/mesh/

## run chamfer distance
python3 dtu-eval-scene.py --input_mesh ../outputs/scan105/neuralangelo/mesh/tsdf_mesh.ply --scan_id 105 --output_dir outputs/scan105/neuralangelo/chamfer_vis/ --DTU ../SampleSet/MVSData/

