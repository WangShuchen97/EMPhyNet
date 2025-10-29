
#=============Train================
export OMP_NUM_THREADS=4

export CUDA_VISIBLE_DEVICES=0,1

CUDA_NUM=2

model_name='RT'

#model_name='RT_U_Net'

# model_name='RT_withoutskip'

# model_name='RT_ViT'

# model_name='RT_VAE'

# model_name='RT_DeepLab'

# model_name='U_Net' 

checkpoint_path="results/checkpoints_${model_name}"

if [ "$model_name" = "U_Net" ]; then
    epochs=500

    load_name=""
    i=32
    torchrun --nproc_per_node=$CUDA_NUM \
    main.py \
        --model_name $model_name \
        --world_size $CUDA_NUM \
        --is_amp 0\
        --is_ddp 1 \
        --epochs $epochs \
        --checkpoint_path $checkpoint_path\
        --dataset_output './data/output_rt_32'\
        --model_mode 'RayMap' \
        --save_name "model_init_${i}" \
        --save_name_final "model_init_${i}_final" \
        --output_times 100 \
        --train_channel $i \
        --train_load_name "$load_name"


    load_name="model_init_32"
    i=32

    torchrun --nproc_per_node=$CUDA_NUM main.py \
        --model_name $model_name \
        --world_size $CUDA_NUM \
        --is_ddp 1 \
        --epochs $epochs \
        --checkpoint_path $checkpoint_path\
        --dataset_output './data/output_real_32'\
        --model_mode 'main_real' \
        --save_name "model_real_${i}" \
        --save_name_final "model_real_${i}_final" \
        --output_times 1 \
        --train_channel $i \
        --train_load_name "$load_name"


    load_name="model_init_32"
    i=32

    torchrun --nproc_per_node=$CUDA_NUM main.py \
        --model_name $model_name \
        --world_size $CUDA_NUM \
        --is_ddp 1 \
        --epochs $epochs \
        --checkpoint_path $checkpoint_path\
        --dataset_output './data/output_imag_32'\
        --model_mode 'main_imag' \
        --save_name "model_imag_${i}" \
        --save_name_final "model_imag_${i}_final" \
        --output_times 1 \
        --train_channel $i \
        --train_load_name "$load_name"


else
    epochs=40
    for i in 4 8 12 16 20 24 28 32
    do
        if [ $i -eq 4 ]; then
            load_name=""
        else
            prev=$((i-4))
            load_name="model_init_${prev}"
        fi

        torchrun --nproc_per_node=$CUDA_NUM \
            main.py \
            --model_name $model_name \
            --world_size $CUDA_NUM \
            --is_amp 1\
            --is_ddp 1 \
            --epochs $epochs \
            --checkpoint_path $checkpoint_path\
            --dataset_output './data/output_rt_32'\
            --model_mode 'RayMap' \
            --save_name "model_init_${i}" \
            --save_name_final "model_init_${i}_final" \
            --output_times 100 \
            --train_channel $i \
            --train_load_name "$load_name"
    done

    for i in 4 8 12 16 20 24 28 32
    do
    if [ $i -eq 4 ]; then
        load_name="model_init_32"
    else
        prev=$((i-4))
        load_name="model_real_${prev}"
    fi

    torchrun --nproc_per_node=$CUDA_NUM main.py \
        --model_name $model_name \
        --world_size $CUDA_NUM \
        --is_ddp 1 \
        --epochs $epochs \
        --checkpoint_path $checkpoint_path\
        --dataset_output './data/output_real_32'\
        --model_mode 'main_real' \
        --save_name "model_real_${i}" \
        --save_name_final "model_real_${i}_final" \
        --output_times 1 \
        --train_channel $i \
        --train_load_name "$load_name"
    done

    for i in 4 8 12 16 20 24 28 32
    do
    if [ $i -eq 4 ]; then
        load_name="model_init_32"
    else
        prev=$((i-4))
        load_name="model_imag_${prev}"
    fi

    torchrun --nproc_per_node=$CUDA_NUM main.py \
        --model_name $model_name \
        --world_size $CUDA_NUM \
        --is_ddp 1 \
        --epochs $epochs \
        --checkpoint_path $checkpoint_path\
        --dataset_output './data/output_imag_32'\
        --model_mode 'main_imag' \
        --save_name "model_imag_${i}" \
        --save_name_final "model_imag_${i}_final" \
        --output_times 1 \
        --train_channel $i \
        --train_load_name "$load_name"
    done
fi


#=================TEST=====================

CUDA_VISIBLE_DEVICES=0 

torchrun --nproc_per_node=1 main.py --model_name $model_name --checkpoint_path $checkpoint_path --world_size 1 --mode "test" --is_ddp 1 --dataset_output './data/output_rt_32'  --model_mode 'RayMap' --test_load_name "model_init_32" --output_times 100 --output_dir "results/results_output_rt_${model_name}_32" --train_channel 32

torchrun --nproc_per_node=1 main.py --model_name $model_name --checkpoint_path $checkpoint_path --world_size 1 --mode "test" --is_ddp 1 --dataset_output './data/output_real_32'  --model_mode 'main_real' --test_load_name 'model_real_32' --output_times 1 --output_dir "results/results_output_real_${model_name}_32" --train_channel 32

torchrun --nproc_per_node=1 main.py --model_name $model_name --checkpoint_path $checkpoint_path --world_size 1 --mode "test" --is_ddp 1 --dataset_output './data/output_imag_32'  --model_mode 'main_imag' --test_load_name 'model_imag_32' --output_times 1 --output_dir "results/results_output_imag_${model_name}_32" --train_channel 32



#====================================





#=============Visualization================

#For more visualizations, please refer to ./Visualization.py

#python ./Visualization.py

#==========================================




