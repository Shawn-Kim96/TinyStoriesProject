emb_dims=(64 128 256)
num_layers=(4 8)
num_heads=(4 8)
ff_dim=(128 256 512)

for emb_dims in "${emb_dims[@]}"; do
    for num_layers in "${num_layers[@]}"; do
        for num_heads in "${num_heads[@]}"; do
            for ff_dim in "${ff_dim[@]}"; do
                sbatch ./run_training_hpc.sh --emb_dim $emb_dim --num_layers $num_layers --num_heads $num_heads --ff_dim $ff_dim
            done
        done
    done
done