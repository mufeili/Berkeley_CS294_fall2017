python run_expert.py experts/Walker2d-v1.pkl Walker2d-v1 --render \
	--num_rollouts 20
python behavioral_cloning.py \
	--data-file data/Walker2d-v1_num_rollouts_20.pkl \
	--batch-size 100 --lr 1e-3 --num-epochs 10 --test --envname Walker2d-v1 \
	--render --num_rollouts 20
