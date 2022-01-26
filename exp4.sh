#普通のFL
python exp.py --noise_for_small 0 --noise_for_large 0 --n_iterations 3 --n_global_iterations 100
#大規模病院だけ
python exp.py --noise_for_small 0 --noise_for_large 0 --n_small_users 0 --n_iterations 3 --n_global_iterations 100
#普通のFLにDP導入 (+大規模だけの場合)
python exp.py --noise_for_small 5 --noise_for_large 5 --n_iterations 3 --n_global_iterations 100
python exp.py --noise_for_small 5 --noise_for_large 5 --n_small_users 0 --n_iterations 3 --n_global_iterations 100
python exp.py --noise_for_small 8 --noise_for_large 8 --n_iterations 3 --n_global_iterations 100
python exp.py --noise_for_small 8 --noise_for_large 8 --n_small_users 0 --n_iterations 3 --n_global_iterations 100
python exp.py --noise_for_small 30 --noise_for_large 30 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 30 --noise_for_large 30 --n_small_users 0 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 100 --noise_for_large 100 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 100 --noise_for_large 100 --n_small_users 0 --n_iterations 6 --n_global_iterations 100
#小規模だけにDP導入
python exp.py --noise_for_small 5 --noise_for_large 0 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 8 --noise_for_large 0 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 30 --noise_for_large 0 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 100 --noise_for_large 0 --n_iterations 6 --n_global_iterations 100
#共有している場合に普通にDP導入
python exp.py --noise_for_small 0 --noise_for_large 0 --share_abnormal_ratio 1 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 5 --noise_for_large 0 --share_abnormal_ratio 1 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 8 --noise_for_large 0 --share_abnormal_ratio 1 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 30 --noise_for_large 0 --share_abnormal_ratio 1 --n_iterations 6 --n_global_iterations 100
python exp.py --noise_for_small 100 --noise_for_large 0 --share_abnormal_ratio 1 --n_iterations 6 --n_global_iterations 100