#普通のFL
python exp.py --noise_for_small 0 --noise_for_large 0
#大規模病院だけ
python exp.py --noise_for_small 0 --noise_for_large 0 --n_small_users 0
#普通のFLにDP導入 (+大規模だけの場合)
python exp.py --noise_for_small 2 --noise_for_large 2
python exp.py --noise_for_small 2 --noise_for_large 2 --n_small_users 0
python exp.py --noise_for_small 5 --noise_for_large 5
python exp.py --noise_for_small 5 --noise_for_large 5 --n_small_users 0
python exp.py --noise_for_small 10 --noise_for_large 10
python exp.py --noise_for_small 10 --noise_for_large 10 --n_small_users 0
python exp.py --noise_for_small 30 --noise_for_large 30
python exp.py --noise_for_small 30 --noise_for_large 30 --n_small_users 0
python exp.py --noise_for_small 50 --noise_for_large 50
python exp.py --noise_for_small 50 --noise_for_large 50 --n_small_users 0
python exp.py --noise_for_small 100 --noise_for_large 100
python exp.py --noise_for_small 100 --noise_for_large 100 --n_small_users 0
#小規模だけにDP導入
python exp.py --noise_for_small 2 --noise_for_large 0
python exp.py --noise_for_small 5 --noise_for_large 0
python exp.py --noise_for_small 10 --noise_for_large 0
python exp.py --noise_for_small 30 --noise_for_large 0
python exp.py --noise_for_small 50 --noise_for_large 0
python exp.py --noise_for_small 100 --noise_for_large 0
#共有している場合に普通にDP導入
python exp.py --noise_for_small 0 --noise_for_large 0 --share_abnormal_ratio 1
python exp.py --noise_for_small 2 --noise_for_large 0 --share_abnormal_ratio 1
python exp.py --noise_for_small 5 --noise_for_large 0 --share_abnormal_ratio 1
python exp.py --noise_for_small 10 --noise_for_large 0 --share_abnormal_ratio 1
python exp.py --noise_for_small 30 --noise_for_large 0 --share_abnormal_ratio 1
python exp.py --noise_for_small 50 --noise_for_large 0 --share_abnormal_ratio 1
python exp.py --noise_for_small 100 --noise_for_large 0 --share_abnormal_ratio 1