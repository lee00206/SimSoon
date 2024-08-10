# BBBP
python main.py --task downstream --data bbbp --criterion bce --lr 5e-5 --nim_classes 1

# Clintox
python main.py --task downstream --data clintox --criterion bce --lr 5e-5 --num_classes 2

# Tox21
python main.py --task downstream --data tox21 --criterion bce --lr 5e-3 --num_classes 12

# HIV
python main.py --task downstream --data hiv --criterion bce --lr 1e-4 --num_classes 1

# ESOL
python main.py --task downstream --data esol --criterion rmse --lr 5e-5 --num_classes 1

# Freesolv
python main.py --task downstream --data freesolv --criterion rmse --lr 1e-4 --num_classes 1

# Lipophilicity
python main.py --task downstream --data lipophilicity --criterion rmse --lr 5e-5 --num_classes 1





