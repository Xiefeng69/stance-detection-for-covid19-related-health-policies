import os

patience = 5
hidden = 256
alpha = 0.01 # trade-off parameter
p_lambda = 0.1 # negative constant of GRL

for seed in [42, 64, 26, 43, 6]:
    for ablation in ['mymodel','woadv', 'wogeoenc', 'wgeoemb', 'wobk']:
        command = f"python run.py --topic stay_at_home_order,face_masks --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda}  && "\
                  f"python run.py --topic stay_at_home_order,vaccination --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                  f"python run.py --topic face_masks,stay_at_home_order --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                  f"python run.py --topic face_masks,vaccination --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                  f"python run.py --topic vaccination,stay_at_home_order --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                  f"python run.py --topic vaccination,face_masks --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                  f"python run.py --topic zeroshot,stay_at_home_order --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                  f"python run.py --topic zeroshot,face_masks --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                  f"python run.py --topic zeroshot,vaccination --model {ablation} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda}"
        print(command)
        os.system(command)