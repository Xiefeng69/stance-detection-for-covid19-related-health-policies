import os
patience = 5
# ['bilstm', 'bicond', 'textcnn', 'tan', 'crossnet', 'toad', 'bert_base', 'ws_bert', 'mymodel']
model = 'mymodel'
for alpha in [0.01, 0.001]:
    for p_lambda in [0.1, 0.01]:
        for hidden in [128, 256]:
            for seed in [42, 64, 26, 43, 6]:
                command = f"python run.py --topic stay_at_home_order,face_masks --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda}  && "\
                          f"python run.py --topic stay_at_home_order,vaccination --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                          f"python run.py --topic face_masks,stay_at_home_order --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                          f"python run.py --topic face_masks,vaccination --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                          f"python run.py --topic vaccination,stay_at_home_order --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                          f"python run.py --topic vaccination,face_masks --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                          f"python run.py --topic zeroshot,stay_at_home_order --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                          f"python run.py --topic zeroshot,face_masks --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda} && "\
                          f"python run.py --topic zeroshot,vaccination --model {model} --batch 16 --epoch 100 --patience {patience} --backbone bert_base --hidden {hidden} --seed {seed} --alpha {alpha} --p_lambda {p_lambda}"
                print(command)
                os.system(command)
