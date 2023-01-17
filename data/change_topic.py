import pandas as pd
# change the target file's topic automatically
data = pd.read_csv('./covid19-policies-stance/face_masks_train.csv')
data['Target'] = data['Target'].map({"mask": "face_masks"})
print(data.head())
data.to_csv('./covid19-policies-stance/face_masks_train.csv', index=False)