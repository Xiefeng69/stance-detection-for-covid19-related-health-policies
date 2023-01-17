geo = {'AL', 'Alabama', 'AK', 'Alaska', 'AZ', 'Arizona', 'AR', 'Arkansas', 'CA', 'California', 'CO', 'Colorado',
       'CT', 'Connecticut', 'DE', 'Delaware', 'FL', 'Florida', 'GA', 'Georgia', 'HI', 'Hawaii', 'ID', 'Idaho',
       'IL', 'Illinois', 'IN', 'Indiana', 'IA', 'Iowa', 'KS', 'Kansas', 'KY',
       'Kentucky', 'LA', 'Louisiana', 'ME', 'Maine', 'MD', 'Maryland',
       'MA', 'Massachusetts', 'MI', 'Michigan', 'MN', 'Minnesota', 'MS', 'Mississippi', 'MO',
       'Missouri', 'MT', 'Montana', 'NE',
       'Nebraska', 'NV', 'Nevada', 'NH', 'New Hampshire', 'NJ', 'New Jersey', 'NM', 'New Mexico', 'NY', 'New York',
       'NC', 'North Carolina', 'ND', 'North Dakota', 'OH', 'Ohio', 'OK', 'Oklahoma', 'OR', 'Oregon', 'PA',
       'Pennsylvania',
       'RI', 'Rhode Island', 'SC', 'South Carolina', 'SD', 'South Dakota', 'TN', 'Tennessee', 'TX', 'Texas', 'UT',
       'Utah',
       'VT', 'Vermont', 'VA', 'Virginia', 'WA', 'Washington', 'WV', 'West Virginia', 'WI', 'Wisconsin', 'WY', 'Wyoming', 'USA'}

import pandas as pd


def load_data(path):
    df = pd.read_csv(path, sep=',',error_bad_lines = False)
    row_id, row_text, row_loc, row_sta, row_target = [], [], [], [], []
    for id, row in df.iterrows():
        # print(row)
        llist = str(row['location']).split(',')
        for i in range(len(llist)):
            llist[i] = llist[i].strip()
        lset = set(llist)
        if bool(lset.intersection(geo)) is False:
            row_id.append(row['id'])
            row_text.append(row['text'].strip())
            row_sta.append(row['stance'].strip())
            row_target.append(row['target'].strip())
            row_loc.append(row['location'])
    row_list = {'id': row_id, 'target': row_target, 'text': row_text, 'stance': row_sta,
                'location': row_loc}
    return row_list


text = load_data('dataset/home.csv')

data = pd.DataFrame.from_dict(text)
data.to_csv('home_unseen.csv', index=False)
