import pandas as pd


abbr_geo = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC'}

full_geo = {'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
            'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
            'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 
            'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 
            'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'}

location_dist = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

def add_location(path):
    df = pd.read_csv(path, sep=',')
    row_id, row_text, row_loc, row_sta, row_target = [], [], [], [], []
    for id, row in df.iterrows():
        llist = str(row['location']).split(',')
        for i in range(len(llist)):
                llist[i] = llist[i].strip()
        
        if len(llist) > 1:
            full_location = llist[0]
            abbr_location = llist[1]
        else:
            full_location = llist[0]
            abbr_location = ''

        if abbr_location in abbr_geo:
            row_id.append(row['id'])
            row_text.append(row['text'].strip())
            row_sta.append(row['stance'].strip())
            row_target.append(str(row['target']).strip())
            row_loc.append(abbr_location)
        elif full_location in full_geo:
            row_id.append(row['id'])
            row_text.append(row['text'].strip())
            row_sta.append(row['stance'].strip())
            row_target.append(str(row['target']).strip())
            row_loc.append(location_dist[full_location])
        else:
            row_id.append(row['id'])
            row_text.append(row['text'].strip())
            row_sta.append(row['stance'].strip())
            row_target.append(str(row['target']).strip())
            row_loc.append(row['location'])
    row_list = {'id': row_id, 'target': row_target, 'text': row_text, 'stance': row_sta,
                    'location': row_loc}
    return row_list

text = add_location('data_preprocess/stayhome_geo.csv')
data = pd.DataFrame.from_dict(text)
data.to_csv('stayhome_geo.csv', index=False)