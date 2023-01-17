import numpy as np
import pickle

data = dict()
with open('wiki_dict.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data.keys())
    data['vaccination'] = "\nA vaccine is a biological preparation that provides active acquired immunity to a particular infectious disease. A vaccine typically contains an agent that resembles a disease-causing microorganism and is often made from weakened or killed forms of the microbe, its toxins, or one of its surface proteins. The agent stimulates the body's immune system to recognize the agent as a threat, destroy it, and to further recognize and destroy any of the microorganisms associated with that agent that it may encounter in the future. Vaccines can be prophylactic, or therapeutic. Some vaccines offer full sterilizing immunity, in which infection is prevented completely.\nCOVID-19 vaccines are safe, effective, and free.\nCOVID-19 vaccines available in the United States are effective at protecting people—especially those who are boosted— from getting seriously ill, being hospitalized, and even dying. As with other diseases, you are protected best from COVID-19 when you stay up to date with the recommended vaccines."
    data['stay_at_home_order'] = data['stay_at_home_orders']
    data['Vaccination'] = data['vaccination']
    data['mask'] = data['face_masks']
with open('wiki_dict.pkl', 'wb') as f:
    pickle.dump(data, f)
    print('write successfully')
with open('wiki_dict.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data.keys())