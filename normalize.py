import pandas as pd

home_ownership_map = {
        'MORTGAGE': 0,
        'RENT': 1,
        'OWN': 2,
        'OTHER': 3
    }

loan_intent_map = {
    'VENTURE': 0,
    'EDUCATION': 1,
    'DEBTCONSOLIDATION': 2,
    'HOMEIMPROVEMENT': 3,
    'MEDICAL': 4,
    'PERSONAL': 5
}

loan_grade_map = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}

cb_person_default_map = {
    'Y': 1,
    'N': 0
}

def normalize (dataset):
    dataset = dataset[dataset['person_age'] <= 120]
    dataset = dataset[dataset['person_emp_length'] <= dataset['person_age']]

    dataset = dataset[dataset['person_age'] <= 120]
    dataset = dataset[dataset['person_emp_length'] <= dataset['person_age']]


    missing_data = dataset.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    dataset = dataset.dropna()
    

    dataset['person_home_ownership'] = dataset['person_home_ownership'].map(home_ownership_map)
    dataset['loan_intent'] = dataset['loan_intent'].map(loan_intent_map)
    dataset['loan_grade'] = dataset['loan_grade'].map(loan_grade_map)
    dataset['cb_person_default_on_file'] = dataset['cb_person_default_on_file'].map(cb_person_default_map)

    return dataset