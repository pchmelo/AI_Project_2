from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def getTrainTest(dataset, random_state=None):
    train_dataset, test_dataset = train_test_split(
        dataset, 
        test_size=0.25,  
        random_state=random_state,  
        stratify=dataset['loan_status']  
    )

    X_train = train_dataset.drop(columns=['loan_status'])
    y_train = train_dataset['loan_status']
    X_test = test_dataset.drop(columns=['loan_status'])
    y_test = test_dataset['loan_status']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test