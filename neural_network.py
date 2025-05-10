import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import getTrainTest
import normalize

# Load and normalize the dataset
dataset = pd.read_csv('data/credit_risk_dataset.csv')
dataset = normalize.normalize(dataset)

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 500]
}

# Initialize the MLPClassifier
nn_model = MLPClassifier(random_state=1)

# Initialize lists to store results
test_accuracies = []
best_params_list = []

# Create or overwrite the CSV file to store results
results_file = 'results/neural_network_multiple_runs.csv'
results_df = pd.DataFrame(columns=['Run', 'Best Parameters', 'Test Accuracy'])
results_df.to_csv(results_file, index=False)

# Run the grid search 5 times with different train-test splits
for i in range(5):
    print(f"Run {i + 1}/5")
    
    # Get a new train-test split
    X_train, X_test, y_train, y_test = getTrainTest.getTrainTest(dataset)
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=nn_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Save the best parameters and test accuracy for this run
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(test_accuracy)
    
    print(f"Best Parameters for Run {i + 1}: {best_params}")
    print(f"Test Accuracy for Run {i + 1}: {test_accuracy:.2f}")
    
    # Append the results of this run to the CSV file
    run_results = pd.DataFrame({
        'Run': [i + 1],
        'Best Parameters': [best_params],
        'Test Accuracy': [test_accuracy]
    })
    run_results.to_csv(results_file, mode='a', header=False, index=False)

# Calculate the mean test accuracy
mean_test_accuracy = sum(test_accuracies) / len(test_accuracies)
print(f"\nMean Test Accuracy over 5 runs: {mean_test_accuracy:.2f}")