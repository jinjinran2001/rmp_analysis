# Data manipulation and analysis
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical tests
from scipy.stats import (
    ttest_ind,
    norm,
    ks_2samp,
    mannwhitneyu
)

# Scikit-learn preprocessing and model selection
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    KFold
)
from sklearn.preprocessing import StandardScaler

# Scikit-learn metrics
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score
)

# Machine Learning Models
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import (
    XGBRegressor,
    XGBClassifier
)

# Imbalanced learning
from imblearn.over_sampling import SMOTE

import stats

print("Libraries imported successfully.")

# Q1
num = pd.read_csv('rmpCapstoneNum.csv', header=None)
qual = pd.read_csv('rmpCapstoneQual.csv', header=None)
tag = pd.read_csv('rmpCapstoneTags.csv', header=None)

num_columns = [
    "Average Rating",            # The arithmetic mean of all individual quality ratings of this professor
    "Average Difficulty",        # The arithmetic mean of all individual difficulty ratings of this professor
    "Number of Ratings",         # Total number of ratings these averages are based on
    "Received a 'pepper'?",      # Boolean - judged as "hot" by the students
    "Proportion Retake",         # Proportion of students that said they would take the class again
    "Online Ratings Count",      # Number of ratings coming from online classes
    "Male Gender",               # Boolean – 1: determined with high confidence that professor is male
    "Female Gender"              # Boolean – 1: determined with high confidence that professor is female
]
num.columns = num_columns

qual_columns = [
    "Major/Field",  # Column 1: Major/Field
    "University",   # Column 2: University
    "US State"      # Column 3: US State (2-letter abbreviation)
]
qual.columns = qual_columns

tags_columns = [
    "Tough grader",              # Column 1
    "Good feedback",             # Column 2
    "Respected",                 # Column 3
    "Lots to read",              # Column 4
    "Participation matters",     # Column 5
    "Don't skip class",          # Column 6
    "Lots of homework",          # Column 7
    "Inspirational",             # Column 8
    "Pop quizzes!",              # Column 9
    "Accessible",                # Column 10
    "So many papers",            # Column 11
    "Clear grading",             # Column 12
    "Hilarious",                 # Column 13
    "Test heavy",                # Column 14
    "Graded by few things",      # Column 15
    "Amazing lectures",          # Column 16
    "Caring",                    # Column 17
    "Extra credit",              # Column 18
    "Group projects",            # Column 19
    "Lecture heavy"              # Column 20
]
tag.columns = tags_columns

# Ensure the datasets have the same number of records
assert len(num) == len(qual) == len(tag), "Datasets lengths do not match."

# Merge the datasets
merged_df = pd.concat([num, qual, tag], axis=1)
print('============Q1============ ')
# set the threshold to 5 ratings and exclude the professors with less than 5 ratings
print('before cleaning n ratings', len(merged_df))
merged_df = merged_df[merged_df['Number of Ratings'] >= 5]
print('after cleaning n ratings', len(merged_df))

merged_df = merged_df.dropna(subset='Average Rating')

female_rating = merged_df[merged_df['Male Gender'] == 0]['Average Rating']
male_rating = merged_df[merged_df['Male Gender'] == 1]['Average Rating']

# run welch's t-test
t_stat, p_value = ttest_ind(male_rating, female_rating, 
                                 alternative='greater',  # one-sided test
                                 equal_var=False)  # Welch's t-test (unequal variances)

print(f"t-statistic for Q1: {t_stat:.4f}")
print(f"p-value Q1: {p_value:.4f}")

# Q2

# Q3

# Q4
# For each tag column
for tag_column in tags_columns:
    # Create new normalized column
    merged_df[f'{tag_column}_normalized'] = merged_df[tag_column] / merged_df['Number of Ratings']
# create a list of normalized cols
normalized_columns = [f'{tag_column}_normalized' for tag_column in tags_columns]

result = []
for tags in normalized_columns:
    male = merged_df[merged_df['Male Gender'] == 1][tags]
    female = merged_df[merged_df['Male Gender'] == 0][tags]
    t_stat, p_value = mannwhitneyu(male, female, 
                                 alternative='two-sided',  # one-sided test
                                 )  # Welch's t-test (unequal variances)
    result.append((tags, p_value))

# Sort the results by p-value
print('============Q4============')
print('p-values for Q4')
print('most significant p-values')
result.sort(key=lambda x: x[1])
print(result[:3])

print('least significant p-values')
# get the top 3 great p-values
top_3 = result[::-1][:3]
print(top_3)

# Q5
print('============Q5============')
num_data = pd.read_csv('rmpCapstoneNum.csv')
qual_data = pd.read_csv('rmpCapstoneQual.csv')
tags_data = pd.read_csv('rmpCapstoneTags.csv')
num_data.head()

# Renaming columns for each dataset based on provided descriptions

# rmpCapstoneNum.csv
num_data.columns = [
    'Average Rating',  # Column 1
    'Average Difficulty',  # Column 2
    'Number of Ratings',  # Column 3
    'Received Pepper',  # Column 4
    'Proportion Retake',  # Column 5
    'Ratings Online',  # Column 6
    'Male Gender',  # Column 7
    'Female Gender'  # Column 8
]

# rmpCapstoneQual.csv
qual_data.columns = [
    'Major/Field',  # Column 1
    'University',  # Column 2
    'US State'  # Column 3
]

# rmpCapstoneTags.csv
tags_data.columns = [
    'Tough Grader', 'Good Feedback', 'Respected', 'Lots to Read',
    'Participation Matters', 'Don’t Skip Class', 'Lots of Homework',
    'Inspirational', 'Pop Quizzes!', 'Accessible', 'So Many Papers',
    'Clear Grading', 'Hilarious', 'Test Heavy', 'Graded by Few Things',
    'Amazing Lectures', 'Caring', 'Extra Credit', 'Group Projects',
    'Lecture Heavy'
]

# Display the updated columns for verification
num_data.columns, qual_data.columns, tags_data.columns

# Step 1: Calculate the mean of 'Number of Ratings'
mean_num_ratings = num_data['Number of Ratings'].mean()

# Step 2: Drop rows where 'Number of Ratings' is less than the mean
num_data_filtered = num_data[num_data['Number of Ratings'] >= mean_num_ratings]

# Step 3: Remove rows with missing values in relevant columns ('Average Difficulty', 'Male Gender', 'Female Gender')
num_data_cleaned = num_data_filtered.dropna(subset=['Average Difficulty', 'Male Gender', 'Female Gender'])
# Extracting relevant columns
difficulty = num_data_cleaned['Average Difficulty']
male_gender = num_data_cleaned['Male Gender']
female_gender = num_data_cleaned['Female Gender']

# Splitting data by gender
male_difficulty = difficulty[male_gender == 1]
female_difficulty = difficulty[female_gender == 1]

# Performing t-test to check for significant difference
t_stat, p_value = ttest_ind(male_difficulty, female_difficulty, equal_var=False)

# Summary statistics
male_mean = np.mean(male_difficulty)
female_mean = np.mean(female_difficulty)
male_std = np.std(male_difficulty)
female_std = np.std(female_difficulty)

# Plotting distributions
plt.figure(figsize=(10, 6))
plt.hist(male_difficulty, bins=20, alpha=0.5, label='Male Professors', color='blue')
plt.hist(female_difficulty, bins=20, alpha=0.5, label='Female Professors', color='red')
plt.title('Distribution of Average Difficulty Ratings by Gender')
plt.xlabel('Average Difficulty')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print("male_mean: ", male_mean)
print("female_mean: ",female_mean)
print("male_std: ", male_std)
print("female_std: ", female_std)
print("t_stat: ", t_stat)
print("p_value: ", p_value)

print('============Q6============')

# Step 1: Calculate the mean of 'Number of Ratings'
mean_num_ratings = num_data['Number of Ratings'].mean()

# Step 2: Drop rows where 'Number of Ratings' is less than the mean
num_data_filtered = num_data[num_data['Number of Ratings'] >= mean_num_ratings]

# Step 3: Clean data by dropping rows with NaN in relevant columns
ratings_cleaned = num_data_filtered.dropna(subset=['Average Rating', 'Male Gender', 'Female Gender'])

# Step 4: Split data by gender
average_rating = ratings_cleaned['Average Rating']
male_ratings = average_rating[ratings_cleaned['Male Gender'] == 1]
female_ratings = average_rating[ratings_cleaned['Female Gender'] == 1]

# Step 5: Calculate means and standard deviations
male_mean_rating = np.mean(male_ratings)
female_mean_rating = np.mean(female_ratings)
mean_difference = male_mean_rating - female_mean_rating

male_std_rating = np.std(male_ratings, ddof=1)
female_std_rating = np.std(female_ratings, ddof=1)
n_male = len(male_ratings)
n_female = len(female_ratings)

# Step 6: Calculate pooled standard deviation
pooled_std = np.sqrt(
    ((n_male - 1) * male_std_rating**2 + (n_female - 1) * female_std_rating**2) / (n_male + n_female - 2)
)

# Step 7: Calculate Cohen's d
cohens_d = mean_difference / pooled_std

# Step 8: Calculate standard error of Cohen's d
se_d = np.sqrt((n_male + n_female) / (n_male * n_female) + cohens_d**2 / (2 * (n_male + n_female)))

# Step 9: Calculate 95% confidence interval for Cohen's d
z_critical = norm.ppf(0.975)  # 95% confidence
ci_lower = cohens_d - z_critical * se_d
ci_upper = cohens_d + z_critical * se_d

# Output results
print("Male Mean Rating:", male_mean_rating)
print("Female Mean Rating:", female_mean_rating)
print("Mean Difference:", mean_difference)
print("Pooled Standard Deviation:", pooled_std)
print("Cohen's d:", cohens_d)
print("Standard Error of Cohen's d:", se_d)
print("95% Confidence Interval for Cohen's d: [{}, {}]".format(ci_lower, ci_upper))

print('============Q7============')


# Reloading numerical dataset for problem 7
num_data_path = 'rmpCapstoneNum.csv'
num_data = pd.read_csv(num_data_path)

# Renaming columns
num_data.columns = [
    'Average Rating', 'Average Difficulty', 'Number of Ratings',
    'Received Pepper', 'Proportion Retake', 'Ratings Online',
    'Male Gender', 'Female Gender'
]

# Step 1: Calculate the mean of 'Number of Ratings'
mean_num_ratings = num_data['Number of Ratings'].mean()

# Step 2: Drop rows where 'Number of Ratings' is less than the mean
num_data_filtered = num_data[num_data['Number of Ratings'] >= mean_num_ratings]
num_data_cleaned = num_data_filtered.dropna()
# Preparing numerical features (X) and target (y)
X_num = num_data_cleaned[['Average Difficulty', 'Number of Ratings', 'Received Pepper',
                          'Proportion Retake', 'Ratings Online', 'Male Gender', 'Female Gender']]
y_num = num_data_cleaned['Average Rating']

# Standardizing the features for models that are sensitive to scaling
scaler = StandardScaler()
X_scaled_num = scaler.fit_transform(X_num)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_num, y_num, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    "Ridge": Ridge()
}

# Step 3: Fit and evaluate each model
evaluation_results_num = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Store the evaluation metrics
    evaluation_results_num[model_name] = {
        'RMSE': rmse,
        'R² Score': r2
    }

# Extracting the model names and their evaluation metrics for plotting
model_names_num = list(evaluation_results_num.keys())
mse_values_num = [evaluation_results_num[model]['RMSE'] for model in model_names_num]
r2_values_num = [evaluation_results_num[model]['R² Score'] for model in model_names_num]

# Plotting RMSE for each model
plt.figure(figsize=(10, 6))
plt.barh(model_names_num, mse_values_num, color='skyblue')
plt.xlabel("Root Mean Squared Error (RMSE)")
plt.title("Root Mean Squared Error for Different Models (Numerical Features)")
plt.gca().invert_yaxis()  # Invert the axis to have the best performer on top
plt.show()

# Plotting R² Score for each model
plt.figure(figsize=(10, 6))
plt.barh(model_names_num, r2_values_num, color='lightgreen')
plt.xlabel("R² Score")
plt.title("R² Score for Different Models (Numerical Features)")
plt.gca().invert_yaxis()  # Invert the axis to have the best performer on top
plt.show()

print(evaluation_results_num)


# Train the Ridge Regression model explicitly
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Extract coefficients (acts as feature importance in Ridge regression)
coefficients = ridge_model.coef_

# Map coefficients to feature names
importance_df = pd.DataFrame({
    'Feature': X_num.columns,  # Use the feature names
    'Importance': np.abs(coefficients)  # Absolute value of coefficients
}).sort_values(by='Importance', ascending=False)

# Display the most important features
print("Top features:")
print(importance_df)

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance (Absolute Coefficients)')
plt.ylabel('Numerical Features')
plt.title('Ridge Regression Feature Importances for Predicting Average Rating')
plt.gca().invert_yaxis()  # Best predictors on top
plt.show()

print('============Q8============')
# Loading tags and numerical datasets
tags_data_path = 'rmpCapstoneTags.csv'
num_data_path = 'rmpCapstoneNum.csv'

tags_data = pd.read_csv(tags_data_path)
num_data = pd.read_csv(num_data_path)

num_data.columns = [
    'Average Rating', 'Average Difficulty', 'Number of Ratings',
    'Received Pepper', 'Proportion Retake', 'Ratings Online',
    'Male Gender', 'Female Gender'
]

tags_data.columns = [
    'Tough Grader', 'Good Feedback', 'Respected', 'Lots to Read',
    'Participation Matters', 'Don’t Skip Class', 'Lots of Homework',
    'Inspirational', 'Pop Quizzes!', 'Accessible', 'So Many Papers',
    'Clear Grading', 'Hilarious', 'Test Heavy', 'Graded by Few Things',
    'Amazing Lectures', 'Caring', 'Extra Credit', 'Group Projects',
    'Lecture Heavy'
]

# Step 1: Merge the datasets
combined_data = pd.concat([num_data[['Average Rating', 'Number of Ratings']], tags_data], axis=1)

# Step 2: Calculate the mean of 'Number of Ratings' and filter rows
mean_num_ratings = combined_data['Number of Ratings'].mean()
combined_data = combined_data[combined_data['Number of Ratings'] >= mean_num_ratings]

# Step 3: Drop rows with any NaN values
combined_data = combined_data.dropna()

# Step 4: Normalize tags by the number of ratings
for col in tags_data.columns:
    combined_data[col] = combined_data[col] / combined_data['Number of Ratings']

# Preparing features and target
X_tags = combined_data.iloc[:, 2:]  # All normalized tag columns as features
y_tags = combined_data['Average Rating']  # Average Rating as the target
scaler = StandardScaler()
X_scaled_tags = scaler.fit_transform(X_tags)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_tags, y_tags, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    "Ridge": Ridge()
}

# Step 3: Fit and evaluate each model
evaluation_results = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Store the evaluation metrics
    evaluation_results[model_name] = {
        'RMSE': rmse,
        'R² Score': r2
    }

# Extracting the model names and their evaluation metrics for plotting
model_names = list(evaluation_results.keys())
mse_values = [evaluation_results[model]['RMSE'] for model in model_names]
r2_values = [evaluation_results[model]['R² Score'] for model in model_names]

# Plotting MSE for each model
plt.figure(figsize=(10, 6))
plt.barh(model_names, mse_values, color='skyblue')
plt.xlabel("Root Mean Squared Error (RMSE)")
plt.title("Root Mean Squared Error for Different Models")
plt.gca().invert_yaxis()  # Invert the axis to have the best performer on top
plt.show()

# Plotting R² Score for each model
plt.figure(figsize=(10, 6))
plt.barh(model_names, r2_values, color='lightgreen')
plt.xlabel("R² Score")
plt.title("R² Score for Different Models")
plt.gca().invert_yaxis()  # Invert the axis to have the best performer on top
plt.show()

# Displaying evaluation results
print(evaluation_results)

# Train the Ridge model explicitly
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Extract coefficients (acts as feature importance in Ridge regression)
coefficients = ridge_model.coef_

# Map coefficients to feature names
importance_df = pd.DataFrame({
    'Feature': X_tags.columns,  # Use the feature names
    'Importance': np.abs(coefficients)  # Absolute value of coefficients
}).sort_values(by='Importance', ascending=False)

# Display the most important features
print("Top features:")
print(importance_df.head())

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance (Absolute Coefficients)')
plt.ylabel('Tags')
plt.title('Ridge Regression Feature Importances for Predicting Average Rating')
plt.gca().invert_yaxis()  # Best predictors on top
plt.show()

print('============Q9============')
num = pd.read_csv('rmpCapstoneNum.csv', header=None)
qual = pd.read_csv('rmpCapstoneQual.csv', header=None)
tag = pd.read_csv('rmpCapstoneTags.csv', header=None)

num_columns = [
    "Average Rating",            # The arithmetic mean of all individual quality ratings of this professor
    "Average Difficulty",        # The arithmetic mean of all individual difficulty ratings of this professor
    "Number of Ratings",         # Total number of ratings these averages are based on
    "Received a 'pepper'?",      # Boolean - judged as "hot" by the students
    "Proportion Retake",         # Proportion of students that said they would take the class again
    "Online Ratings Count",      # Number of ratings coming from online classes
    "Male Gender",               # Boolean – 1: determined with high confidence that professor is male
    "Female Gender"              # Boolean – 1: determined with high confidence that professor is female
]
num.columns = num_columns

qual_columns = [
    "Major/Field",  # Column 1: Major/Field
    "University",   # Column 2: University
    "US State"      # Column 3: US State (2-letter abbreviation)
]
qual.columns = qual_columns

tags_columns = [
    "Tough grader",              # Column 1
    "Good feedback",             # Column 2
    "Respected",                 # Column 3
    "Lots to read",              # Column 4
    "Participation matters",     # Column 5
    "Don't skip class",          # Column 6
    "Lots of homework",          # Column 7
    "Inspirational",             # Column 8
    "Pop quizzes!",              # Column 9
    "Accessible",                # Column 10
    "So many papers",            # Column 11
    "Clear grading",             # Column 12
    "Hilarious",                 # Column 13
    "Test heavy",                # Column 14
    "Graded by few things",      # Column 15
    "Amazing lectures",          # Column 16
    "Caring",                    # Column 17
    "Extra credit",              # Column 18
    "Group projects",            # Column 19
    "Lecture heavy"              # Column 20
]
tag.columns = tags_columns

# Ensure the datasets have the same number of records
assert len(num) == len(qual) == len(tag), "Datasets lengths do not match."

# Merge the datasets
merged_df = pd.concat([num, qual, tag], axis=1)

# drop na of average rating 
merged_df = merged_df.dropna(subset=['Average Difficulty'])
merged_df.head()

# check na
print('before cleaning NA', len(merged_df))
print(merged_df[tags_columns].isnull().sum())

print(merged_df['Number of Ratings'].describe())
plt.hist(merged_df['Number of Ratings'], bins=100)
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.title('Number of Ratings')
plt.show()

# set the threshold to 5 ratings and exclude the professors with less than 5 ratings
print('before dropping n rating', len(merged_df))
merged_df = merged_df[merged_df['Number of Ratings'] >= 5]
print('after dropping n rating', len(merged_df))

# For each tag column
for tag_column in tags_columns:
    # Create new normalized column
    merged_df[f'{tag_column}_normalized'] = merged_df[tag_column] / merged_df['Number of Ratings']
# create a list of normalized cols
normalized_columns = [f'{tag_column}_normalized' for tag_column in tags_columns]

import matplotlib.pyplot as plt
import numpy as np

# Calculate the correlation matrix
correlation_matrix = merged_df[normalized_columns].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 12))
plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)  # Use a diverging colormap
plt.xticks(range(len(normalized_columns)), normalized_columns, rotation=90)
plt.yticks(range(len(normalized_columns)), normalized_columns)
plt.colorbar(label='Correlation Coefficient')  # Add label for clarity
plt.title('Correlation Matrix', y=1.2)  # Add title with some spacing
plt.show()

# 1. Prepare features
feature_columns = (
    normalized_columns
)

X = merged_df[feature_columns]
y = merged_df['Average Difficulty']

# 2. Single train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model definitions with proper parameter grids
ridge_params = {'alpha': np.logspace(-3, 4, 50)} 
lasso_params = {'alpha': np.logspace(-3, 4, 50)}

# 5. Initialize models
ridge = Ridge(random_state=42)
lasso = Lasso(random_state=42)

# 6. Initialize cross-validation for training data
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 7. Train and evaluate models
models = {
    'Ridge': (ridge, ridge_params),
    'Lasso': (lasso, lasso_params),
}

results = {}
for name, (model, params) in models.items():
    # Perform GridSearchCV on training data
    grid_search = GridSearchCV(
        model, 
        params, 
        cv=kf, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit on training data
    grid_search.fit(X_train_scaled, y_train)
    
    # Get cross-validation results
    cv_results = {
        'CV_RMSE': np.sqrt(-grid_search.cv_results_['mean_test_score']),
        'CV_RMSE_std': np.sqrt(grid_search.cv_results_['std_test_score'])
    }
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate final test metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'CV_RMSE': cv_results['CV_RMSE'][grid_search.best_index_],
        'CV_RMSE_std': cv_results['CV_RMSE_std'][grid_search.best_index_],
        'Test_RMSE': test_rmse,
        'Test_R2': test_r2,
        'Best_Params': grid_search.best_params_
    }

# 8. Display results
results_df = pd.DataFrame.from_dict(results, orient='index')
print("\nModel Performance Results:")
print("\nCross-validation results (on training data):")
print(results_df[['CV_RMSE', 'CV_RMSE_std']])
print("\nTest set results:")
print(results_df[['Test_RMSE', 'Test_R2']])

print("\nBest Parameters:")
for model_name, result in results.items():
    print(f"\n{model_name}:")
    print(result['Best_Params'])

    # create a visualization of the results
import matplotlib.pyplot as plt

# Extract RMSE and R2 values
rmse_values = results_df['Test_RMSE']
r2_values = results_df['Test_R2']

# Create a figure and axis
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot RMSE
rmse_values.plot(kind='bar', ax=ax[0], color='skyblue')
ax[0].set_title('RMSE of Models')
ax[0].set_ylabel('RMSE')
ax[0].set_xlabel('Model')

# Plot R2
r2_values.plot(kind='bar', ax=ax[1], color='salmon')
ax[1].set_title('R2 of Models')
ax[1].set_ylabel('R2')
ax[1].set_xlabel('Model')

# Display the plot
plt.tight_layout()
plt.show()

# We use Lasso regression to identify the most important features
# Fit the Lasso model with the best parameters
lasso_best = Lasso(**results['Lasso']['Best_Params'])
lasso_best.fit(X_train_scaled, y_train)

# Extract the coefficients
lasso_coefs = lasso_best.coef_

# Create a DataFrame to display the coefficients
lasso_coefs_df = pd.DataFrame(lasso_coefs, index=feature_columns, columns=['Coefficient'])

# Sort the coefficients by absolute value
lasso_coefs_df['Coefficient_abs'] = np.abs(lasso_coefs_df['Coefficient'])
lasso_coefs_df = lasso_coefs_df.sort_values(by='Coefficient_abs', ascending=False)

# Display the sorted coefficients
print("\nLasso Coefficients (Sorted):")
print(lasso_coefs_df)

# Plot the coefficients
plt.figure(figsize=(10, 6))
plt.barh(lasso_coefs_df.index, lasso_coefs_df['Coefficient_abs'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Lasso Coefficients')
plt.grid(axis='x')
plt.show()

print('============Q10============')
print('before cleaning NA', len(merged_df))
print(merged_df.isnull().sum().sort_values(ascending=False))

merged_df['Proportion Retake'].describe()
# plot it
plt.hist(merged_df['Proportion Retake'])
plt.show()

merged_df = merged_df.dropna(subset='Proportion Retake')

x_cols = list(set(normalized_columns).union(set(num_columns)))
x_cols.remove("Received a 'pepper'?")
y = merged_df["Received a 'pepper'?"]
X = merged_df[x_cols]

print(y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features and apply SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Define expanded set of models with parameter grids
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'class_weight': ['balanced', None],
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', None]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'scale_pos_weight': [1, 3],
            'class_weight': ['balanced', None]
        }
    },
}

# Train and evaluate models with cross-validation
results = {}
predictions = {}
cv_results = {}  # New dictionary to store cross-validation results
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Define KFold cross-validator

for name, model_info in models.items():
    print(f"\nTraining {name}...")
    
    # Grid search
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Fit
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    # Perform cross-validation with the best model
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, 
                              cv=kf, scoring='roc_auc', n_jobs=-1)
    
    # Store cross-validation results
    cv_results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    # Predict
    y_pred_proba = grid_search.predict_proba(X_test_scaled)[:, 1]
    y_pred = grid_search.predict(X_test_scaled)
    
    # Store results
    results[name] = {
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'Best Params': grid_search.best_params_
    }
    
    predictions[name] = {
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    # Plot ROC curves (with multiple subplots for better visibility)
n_cols = 2
n_rows = (len(models) + 1) // 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.ravel()

for idx, (name, preds) in enumerate(predictions.items()):
    fpr, tpr, _ = roc_curve(y_test, preds['y_pred_proba'])
    auc = roc_auc_score(y_test, preds['y_pred_proba'])
    
    axes[idx].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    axes[idx].plot([0, 1], [0, 1], 'k--')
    axes[idx].set_xlabel('False Positive Rate')
    axes[idx].set_ylabel('True Positive Rate')
    axes[idx].set_title(f'{name} ROC Curve')
    axes[idx].legend()
    axes[idx].grid(True)

# Remove empty subplots if any
for idx in range(len(models), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# Plot confusion matrices
n_cols = 4
n_rows = (len(models) + 3) // 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.ravel()

for idx, (name, preds) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, preds['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
    axes[idx].set_title(f'{name}\nConfusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')

# Remove empty subplots if any
for idx in range(len(models), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# Create performance comparison bar plot
metrics_df = pd.DataFrame(
    [(name, results[name]['AUC-ROC']) 
     for name in results.keys()],
    columns=['Model', 'AUC-ROC']
).melt(id_vars=['Model'], var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(data=metrics_df, x='Model', y='Score', hue='Metric')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Print detailed results
print("\nModel Performance Summary:")
print("=" * 50)
for name, preds in predictions.items():
    print(f"\n{name}:")
    print("-" * 20)
    print(f"Precision: {precision_score(y_test, preds['y_pred'], average='weighted'):.3f}")
    print(f"Recall: {recall_score(y_test, preds['y_pred'], average='weighted'):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, preds['y_pred']):.3f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, preds['y_pred_proba']):.3f}")

# Print detailed results for best model
print("\nBest Model Performance Summary:")
print("=" * 50)
best_model_name = max(results, key=lambda x: results[x]['AUC-ROC'])
best_model_preds = predictions[best_model_name]
print(f"\n{best_model_name}:")
print("-" * 20)
print(f"Best Parameters: {results[best_model_name]['Best Params']}")
print(f"Precision: {precision_score(y_test, best_model_preds['y_pred'], average='weighted'):.3f}")
print(f"Recall: {recall_score(y_test, best_model_preds['y_pred'], average='weighted'):.3f}")
print(f"Accuracy: {accuracy_score(y_test, best_model_preds['y_pred']):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, best_model_preds['y_pred_proba']):.3f}")

print('============Extra Credit============')
print(merged_df["Major/Field"].value_counts())

math_rating = merged_df[merged_df["Major/Field"] == "Mathematics"]["Average Rating"]
english_rating = merged_df[merged_df["Major/Field"] == "English"]["Average Rating"]

plt.figure(figsize=(10, 6))
sns.kdeplot(math_rating, label="Mathematics", color='skyblue', shade=True)
sns.kdeplot(english_rating, label="English", color='salmon', shade=True)
plt.xlabel("Average Rating")
plt.ylabel("Density")
plt.title("Distribution of Ratings by Major")
plt.legend()
plt.show()

# Perform the Kolmogorov-Smirnov test
ks_stat, p_value = ks_2samp(math_rating, english_rating)
print(f"KS Statistic: {ks_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

# Perform the Mann-Whitney U test
u_stat, p_value = mannwhitneyu(math_rating, english_rating)
print(f"U Statistic: {u_stat:.4f}")
print(f"P-Value: {p_value:.4f}")



