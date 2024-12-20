"""#Question 1import pandas as pdimport scipy.stats as stats# Load the datasetfilelocation = '/users/kevinwang/desktop/rmpCapstoneNum.csv'num_data = pd.read_csv(filelocation, header=None)#I use header= None here because the CSV file doesn't contain any headers# Assign appropriate column namesnum_data.columns = [    "Average Rating", "Average Difficulty", "Number of Ratings", "Received Pepper",    "Proportion Retake", "Online Ratings", "Male gender", "Female"]# Extract ratings based on genderrating_male = num_data[num_data['Male gender'] == 1]['Average Rating']rating_female = num_data[num_data['Female'] == 1]['Average Rating']# Perform an independent t-test to compare average ratings for male and female professorst_stat, p_value = stats.ttest_ind(rating_male.dropna(), rating_female.dropna(), equal_var=False)#since independent t-tests assume equal variance, but I can't make that assumption, so I set it to false# Print results in consoleprint(f"t-statistic: {t_stat}, p-value: {p_value}")""""""# Question 2import pandas as pdimport scipy.stats as stats# Load the datasetnum_data_path = '/users/kevinwang/desktop/rmpCapstoneNum.csv'num_data = pd.read_csv(num_data_path, header=None)# Assign appropriate column namesnum_data.columns = [    "Average Rating", "Average Difficulty", "Number of Ratings", "Received Pepper",    "Proportion Retake", "Online Ratings", "Male gender", "Female"]# Extract ratings based on genderrating_male = num_data[num_data['Male gender'] == 1]['Average Rating']rating_female = num_data[num_data['Female'] == 1]['Average Rating']# Perform Levene's test to compare variances (spread) between male and female ratingslevene_stat, levene_p_value = stats.levene(    rating_male.dropna(),    rating_female.dropna())# Print results of the Levene's testprint(f"Levene's statistic: {levene_stat}, p-value: {levene_p_value}")# Calculate the variance for male and female ratingsvariance_male = rating_male.var()variance_female = rating_female.var()# Print variancesprint(f"Variance (Male): {variance_male}, Variance (Female): {variance_female}")""""""#Question 3import pandas as pdimport numpy as npfrom scipy.stats import ttest_ind, f# Load the dataset from the provided pathfile_path = '/users/kevinwang/desktop/rmpCapstoneNum.csv'rmpCapstoneNum = pd.read_csv(file_path)# Clean and preprocess datarmpCapstoneNum.columns = [    "Average Rating", "Average Difficulty", "Number of Ratings",    "Received Pepper", "Proportion Take Again", "Online Ratings",    "Male", "Female"]# Remove rows with missing Average Rating or Male/Female columnscleaned_data = rmpCapstoneNum.dropna(subset=["Average Rating", "Male", "Female"])# Separate data by gendermale_ratings = cleaned_data[cleaned_data["Male"] == 1]["Average Rating"]female_ratings = cleaned_data[cleaned_data["Female"] == 1]["Average Rating"]# Calculate means and variances for gender groupsmale_mean = np.mean(male_ratings)female_mean = np.mean(female_ratings)male_var = np.var(male_ratings, ddof=1)female_var = np.var(female_ratings, ddof=1)# Perform t-tests for meansmean_diff = male_mean - female_meant_stat, p_value_mean = ttest_ind(male_ratings, female_ratings, equal_var=False)# Confidence interval for mean differencese_diff = np.sqrt(male_var / len(male_ratings) + female_var / len(female_ratings))ci_lower_mean = mean_diff - 1.96 * se_diffci_upper_mean = mean_diff + 1.96 * se_diff# Variance ratiovariance_ratio = male_var / female_vardf1, df2 = len(male_ratings) - 1, len(female_ratings) - 1# Confidence interval for variance ratio alpha = 0.05f_critical_low = f.ppf(alpha / 2, df1, df2)f_critical_up = f.ppf(1 - alpha / 2, df1, df2)ci_lower_var = variance_ratio / f_critical_upci_upper_var = variance_ratio / f_critical_low# I am printing the resultsresults = {    "Male Mean Rating": male_mean,    "Female Mean Rating": female_mean,    "Mean Difference": mean_diff,    "95% CI for Mean Difference": (ci_lower_mean, ci_upper_mean),    "P-Value for Mean Difference": p_value_mean,    "Male Variance": male_var,    "Female Variance": female_var,    "Variance Ratio (Male/Female)": variance_ratio,    "95% CI for Variance Ratio": (ci_lower_var, ci_upper_var),}print(results)""""""#Question 4# Import necessary librariesimport pandas as pdfrom scipy.stats import chi2_contingency# Load the datasets (replace with your paths if necessary)tags_data_path = '/users/kevinwang/desktop/rmpCapstoneTags.csv'num_data_path = '/users/kevinwang/desktop/rmpCapstoneNum.csv'rmpCapstoneTags = pd.read_csv(tags_data_path)rmpCapstoneNum = pd.read_csv(num_data_path)# Add meaningful column names to the tags datasettag_columns = [    "Tough Grader", "Good Feedback", "Respected", "Lots to Read", "Participation Matters",    "Don’t Skip Class", "Lots of Homework", "Inspirational", "Pop Quizzes!", "Accessible",    "So Many Papers", "Clear Grading", "Hilarious", "Test Heavy", "Graded by Few Things",    "Amazing Lectures", "Caring", "Extra Credit", "Group Projects", "Lecture Heavy"]rmpCapstoneTags.columns = tag_columns# Add the gender information from the numerical dataset to the tags datasetrmpCapstoneTags["Male"] = rmpCapstoneNum.iloc[:, 6]  # Male columnrmpCapstoneTags["Female"] = rmpCapstoneNum.iloc[:, 7]  # Female column# Prepare results dictionarytag_results = []# Perform a chi-square test for each tagfor tag in tag_columns:    male_counts = rmpCapstoneTags[rmpCapstoneTags["Male"] == 1][tag]    female_counts = rmpCapstoneTags[rmpCapstoneTags["Female"] == 1][tag]        # Create a contingency table    contingency_table = pd.DataFrame({        "Male": male_counts.value_counts(),        "Female": female_counts.value_counts()    }).fillna(0)        # Perform chi-square test    chi2, p_value, _, _ = chi2_contingency(contingency_table)        # Store the results    tag_results.append({"Tag": tag, "Chi2": chi2, "P-Value": p_value})# Convert results to a DataFrame and sort by p-valuetag_results_df = pd.DataFrame(tag_results)tag_results_df = tag_results_df.sort_values(by="P-Value")# Display the top 3 most gendered and least gendered tagsmost_gendered_tags = tag_results_df.head(3)least_gendered_tags = tag_results_df.tail(3)# Save or print resultsprint("Most Gendered Tags:")print(most_gendered_tags)print("\nLeast Gendered Tags:")print(least_gendered_tags)"""