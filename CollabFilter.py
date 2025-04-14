# Import needed libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt

# Initialize the CollabFilter class to store all CF functions and external import
class CollabFilter:
    @staticmethod
    # Function to pare down the data to only the columns that we need
    def prepare_data(fashion_df):
        # Define the columns to keep
        cols_to_keep = ['Customer Reference ID', 'Item Purchased', 'Review Rating']
        # Update the dataframe to only include these columns
        fashion_df = fashion_df[cols_to_keep]
        # Return the updated dataframe
        return fashion_df

    @staticmethod
    # Function to create the user-item matrix
    def create_user_item_matrix(fashion_df):
        # Create a user-item matrix from the dataframe
        user_item = pd.pivot_table(
            # Define the data that will make up the user-item matrix
            fashion_df,
            # Each row represents a unique customer
            index='Customer Reference ID',
            # Each column represents a unique item
            columns='Item Purchased',
            # Each value are the rating reviews
            values='Review Rating')
        # Return the created user-item matrix
        return user_item

    @staticmethod
    # Function to compute the similarity matrix for item-item CF
    def compute_item_similarity_matrix(user_item_matrix):
        # Fill missing values with 0 for the user-item matrix
        filled_matrix = user_item_matrix.fillna(0)
        # Create an empty DataFrame to store similarity values between items
        similarity_matrix = pd.DataFrame(index=filled_matrix.columns, columns=filled_matrix.columns)

        # Loop through all pairs of items
        for item1 in filled_matrix.columns:
            for item2 in filled_matrix.columns:
                # If comparing the same item to itself
                if item1 == item2:
                    # Make the similarity score 1
                    similarity_matrix.at[item1, item2] = 1.0
                # If were not comparing the same item to itself
                else:
                    # Get the ratings for the two items
                    vec1 = filled_matrix[item1].values
                    vec2 = filled_matrix[item2].values
                    try:
                        # Compute cosine similarity for the items
                        sim = 1 - cosine(vec1, vec2)
                        similarity_matrix.at[item1, item2] = sim
                    except ValueError:
                        # In case of any error set similarity to 0
                        similarity_matrix.at[item1, item2] = 0.0

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        # Scale the similarity matrix values between 0 and 1
        similarity_matrix = pd.DataFrame(scaler.fit_transform(similarity_matrix),
                                         columns=similarity_matrix.columns,
                                         index=similarity_matrix.index)

        # Return the scaled similarity matrix
        return similarity_matrix.astype(float)

    @staticmethod
    # Function to predict the ratings for each user based on k=7 items
    def predict_ratings_for_user(user_item_matrix, similarity_matrix, target_user, k=7):
        # Get the ratings of the target user from the user-item matrix
        user_ratings = user_item_matrix.loc[target_user]
        # Create a copy of the user's ratings to store the predicted ratings
        predicted_ratings = user_ratings.copy()

        # Loop through each item the user has rated
        for item in user_ratings.index:
            # If the item has not been rated, predict its rating
            if pd.isna(user_ratings[item]):
                # Get the list of items the user has rated
                rated_items = user_ratings[user_ratings.notna()].index

                # Get similarities between the current item and all the rated items by the user
                similar_items = similarity_matrix[item][rated_items].dropna()
                # Sort the similar items and pick the top-k most similar ones
                top_k_similar_items = similar_items.sort_values(ascending=False).head(k)

                # Initialize variables to calculate the predicted rating
                numerator = 0
                denominator = 0

                # Loop through the top-k most similar items
                for sim_item, sim_score in top_k_similar_items.items():
                    # Get the rating for the similar item
                    rating = user_ratings[sim_item]
                    # Update the numerator and denominator for the weighted sum of ratings
                    numerator += sim_score * rating
                    denominator += sim_score

                # If the denominator is not zero
                if denominator != 0:
                    # Calculate the predicted rating
                    predicted_ratings[item] = numerator / denominator
                # If the denominator is zero
                else:
                    # Assign the average rating of the user
                    predicted_ratings[item] = user_ratings.mean()

        # Return the predicted ratings for the user
        return predicted_ratings

    @staticmethod
    # Function to get the top k=3 items
    def get_top_k_recommendations(predicted_ratings, original_ratings, k=3):
        # Filter the predicted ratings to include only items the user has NOT rated
        unrated_predictions = predicted_ratings[original_ratings.isna()]

        # Sort the unrated items by predicted rating in descending order and take the top k items
        top_k_recs = unrated_predictions.sort_values(ascending=False).head(k)

        # Return the top k recommended items
        return top_k_recs


class Eval_Model:
    @staticmethod
    # Function to evaluate the CF model
    def evaluate_data(original_fashion_df, k=7):
        # Split the data into training and testing datasets
        train_df, test_df = train_test_split(original_fashion_df, test_size=0.5, random_state=42)

        # Prepare both the training and testing datasets for collaborative filtering
        train_df = CollabFilter.prepare_data(train_df.copy())
        test_df = CollabFilter.prepare_data(test_df.copy())

        # Create user-item matrix for training data
        train_user_item_matrix = CollabFilter.create_user_item_matrix(train_df)
        # Compute similarity matrix for items in the training data
        similarity_matrix = CollabFilter.compute_item_similarity_matrix(train_user_item_matrix)

        # Initialize and empty list to store the predictions
        predictions = []
        # Initialize and empty list to store the actual values
        actuals = []

        # Loop through each row in the test set
        for _, row in test_df.iterrows():
            # Extract user ID from the test row
            user = row['Customer Reference ID']
            # Extract the item ID from the test row
            item = row['Item Purchased']
            # Extract the actual rating from the test row
            actual_rating = row['Review Rating']

            # Check if the user exists in the training matrix
            if user in train_user_item_matrix.index:
                # If they do, predict all ratings for the user using the collaborative filtering model
                predicted_ratings = CollabFilter.predict_ratings_for_user(train_user_item_matrix, similarity_matrix,
                                                                          user, k=k)

                # Check if the predicted rating for the item exists and is valid
                if item in predicted_ratings.index and pd.notna(predicted_ratings[item]):
                    # Store the predicted rating for evaluation
                    predictions.append(predicted_ratings[item])
                    # Store the actual rating for comparison
                    actuals.append(actual_rating)

        # Initialize a list to store the clean actuals
        clean_actuals = []

        # Loop through each actual rating
        for a in actuals:
            # Check if the actual rating is not NaN
            if pd.notna(a):
                # If valid, append the actual rating to the cleaned list
                clean_actuals.append(a)
            else:
                # If NaN, replace with 0 and append
                clean_actuals.append(0)

        # Initialize a list to store the clean predictions
        clean_predictions = []

        # Loop through each predicted rating
        for p in predictions:
            # Check if the predicted rating is not NaN
            if pd.notna(p):
                # If valid, append the predicted rating to the cleaned list
                clean_predictions.append(p)
            else:
                # If NaN, replace with 0 and append
                clean_predictions.append(0)

        # Reassign the cleaned lists back to original variables
        actuals = clean_actuals
        predictions = clean_predictions

        # Compute and return the evaluation metrics (MSE, RMSE, MAE)
        if predictions and actuals:
            # Compute Mean Squared Error (MSE)
            mse = mean_squared_error(actuals, predictions)
            # Compute Root Mean Squared Error (RMSE)
            rmse = sqrt(mse)
            # Compute Mean Absolute Error (MAE)
            mae = mean_absolute_error(actuals, predictions)

            # Return the computed MSE, RMSE, and MAE
            return mse, rmse, mae

# Load in the fashion retail sales dataset
fashion_df = pd.read_csv('Fashion_Retail_Sales.csv')

# Create an instance of the CollabFilter Class
cf = CollabFilter()
# Prepare the data
fashion_df = cf.prepare_data(fashion_df)
# Create the user-item matrix
user_item_matrix = cf.create_user_item_matrix(fashion_df)
# Create the similarity score matrix
item_similarity_matrix = cf.compute_item_similarity_matrix(user_item_matrix)

# Choose a target user to use for the recommender system (5th user in the matrix)
target_user = user_item_matrix.index[4]
# Predict ratings for all items for the selected user with k=7
predicted = cf.predict_ratings_for_user(user_item_matrix, item_similarity_matrix, target_user, k=7)
# Get the original ratings of the selected user for comparison
original_ratings = user_item_matrix.loc[target_user]
# Get top 3 recommended items for the target user that they haven’t rated yet
top_3_recs = cf.get_top_k_recommendations(predicted, original_ratings, k=3)
# Print the top 3 recommendations
print("Top 3 Recommended Items:")
print(top_3_recs)

# Evaluate model performance using evaluation metrics on the dataset
mse, rmse, mae = Eval_Model.evaluate_data(fashion_df, k=7)
print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}")

# Define a range of k values to test model performance
k_values = [1, 5, 15, 20, 25, 30, 35, 40, 45, 50]

# Initialize empty lists to store evaluation metrics for each k
mse_values = []
rmse_values = []
mae_values = []

# Loop through each k values
for k in k_values:
    # Evaluate model for the given k
    mse, rmse, mae = Eval_Model.evaluate_data(fashion_df)

    # Append results their respective lists
    mse_values.append(mse)
    rmse_values.append(rmse)
    mae_values.append(mae)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_values, label="MSE", marker='o')
plt.plot(k_values, rmse_values, label="RMSE", marker='o')
plt.plot(k_values, mae_values, label="MAE", marker='o')
plt.xlabel("k (Number of Nearest Neighbors)")
plt.ylabel("Error")
plt.title("Evaluation Metrics for Different k Values")
plt.legend()
plt.grid(True)
plt.show()