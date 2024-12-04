import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# File paths for datasets
REVIEWS_FILE_PATH = "All_Beauty.jsonl"
META_FILE_PATH = "meta_All_Beauty.jsonl"


def load_data():
    """
    Load review and metadata files into Pandas DataFrames.

    Returns:
        tuple: Tuple containing reviews DataFrame and metadata DataFrame.
    """
    reviews_df = pd.read_json(REVIEWS_FILE_PATH, lines=True)
    meta_df = pd.read_json(META_FILE_PATH, lines=True)
    return reviews_df, meta_df


def filter_valid_entries(reviews_df, meta_df, min_user_interactions=1, min_item_interactions=1):
    """
    Filter out invalid entries and users/items with insufficient interactions.

    Args:
        reviews_df (pd.DataFrame): DataFrame containing review data.
        meta_df (pd.DataFrame): DataFrame containing metadata.
        min_user_interactions (int): Minimum number of interactions per user.
        min_item_interactions (int): Minimum number of interactions per item.

    Returns:
        tuple: Filtered reviews DataFrame and metadata DataFrame.
    """
    # Remove duplicate user-item interactions to prevent data skew
    reviews_df.drop_duplicates(subset=['user_id', 'parent_asin'], inplace=True)

    # Drop reviews missing essential columns
    required_columns_reviews = ['user_id', 'parent_asin', 'rating']
    reviews_df.dropna(subset=required_columns_reviews, inplace=True)

    # Ensure ratings are within the valid range
    reviews_df = reviews_df[(reviews_df['rating'] >= 1) & (reviews_df['rating'] <= 5)]

    # Drop metadata entries missing 'parent_asin'
    required_columns_meta = ['parent_asin']
    meta_df.dropna(subset=required_columns_meta, inplace=True)

    # Count interactions per user and per item
    user_counts = reviews_df['user_id'].value_counts()
    item_counts = reviews_df['parent_asin'].value_counts()

    # Identify users and items meeting the minimum interaction criteria
    filtered_users = user_counts[user_counts >= min_user_interactions].index
    filtered_items = item_counts[item_counts >= min_item_interactions].index

    # Filter reviews to include only valid users and items
    reviews_df = reviews_df[
        reviews_df['user_id'].isin(filtered_users) &
        reviews_df['parent_asin'].isin(filtered_items)
    ]

    return reviews_df, meta_df


def preprocess_data(reviews_df, meta_df):
    """
    Preprocess the data by selecting relevant columns, merging datasets, and encoding categorical variables.

    Args:
        reviews_df (pd.DataFrame): Filtered reviews DataFrame.
        meta_df (pd.DataFrame): Filtered metadata DataFrame.

    Returns:
        tuple:
            - merged_df (pd.DataFrame): Merged and preprocessed DataFrame.
            - num_users (int): Number of unique users.
            - num_items (int): Number of unique items.
            - user_encoder (dict): Mapping from user_id to integer index.
            - product_encoder (dict): Mapping from parent_asin to integer index.
            - meta_df (pd.DataFrame): Metadata DataFrame with relevant columns.
    """
    # Select relevant columns from reviews and metadata
    reviews_df = reviews_df[['user_id', 'parent_asin', 'rating']]
    meta_df = meta_df[['parent_asin', 'title', 'price']]

    # Merge reviews with metadata on 'parent_asin'
    merged_df = pd.merge(reviews_df, meta_df, on='parent_asin', how='left')

    # Drop any rows with missing values after merging
    merged_df.dropna(inplace=True)

    # Encode 'user_id' and 'parent_asin' to integer indices for matrix operations
    user_encoder = {user: idx for idx, user in enumerate(merged_df['user_id'].unique())}
    product_encoder = {product: idx for idx, product in enumerate(merged_df['parent_asin'].unique())}

    merged_df['user'] = merged_df['user_id'].map(user_encoder)
    merged_df['product'] = merged_df['parent_asin'].map(product_encoder)

    return merged_df, len(user_encoder), len(product_encoder), user_encoder, product_encoder, meta_df


class MatrixFactorization:
    def __init__(self, num_users, num_items, num_factors=10, lr=0.01,
                 reg_user=0.02, reg_item=0.02, reg_user_bias=0.02, reg_item_bias=0.02,
                 epochs=50, patience=5, decay_rate=0.0, trial=None):
        """
        Initialize the Matrix Factorization model with latent factors and bias terms.

        Args:
            num_users (int): Number of unique users.
            num_items (int): Number of unique items.
            num_factors (int): Number of latent factors.
            lr (float): Learning rate.
            reg_user (float): Regularization for user factors.
            reg_item (float): Regularization for item factors.
            reg_user_bias (float): Regularization for user bias.
            reg_item_bias (float): Regularization for item bias.
            epochs (int): Maximum number of training epochs.
            patience (int): Number of epochs to wait for improvement before early stopping.
            decay_rate (float): Learning rate decay per epoch.
            trial (optuna.trial.Trial, optional): Optuna trial for pruning.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.lr = lr  # Initial learning rate
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.reg_user_bias = reg_user_bias
        self.reg_item_bias = reg_item_bias
        self.epochs = epochs
        self.patience = patience
        self.decay_rate = decay_rate
        self.trial = trial  # For Optuna pruning

        # Initialize user and item latent factor matrices using Xavier initialization
        limit = np.sqrt(6 / (self.num_factors + self.num_factors))
        self.user_factors = np.random.uniform(-limit, limit, (self.num_users, self.num_factors))
        self.item_factors = np.random.uniform(-limit, limit, (self.num_items, self.num_factors))

        # Initialize bias terms
        self.user_biases = np.zeros(self.num_users)
        self.item_biases = np.zeros(self.num_items)
        self.global_bias = 0.0

    def train(self, train_ratings, val_ratings):
        """
        Train the Matrix Factorization model using Stochastic Gradient Descent (SGD) with early stopping.

        Args:
            train_ratings (np.ndarray): Training ratings array.
            val_ratings (np.ndarray): Validation ratings array.
        """
        # Compute the global bias as the mean of training ratings
        self.global_bias = np.mean(train_ratings[:, 2])

        best_val_loss = float('inf')  # Best validation loss observed
        epochs_no_improve = 0  # Counter for epochs without improvement
        best_params = None  # Placeholder for best model parameters

        initial_lr = self.lr  # Store initial learning rate for scheduling

        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            # Update learning rate with decay
            self.lr = initial_lr / (1 + self.decay_rate * epoch)

            # Shuffle training data for SGD
            np.random.shuffle(train_ratings)
            self.sgd(train_ratings)

            # Compute training and validation loss
            train_loss = self.compute_mse(train_ratings)
            val_loss = self.compute_mse(val_ratings)

            # Log metrics to WandB with the current epoch as the step
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'lr': self.lr}, step=epoch + 1)

            # Display progress
            tqdm.write(
                f"Epoch {epoch + 1}/{self.epochs}, LR: {self.lr:.6f}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Check for improvement in validation loss
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save current best parameters
                best_params = (self.user_factors.copy(), self.item_factors.copy(),
                               self.user_biases.copy(), self.item_biases.copy())
            else:
                epochs_no_improve += 1
                # If no improvement for 'patience' epochs, perform early stopping
                if epochs_no_improve >= self.patience:
                    tqdm.write("Early stopping!")
                    # Restore best parameters observed during training
                    self.user_factors, self.item_factors, self.user_biases, self.item_biases = best_params
                    break

            # Optional: Implement pruning with Optuna
            if self.trial is not None:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    # Log the current validation loss before pruning
                    wandb.log({'val_loss': val_loss}, step=epoch + 1)
                    wandb.finish()
                    raise optuna.exceptions.TrialPruned()

    def sgd(self, ratings):
        """
        Perform one epoch of Stochastic Gradient Descent (SGD).

        Args:
            ratings (np.ndarray): Array of user-item-rating triples.
        """
        for user, item, rating in ratings:
            user = int(user)
            item = int(item)
            prediction = self.predict_single(user, item)
            error = rating - prediction

            # Update user and item biases with regularization
            self.user_biases[user] += self.lr * (error - self.reg_user_bias * self.user_biases[user])
            self.item_biases[item] += self.lr * (error - self.reg_item_bias * self.item_biases[item])

            # Save current user factors for simultaneous updates
            user_factors_old = self.user_factors[user, :].copy()
            # Update user and item latent factors with regularization
            self.user_factors[user, :] += self.lr * (
                        error * self.item_factors[item, :] - self.reg_user * self.user_factors[user, :])
            self.item_factors[item, :] += self.lr * (
                        error * user_factors_old - self.reg_item * self.item_factors[item, :])

    def predict_single(self, user, item):
        """
        Predict the rating for a single user-item pair.

        Args:
            user (int): User index.
            item (int): Item index.

        Returns:
            float: Predicted rating.
        """
        prediction = (self.global_bias +
                      self.user_biases[user] +
                      self.item_biases[item] +
                      np.dot(self.user_factors[user, :], self.item_factors[item, :]))
        return prediction

    def predict(self, users, items):
        """
        Predict ratings for multiple user-item pairs.

        Args:
            users (np.ndarray): Array of user indices.
            items (np.ndarray): Array of item indices.

        Returns:
            np.ndarray: Array of predicted ratings.
        """
        predictions = (self.global_bias +
                       self.user_biases[users] +
                       self.item_biases[items] +
                       np.sum(self.user_factors[users] * self.item_factors[items], axis=1))
        return predictions

    def compute_mse(self, ratings):
        """
        Compute Mean Squared Error (MSE) for the given ratings.

        Args:
            ratings (np.ndarray): Array of user-item-rating triples.

        Returns:
            float: Computed MSE.
        """
        users = ratings[:, 0].astype(int)
        items = ratings[:, 1].astype(int)
        true_ratings = ratings[:, 2]
        predictions = self.predict(users, items)
        mse = mean_squared_error(true_ratings, predictions)
        return mse

    def predict_for_user(self, user):
        """
        Predict ratings for all items for a given user.

        Args:
            user (int): User index.

        Returns:
            np.ndarray: Predicted ratings for all items.
        """
        user = int(user)
        items = np.arange(self.num_items)
        users = np.full(self.num_items, user)
        predictions = self.predict(users, items)
        return predictions


def objective(trial, train_ratings, val_ratings, num_users, num_items):
    """
    Objective function for Optuna to minimize MSE with hyperparameter tuning.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        train_ratings (np.ndarray): Training ratings array.
        val_ratings (np.ndarray): Validation ratings array.
        num_users (int): Number of unique users.
        num_items (int): Number of unique items.

    Returns:
        float: Validation MSE for the trial.
    """
    # Suggest hyperparameters to be optimized
    num_factors = trial.suggest_int('num_factors', 10, 200)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    reg_user = trial.suggest_float('reg_user', 1e-6, 1e-1, log=True)
    reg_item = trial.suggest_float('reg_item', 1e-6, 1e-1, log=True)
    reg_user_bias = trial.suggest_float('reg_user_bias', 1e-6, 1e-1, log=True)
    reg_item_bias = trial.suggest_float('reg_item_bias', 1e-6, 1e-1, log=True)
    epochs = trial.suggest_int('epochs', 20, 100)
    patience = trial.suggest_int('patience', 5, 15)
    decay_rate = trial.suggest_float('decay_rate', 0.0, 0.1)

    # Aggregate hyperparameters into a dictionary
    hyperparams = {
        'num_factors': num_factors,
        'lr': lr,
        'reg_user': reg_user,
        'reg_item': reg_item,
        'reg_user_bias': reg_user_bias,
        'reg_item_bias': reg_item_bias,
        'epochs': epochs,
        'patience': patience,
        'decay_rate': decay_rate
    }

    # Create a unique run name based on hyperparameters
    run_name = "SVD_" + "_".join([f"{key}={value}" for key, value in hyperparams.items()])

    # Initialize WandB for this Optuna trial
    wandb.init(project='matrix_factorization', reinit=True, name=run_name)

    # Log hyperparameters to WandB
    wandb.config.update(hyperparams)

    # Initialize the Matrix Factorization model with suggested hyperparameters
    mf_model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        num_factors=num_factors,
        lr=lr,
        reg_user=reg_user,
        reg_item=reg_item,
        reg_user_bias=reg_user_bias,
        reg_item_bias=reg_item_bias,
        epochs=epochs,
        patience=patience,
        decay_rate=decay_rate,
        trial=trial  # Pass the trial for potential pruning
    )

    try:
        # Train the model with training and validation data
        mf_model.train(train_ratings, val_ratings)
    except optuna.exceptions.TrialPruned:
        # If the trial is pruned, finish the WandB run and raise the exception
        wandb.finish()
        raise optuna.exceptions.TrialPruned()

    # Compute validation MSE after training
    val_mse = mf_model.compute_mse(val_ratings)
    print(f"MSE for this trial: {val_mse:.4f}")

    # Log the final validation MSE to WandB
    wandb.log({'final_val_mse': val_mse}, step=epochs)
    wandb.finish()
    return val_mse


def main():
    """
    Main function to execute the training pipeline:
    - Load and preprocess data
    - Perform hyperparameter tuning with Optuna
    - Train the final model with the best hyperparameters
    - Evaluate and display top product recommendations for a random user
    """
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Ensure WandB is authenticated (you should have run 'wandb login' beforehand)
    wandb.login()

    # Load datasets
    reviews_df, meta_df = load_data()

    # Filter out invalid entries and users/items with insufficient interactions
    # Set min_user_interactions and min_item_interactions to 1 to include all users and items with at least one interaction
    reviews_df, meta_df = filter_valid_entries(reviews_df, meta_df, min_user_interactions=1, min_item_interactions=1)

    # Preprocess data: select relevant columns, merge datasets, and encode categorical variables
    merged_df, num_users, num_items, user_encoder, product_encoder, meta_df = preprocess_data(reviews_df, meta_df)

    # Create a reverse mapping from user index to user_id for user-friendly outputs
    user_decoder = {idx: user for user, idx in user_encoder.items()}

    # Create a reverse mapping from product index to parent_asin
    product_decoder = {idx: parent_asin for parent_asin, idx in product_encoder.items()}

    # Extract ratings as a NumPy array for efficient processing
    ratings = merged_df[['user', 'product', 'rating']].values

    # Split data into training and testing sets (80% train, 20% test)
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    # Further split training data into training and validation sets (80% train, 20% validation of training data)
    train_ratings, val_ratings = train_test_split(train_ratings, test_size=0.2, random_state=42)

    # Display the sizes of each split
    print(f"Training set size: {train_ratings.shape[0]} samples")
    print(f"Validation set size: {val_ratings.shape[0]} samples")
    print(f"Test set size: {test_ratings.shape[0]} samples")

    # Initialize and run Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, train_ratings, val_ratings, num_users, num_items),
        n_trials=50  # Increased number of trials for better hyperparameter exploration
    )

    # Display the best trial results
    print(f"Best trial MSE: {study.best_trial.value:.4f}")
    print(f"Best parameters: {study.best_trial.params}")

    # Initialize WandB for final model training with the best hyperparameters
    final_run_name = "Final_Model_Training"
    wandb.init(project='matrix_factorization', reinit=True, name=final_run_name)

    # Log the best hyperparameters to WandB
    wandb.config.update(study.best_trial.params)

    # Extract the best hyperparameters
    best_params = study.best_trial.params

    # Initialize the Matrix Factorization model with the best hyperparameters
    best_model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        num_factors=best_params['num_factors'],
        lr=best_params['lr'],
        reg_user=best_params['reg_user'],
        reg_item=best_params['reg_item'],
        reg_user_bias=best_params['reg_user_bias'],
        reg_item_bias=best_params['reg_item_bias'],
        epochs=best_params['epochs'],
        patience=best_params['patience'],
        decay_rate=best_params['decay_rate']
    )

    # Combine training and validation sets for final training
    combined_train_val = np.vstack((train_ratings, val_ratings))
    print("Training the final model on combined training and validation sets...")
    best_model.train(combined_train_val, val_ratings)

    # Compute final MSE on training, validation, and test sets
    final_train_mse = best_model.compute_mse(combined_train_val)
    final_val_mse = best_model.compute_mse(val_ratings)
    final_test_mse = best_model.compute_mse(test_ratings)

    # Print the final MSEs to the console
    print(f"Final Training MSE: {final_train_mse:.4f}")
    print(f"Final Validation MSE: {final_val_mse:.4f}")
    print(f"Final Test MSE: {final_test_mse:.4f}")

    # Log the final MSEs to WandB
    wandb.log({
        'final_train_mse': final_train_mse,
        'final_val_mse': final_val_mse,
        'final_test_mse': final_test_mse
    }, step=best_params['epochs'])
    wandb.finish()

    # Select a random user from the test set for generating recommendations
    # Retrieve the unique user indices from the test set
    unique_test_users = np.unique(test_ratings[:, 0])
    random_user_idx = int(np.random.choice(unique_test_users))

    # Map the integer index back to the original user_id
    user_id = user_decoder[random_user_idx]
    print(f"\nGenerating top 10 product recommendations for User ID {user_id}...")

    # Predict ratings for all items for the selected user
    predictions = best_model.predict_for_user(random_user_idx)

    # Identify the top 10 items with the highest predicted ratings
    top_10_product_indices = predictions.argsort()[-10:][::-1]

    # Decode item indices back to 'parent_asin'
    top_10_asins = [product_decoder[idx] for idx in top_10_product_indices]

    # Retrieve product details from metadata
    top_10_products = meta_df[meta_df['parent_asin'].isin(top_10_asins)]

    # Display the top 10 recommended products
    print(f"\nTop 10 Products for User ID {user_id}:")
    if not top_10_products.empty:
        for _, row in top_10_products.iterrows():
            print(f"Product Title: {row['title']}, Parent ASIN: {row['parent_asin']}, Price: {row['price']}")
    else:
        print("No product information available for the recommended items.")


if __name__ == "__main__":
    main()
