import multiprocessing
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets

def train_model(n_factors, n_users, n_items, train_data, valid_data):
    model = CollabFilterOneVectorPerItem(n_epochs=600, batch_size=32, step_size=0.2, n_factors=n_factors, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_data)
    model.fit(train_data, valid_data)
    return model.trace_loss, model.trace_smooth_loss

if __name__ == '__main__':
    # Load the dataset
    train_data, valid_data, test_data, n_users, n_items = load_train_valid_test_datasets()

    # Create a pool of worker processes
    pool = multiprocessing.Pool()

    # Define the hyperparameter values for each model
    hyperparams = [50, 10, 2]

    # Train models in parallel using multiprocessing
    results = [pool.apply_async(train_model, args=(n_factors, n_users, n_items, train_data, valid_data)) for n_factors in hyperparams]

    # Retrieve the results from the worker processes
    train_loss_histories = []
    valid_loss_histories = []
    for result in results:
        train_loss, valid_loss = result.get()
        train_loss_histories.append(train_loss)
        valid_loss_histories.append(valid_loss)

    # Close the pool of worker processes
    pool.close()
    pool.join()

    # Create the plots using the stored loss history lists
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(train_loss_histories[2], label='Training Loss')
    ax1.plot(valid_loss_histories[2], label='Validation Loss')
    ax1.set_title('K=2 Factors')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.legend()

    ax2.plot(train_loss_histories[1], label='Training Loss')
    ax2.plot(valid_loss_histories[1], label='Validation Loss')
    ax2.set_title('K=10 Factors')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()

    ax3.plot(train_loss_histories[0], label='Training Loss')
    ax3.plot(valid_loss_histories[0], label='Validation Loss')
    ax3.set_title('K=50 Factors')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.legend()

    plt.tight_layout()
    plt.show()