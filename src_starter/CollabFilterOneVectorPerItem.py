'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=random_state.randn(n_users), # FIX dimensionality
            c_per_item=random_state.randn(n_items), # FIX dimensionality
            U=ag_np.array(0.01 * random_state.randn(n_users, self.n_factors)), # FIX dimensionality
            V=0.01 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic

        # Gather all the parameters from dict if they are not given
        # if (mu == None):
        mu = self.param_dict['mu']
        # if (b_per_user == None):
        b_per_user = self.param_dict['b_per_user']
        # if (c_per_item == None):
        c_per_item = self.param_dict['c_per_item']
        # if (U == None):
        U = self.param_dict['U']
        # if (V == None):
        V = self.param_dict['V']
        
        # Get indices from inputted array
        user_indices = user_id_N.astype(int)
        item_indices = item_id_N.astype(int)

        # Calculate predictions for the ratings
        yhat_N = mu + b_per_user[user_indices] + c_per_item[item_indices] + ag_np.sum(U[user_indices] * V[item_indices], axis=1)

        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        y_N = data_tuple[2]
        
        # Get predictions
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        # Get parameters from class dict
        mu = param_dict['mu']
        b_per_user = param_dict['b_per_user']
        c_per_item = param_dict['c_per_item']
        U = param_dict['U'][data_tuple[0]]
        V = param_dict['V'][data_tuple[1]]

        # Calculate MSE
        mse = ag_np.mean((y_N - yhat_N) ** 2)

        # Calculate regularization loss
        reg_loss = self.alpha * (ag_np.sum(b_per_user ** 2) + ag_np.sum(c_per_item ** 2) + ag_np.sum(U ** 2) + ag_np.sum(V ** 2))

        return mse + reg_loss  


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=2, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)