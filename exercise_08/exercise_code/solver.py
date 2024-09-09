import numpy as np
import torch
from torch.optim import Adam
from exercise_code.loss import Loss
import os

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    or regression models.
    The Solver performs gradient descent using the given learning rate.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, learning_rate to the constructor.
    You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_loss_history and solver.val_loss_history will be lists
    containing the losses of the model on the training and validation set at
    each epoch.
    """

    def __init__(self, model, train_dataloader, val_dataloader,
                 hparams, optimizer=Adam, verbose=True, print_every=1,
                 **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above

        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

        - loss_func: Loss function object.
        - learning_rate: Float, learning rate used for gradient descent.

        - optimizer: The optimizer specifying the update rule

        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model = model
        self.loss_func=Loss(**hparams)
        optimizer=getattr(torch.optim, hparams["optimizer"].capitalize())

        optim_params = optimizer.__init__.__code__.co_varnames[:optimizer.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in hparams.items() if key in optim_params}

        self.opt = optimizer(model.parameters(), **parsed_arguments)
        self.verbose = verbose
        self.print_every = print_every
        self.hparams=hparams

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.current_patience = 0
        i2dl_exercises_path=os.path.dirname(os.path.abspath(os.getcwd()))
        temp_dir=os.path.join(i2dl_exercises_path,"temp")
        os.makedirs(temp_dir, exist_ok=True)
        self.weight_paths=os.path.join(temp_dir, "best_model_weights.pth")

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_batch_loss = []
        self.val_batch_loss = []

        self.num_operation = 0
        self.current_patience = 0

    def _step(self, X, y, validation=False):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.

        :param X: batch of training features
        :param y: batch of corresponding training labels
        :param validation: Boolean indicating whether this is a training or
            validation step

        :return loss: Loss between the model prediction for X and the target
            labels y
        """
        loss = None

        # Forward pass
        y_pred = self.model(X)
        # Compute loss
        loss = self.loss_func(y_pred, y, self.model)
        # Loss without reg term
        loss_without_reg=self.loss_func.compute_loss_without_regularization(y_pred, y).item()

        # Perform gradient update (only in train mode)
        if not validation:
            self.opt.zero_grad()
            # Compute gradients
            loss.backward()
            # Update weights
            self.opt.step()
            # If it was a training step, we need to count operations for
            # backpropagation as well

        return loss_without_reg

    def train(self, epochs=100, patience = None, patience_delta=0, **kwargs):
        """
        Run optimization to train the model.
        """
        # Start an epoch
        for t in range(epochs):
            
            # Iterate over all training samples
            train_epoch_loss = 0.0
            self.model.train()
            for X,y in self.train_dataloader:

                X=X.view(X.shape[0],-1).to(self.hparams["device"])
                y=y.view(y.shape[0],-1).to(self.hparams["device"])
                # Update the model parameters.
                validate = False
                train_loss = self._step(X, y, validation=validate)

                self.train_batch_loss.append(train_loss)
                train_epoch_loss += train_loss

            train_epoch_loss /= len(self.train_dataloader)

                        
            
            # Iterate over all validation samples
            val_epoch_loss = 0.0

            self.model.eval()
            for X,y in self.val_dataloader:
                X=X.view(X.shape[0],-1).to(self.hparams["device"])
                y=y.view(y.shape[0],-1).to(self.hparams["device"])
                # Compute Loss - no param update at validation time!
                with torch.no_grad():
                    val_loss = self._step(X, y, validation=True)
                self.val_batch_loss.append(val_loss)
                val_epoch_loss += val_loss

            val_epoch_loss /= len(self.val_dataloader)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.verbose and t % self.print_every == 0:
                print('(Epoch %d / %d) train loss: %f; val loss: %f' % (
                    t + 1, epochs, train_epoch_loss, val_epoch_loss))

            # Keep track of the best model
            self.update_best_loss(val_epoch_loss, train_epoch_loss, patience_delta)
            if patience and self.current_patience >= patience:
                print("Stopping early at epoch {}!".format(t))
                break



        # At the end of training swap the best params into the model
        #self.model.load_state_dict(self.best_params)
        self.model.load_state_dict(torch.load(self.weight_paths))
        
        print("Best val loss of current config:{}".format(val_epoch_loss))

    def get_dataset_accuracy(self, loader):
        correct = 0
        total = 0
        for X,y in loader:
            X=X.view(X.shape[0],-1).to(self.hparams["device"])
            y=y.to(self.hparams["device"])

            y_pred = self.model.forward(X)
            label_pred = torch.argmax(y_pred, axis=1)
            correct += torch.sum(label_pred == y).item()
            if y.shape:
                total += y.shape[0]
            else:
                total += 1
        return correct / total

    def update_best_loss(self, val_loss, train_loss, patience_delta):
        # Update the model and best loss if we see improvements.
        if not self.best_model_stats or val_loss < self.best_model_stats["val_loss"]:
            self.best_model_stats = {"val_loss":val_loss, "train_loss":train_loss}
            torch.save(self.model.state_dict(), self.weight_paths)
        if val_loss < self.best_model_stats["val_loss"] + patience_delta:
            self.current_patience = 0
        else:
            self.current_patience += 1


def validation_loss(model, loss_func, val_loader):
    model.eval()
    with torch.no_grad():
        val_epoch_loss = 0.0
        for X,y in val_loader:
            X=X.view(X.shape[0],-1).to("mps")
            y=y.to("mps")
            # Compute Loss - no param update at validation time!
            # Forward pass
            y_pred = model(X)
            # Loss without reg term
            loss_without_reg=loss_func.compute_loss_without_regularization(y_pred, y).item()
            val_epoch_loss += loss_without_reg

    val_epoch_loss /= len(val_loader)
    return val_epoch_loss