import torch
import torch.nn as nn
import numpy as np


class TransformerBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams=hparams
        n_hidden=hparams["n_hidden"]
        if hparams["batch_norm"]:
            self.norm=nn.BatchNorm1d(n_hidden)
        elif hparams["layer_norm"]:
            self.norm=nn.LayerNorm(n_hidden)
        else:
            self.norm=nn.Identity()
        self.ff=nn.Sequential(self.norm,nn.Linear(n_hidden,4 *n_hidden),nn.GELU(),nn.Linear(4*n_hidden,n_hidden))

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.ff(x)
        if self.hparams["shortcut"]:
            x = x + shortcut  # Add the original input back
        return x
class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            # 
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################


        if  hparams["batch_norm"] or hparams["layer_norm"]:
            assert hparams["batch_norm"] != hparams["layer_norm"], "For mlp normalization either batch_norm or layer_norm must be true"

        if hparams["num_layers"]==0:
            if hparams["batch_norm"]:
                self.norm=nn.BatchNorm1d(hparams["input_size"])
            elif hparams["layer_norm"]:
                self.norm=nn.LayerNorm(hparams["input_size"])
            else:
                self.norm=nn.Identity()
            self.encoder=nn.Sequential(self.norm,nn.Linear(hparams["input_size"],hparams["latent_dim"]),nn.GELU())
            return 
        
        self.up=nn.Linear(hparams["input_size"],hparams["n_hidden"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(hparams) for _ in range(hparams["num_layers"])])
        if hparams["batch_norm"]:
            self.norm_out=nn.BatchNorm1d(hparams["n_hidden"])
        elif hparams["layer_norm"]:
            self.norm_out=nn.LayerNorm(hparams["n_hidden"])
        else:
            self.norm_out=nn.Identity()

        self.down=nn.Sequential(self.norm_out,nn.Linear(hparams["n_hidden"],hparams["latent_dim"]))
        self.encoder=nn.Sequential(self.up,self.trf_blocks,self.down)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        x=self.encoder(x)
        return x
    

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################


        if  hparams["batch_norm"] or hparams["layer_norm"]:
            assert hparams["batch_norm"] != hparams["layer_norm"], "For mlp normalization either batch_norm or layer_norm must be true"

        if hparams["num_layers"]==0:
            if hparams["batch_norm"]:
                self.norm=nn.BatchNorm1d(hparams["latent_dim"])
            elif hparams["layer_norm"]:
                self.norm=nn.LayerNorm(hparams["latent_dim"])
            else:
                self.norm=nn.Identity()
            self.decoder=nn.Sequential(self.norm,nn.Linear(hparams["latent_dim"],hparams["latent_dim"]))
            return
        
        self.up=nn.Linear(hparams["latent_dim"],hparams["n_hidden"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(hparams) for _ in range(hparams["num_layers"])])
        if hparams["batch_norm"]:
            self.norm_out=nn.BatchNorm1d(hparams["n_hidden"])
        elif hparams["layer_norm"]:
            self.norm_out=nn.LayerNorm(hparams["n_hidden"])
        else:
            self.norm_out=nn.Identity()

        self.down=nn.Sequential(self.norm_out,nn.Linear(hparams["n_hidden"],hparams["output_dim"]))
        self.decoder=nn.Sequential(self.up,self.trf_blocks,self.down)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        latent=self.encoder(x)
        reconstruction=self.decoder(latent)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        optimizer=getattr(torch.optim, self.hparams["optimizer"].capitalize())
        optim_params = optimizer.__init__.__code__.co_varnames[:optimizer.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hparams.items() if key in optim_params}
        self.optimizer=optimizer(self.parameters(), **parsed_arguments)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################
        images= batch # Get the images and labels from the batch, in the fashion we defined in the dataset and dataloader.
        images= images.to(self.hparams["device"]) # Send the data to the device (GPU or CPU) - it has to be the same device as the model.
        images = images.view(images.shape[0], -1) 
        self.train()
        self.optimizer.zero_grad()
        pred=self.forward(images)
        loss = loss_func(pred, images, self)
        loss.backward()
        self.optimizer.step()
        loss=loss_func.compute_loss_without_regularization(pred, images)
        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        self.eval()
        with torch.no_grad():
            images= batch # Get the images and labels from the batch, in the fashion we defined in the dataset and dataloader.
            images= images.to(self.hparams["device"]) # Send the data to the device (GPU or CPU) - it has to be the same device as the model.
            images = images.view(images.shape[0], -1) 
            pred=self.forward(images)
            loss = loss_func.compute_loss_without_regularization(pred, images)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################

        self.model = nn.Sequential(nn.Linear(hparams["latent_dim"] ,(hparams["num_classes"])))

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        adam_params = torch.optim.Adam.__init__.__code__.co_varnames[:torch.optim.Adam.__init__.__code__.co_argcount]
        parsed_arguments={ key: value for key,value in self.hparams.items() if key in adam_params}
        self.optimizer=torch.optim.Adam(self.parameters(), **parsed_arguments)


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
