import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # DONE: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.


        # ====== YOUR CODE: ======

        self.weights = torch.randn(n_features, n_classes) * weight_std
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # DONE: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.


        # ====== YOUR CODE: ======
        class_scores = x @ self.weights
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # DONE: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.


        # ====== YOUR CODE: ======
        correct_num = int(torch.eq(y, y_pred).sum())
        acc = correct_num / y.shape[0]
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):

            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            # ====== YOUR CODE: ======
            # Train stage
            average_loss = 0
            average_acc = 0

            for batch, data in enumerate(dl_train):
                x, y = data
                y_pred, x_scores = self.predict(x)
                term1 = loss_fn(x, y, x_scores, y_pred)
                term2 = (weight_decay/2) * torch.sum(torch.mul(self.weights, self.weights))
                average_loss += (term1 + term2) * y.shape[0]
                average_acc = self.evaluate_accuracy(y, y_pred) * y.shape[0]
                grad_w = loss_fn.grad() + weight_decay * self.weights
                self.weights = self.weights - learn_rate * grad_w
            average_loss /= len(dl_train.dataset)
            average_acc /= len(dl_train.dataset)
            train_res.loss.append(average_loss)
            train_res.accuracy.append(average_acc)

            # Validation stage
            average_loss = 0
            average_acc = 0

            for batch, data in enumerate(dl_valid):
                x, y = data
                y_pred, x_scores = self.predict(x)
                term1 = loss_fn(x, y, x_scores, y_pred)
                term2 = (weight_decay/2) * torch.sum(torch.mul(self.weights, self.weights))
                average_loss += (term1 + term2) * y.shape[0]
                average_acc = self.evaluate_accuracy(y, y_pred) * y.shape[0]
            average_loss /= len(dl_valid.dataset)
            average_acc /= len(dl_valid.dataset)
            valid_res.loss.append(average_loss)
            valid_res.accuracy.append(average_acc)

            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        weights = self.weights.t()
        if has_bias:
            weights = weights.narrow(1, 0, weights.shape[1]-1)
        C, H, W = img_shape
        w_images = weights.view(-1, C, H, W)
        # ========================

        return w_images
