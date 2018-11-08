import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).


        # ====== YOUR CODE: ======
        y.unsqueeze_(1)
        # print("y", y.size())
        num_class = x_scores.shape[1]
        y_expended = y.expand(-1, num_class)
        # print(y_expended.size())
        s_yi = torch.gather(x_scores, dim=1, index=y_expended)
        print("s_yi[0][0]", s_yi[0][0])
        print("y[0]", y[0])
        print("x_scores[0][y[0]]", x_scores[0][y[0]])
        m = x_scores - s_yi + self.delta
        # print("type m", m.type())
        m_max = torch.max(m, torch.FloatTensor([0]).expand_as(m))
        l_iw = torch.sum(m_max, dim=1) - self.delta
        print("m_max[0][0]", m_max[0][0])
        print("x_scores[0][0] - x_scores[0][y[0]] + self.delta", x_scores[0][0] - x_scores[0][y[0]] + self.delta)

        i, j = 0, 5
        print("m_max[i][j]", m_max[i][j])
        print("x_scores[i][j] - x_scores[i][y[i]] + self.delta", x_scores[i][j] - x_scores[i][y[i]] + self.delta)
        print("diff", m_max[i][j] - (x_scores[i][j] - x_scores[i][y[i]] + self.delta))
        loss = l_iw.sum() / y.shape[0]

        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return grad
