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
        y1 = y.unsqueeze(1)
        num_class = x_scores.shape[1]
        y_expended = y1.expand(-1, num_class)
        s_yi = torch.gather(x_scores, dim=1, index=y_expended)
        m = x_scores - s_yi + self.delta
        m_max = torch.max(m, torch.FloatTensor([0]).expand_as(m))
        l_iw = torch.sum(m_max, dim=1) - self.delta
        loss = l_iw.sum() / y1.shape[0]

        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['x'] = x
        self.grad_ctx['m'] = m
        self.grad_ctx['y'] = y1

        # ========================

        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.
        x = self.grad_ctx['x']
        m = self.grad_ctx['m']
        y = self.grad_ctx['y']
        n = x.shape[0]
        c = m.shape[1]
        # ====== YOUR CODE: ======
        yi_eq = torch.zeros(n, c).scatter_(1, y, 1).byte()
        yi_neq = yi_eq ^ 1
        m_gr = m > 0
        sum_m_gr = torch.sum(m_gr, dim=1, keepdim=True).float() - 1.0
        sum_m_gr = sum_m_gr.expand(-1, c)
        g = (m_gr.float() * yi_neq.float()) - (sum_m_gr * yi_eq.float())
        g = g * (1 / n)
        grad = x.t() @ g
        # ========================

        return grad
