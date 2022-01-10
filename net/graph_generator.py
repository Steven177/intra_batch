import torch
import torch.nn.functional as F


class GraphGenerator():
    def __init__(self, dev, thresh=0, thresh_mode="fc", sim_type='correlation', set_negative='hard'):
        self.device = dev
        self.thresh = thresh
        self.thresh_mode = thresh_mode
        self.sim = sim_type
        self.set_negative  = set_negative

    @staticmethod
    def set_negative_to_zero(W):
        return F.relu(W)

    def set_negative_to_zero_soft(self, W):
        """ It shifts the negative probabilities towards the positive regime """
        n = W.shape[0]
        minimum = torch.min(W)
        W = W - minimum
        W = W * (torch.ones((n, n)).to(self.device) - torch.eye(n).to(self.device))
        return W

    def _get_A(self, W):
        # hn, hp, hn&hp, fc
        if self.thresh_mode == 'hn': # hn = hard negative
            W  = torch.where(W < self.thresh, W, torch.tensor(0).float().to(self.device))
            A = torch.ones_like(W).where(W < self.thresh, torch.tensor(0).float().to(self.device))

        elif self.thresh_mode == 'hp': # hp = hard positive
            W  = torch.where(W > self.thresh, W, torch.tensor(0).float().to(self.device))
            A = torch.ones_like(W).where(W > self.thresh, torch.tensor(0).float().to(self.device))

        elif self.thresh_mode == 'hn_hp': # hn_hp = hard negative and hard positive
            W  = torch.where(W > self.thresh, W, torch.tensor(0).float().to(self.device))
            A = torch.ones_like(W).where(W > self.thresh, torch.tensor(0).float().to(self.device))

        elif self.thresh_mode == 'fc': # fc = fully-connected
            A = torch.ones_like(W)
        else:
            raise ValueError("Check config file: thresh_mode invalid")

        return W, A

    def _get_W(self, x):
        if self.sim == 'correlation':
            x = (x - x.mean(dim=1).unsqueeze(1))
            norms = x.norm(dim=1)
            W = torch.mm(x, x.t()) / torch.ger(norms, norms)
        elif self.sim == 'absolute_correlation':
            x = (x - x.mean(dim=1).unsqueeze(1))
            norms = x.norm(dim=1)
            W = torch.mm(x, x.t()) / torch.ger(norms, norms)
            W = torch.abs(W) # taking absolute correlations
        elif self.sim == 'cosine':
            W = torch.mm(x, x.t())
        elif self.sim == 'learnt':
            n = x.shape[0] # samples
            W = torch.zeros(n, n)
            for i, xi in enumerate(x):
                for j, xj in enumerate(x[(i + 1):], i + 1):
                    W[i, j] = W[j, i] = self.sim(xi, xj) + 1e-8
            W = W.cuda()

        if self.set_negative == 'hard':
            W = self.set_negative_to_zero(W.to(self.device))

        if self.set_negative == 'soft':
            W = self.set_negative_to_zero_soft(W)

        return W

    def get_graph(self, x, Y=None):
        W = self._get_W(x)
        W, A = self._get_A(W)

        A = torch.nonzero(A)
        W = W[A[:, 0], A[:, 1]]

        return W, A, x
