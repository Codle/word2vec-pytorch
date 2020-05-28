import torch.nn as nn
import torch


class CBOW(nn.Module):

    def __init__(self, vocab_size: int, vec_size: int) -> None:
        """Init model

        Arguments:
            vocab_size {int} -- word count number of vocab
            vec_size {int} -- vector size
        """

        super(CBOW, self).__init__()

        self.u_embed = nn.Embedding(vocab_size, vec_size)
        self.v_embed = nn.Embedding(vocab_size, vec_size)

        nn.init.uniform_(self.u_embed.weight, -1, 1)
        nn.init.uniform_(self.v_embed.weight, -1, 1)

        self.loss_func = nn.BCELoss()

    def forward(self,
                context: torch.Tensor,
                pos: torch.Tensor,
                neg: torch.Tensor) -> torch.FloatTensor:
        """Compute loss of word2vec

        Arguments:
            context {torch.Tensor} -- context vectors
            pos {torch.Tensor} -- pos vector
            neg {torch.Tensor} -- neg vector

        Returns:
            torch.FloatTensor -- Loss value
        """
        # shape: batch * vs * 1
        vm = self.v_embed(context).mean(1).unsqueeze(-1)

        # shape: batch * 1 * vs
        u_pos = self.u_embed(pos)
        u_neg = self.u_embed(neg)

        # shape: batch * 1
        pos_uv = torch.bmm(u_pos, vm).squeeze(-1)
        pos_prob = torch.sigmoid(pos_uv)
        neg_uv = torch.bmm(u_neg, vm).squeeze(-1)
        neg_prob = torch.sigmoid(neg_uv)

        # create label tensor
        pos_label = torch.ones_like(pos_prob).to(pos_prob.device)
        neg_label = torch.zeros_like(neg_prob).to(neg_prob.device)

        return self.loss_func(pos_prob, pos_label) + self.loss_func(neg_prob, neg_label)
