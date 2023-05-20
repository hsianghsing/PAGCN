import torch
from torch import nn
import torch.nn.functional as F
import world
from dataloader import GraphDataset, SocialGraphDataset
from math import sqrt
from utils import cust_mul


class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim).to(self.device)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim).to(self.device)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long().to(self.device)
        users_emb = self.embedding_user(users).to(self.device)
        items_emb = self.embedding_item.weight.to(self.device)
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class LightGCN(nn.Module):
    def __init__(self, configs, datasets):
        super(LightGCN, self).__init__()
        self.config = configs
        self.dataset = datasets
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim']
        self.n_layers = self.config['layers']

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        ).to(self.device)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        ).to(self.device)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.interactionGraph = self.dataset.getSparseGraph()
        self.f = nn.Sigmoid()

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb]).to(self.device)
        embeds = [all_emb]
        G = self.interactionGraph.to(self.device)
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(G, all_emb)
            embeds.append(all_emb)
        embeds = torch.stack(embeds, dim=1)
        embeds = torch.mean(embeds, dim=1)
        _users, _items = torch.split(embeds, [self.num_users, self.num_items])
        return _users, _items

    def getUsersRating(self, _users):
        all_users, all_items = self.computer()
        users_emb = all_users[_users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, _users, pos_items, neg_items):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        all_users, all_items = self.computer()
        users_emb = all_users[_users].to(device)
        pos_emb = all_items[pos_items].to(device)
        neg_emb = all_items[neg_items].to(device)
        users_emb_ego = self.embedding_user(_users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, _users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         users_emb0, pos_emb0, neg_emb0) = self.getEmbedding(_users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (users_emb0.norm(2).pow(2) +
                              pos_emb0.norm(2).pow(2) +
                              neg_emb0.norm(2).pow(2)) / float(len(_users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # return loss, reg_loss
        return loss, reg_loss

    def forward(self, _users, _items):
        all_users, all_items = self.computer()
        users_emb = all_users[_users]
        items_emb = all_items[_items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class PAGCN(nn.Module):
    def __init__(self, configs, datasets):
        super(PAGCN, self).__init__()

        self.config = configs
        self.dataset = datasets

        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.inter_subGraph_list = None
        self.social_subGraph_list = None

        self._init_weight()

    def _init_weight(self):
        self.latent_dim = self.config['latent_dim']
        self.n_layers = self.config['layers']
        self.personalities = self.config['personalities']

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items

        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim).to(self.device)
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim).to(self.device)

        self.fc = torch.nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        self.leaky = torch.nn.LeakyReLU().to(self.device)
        self.dropout = torch.nn.Dropout(p=0.5).to(self.device)

        self.fc2 = torch.nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
        self.fc_personalities = torch.nn.Linear(self.latent_dim, self.personalities).to(self.device)
        self.f = nn.Sigmoid()

        self.interactionGraph = self.dataset.getSparseGraph().to(self.device)
        self.socialGraph = self.dataset.getSocialGraph()

        self.Graph_Comb = ANewFusionModel(self.latent_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.01)
                torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out")
                torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        e0 = torch.cat([users_emb, items_emb]).to(self.device)

        interaction_graph = self.interactionGraph.to(self.device)
        social_graph = self.socialGraph.to(self.device)
        e1 = torch.sparse.mm(interaction_graph, e0)
        e1_u, e1_i = torch.split(e1, [self.num_users, self.num_items])

        e01_u = self.Graph_Comb(e1_u, users_emb)
        tmp = self.dropout(self.leaky(self.fc(e01_u)))
        tmp2 = self.dropout(self.leaky(self.fc2(tmp)))
        group_scores = self.dropout(self.fc_personalities(tmp2))
        a_top, a_top_index = torch.topk(group_scores, k=1, sorted=False)

        if self.inter_subGraph_list is None:
            subgraph_list = []
            for p in range(self.personalities):
                tmp_list = []
                for _ in range(len(a_top_index)):
                    if a_top_index[_] == p:
                        tmp_list.append(_)
                temp = self.dataset.getSubGraph(tmp_list)
                subgraph_list.append(temp)
            self.inter_subGraph_list = subgraph_list

        _init_personality = [e0 for _ in range(self.personalities)]
        u_social = users_emb
        layers_emb = [sum(_init_personality)]

        for layer in range(0, self.n_layers):
            for p in range(self.personalities):
                _init_personality[p] = torch.sparse.mm(self.inter_subGraph_list[p], _init_personality[p])
            UI_from_personality = sum(_init_personality)
            _u, _i = torch.split(UI_from_personality, [self.num_users, self.num_items])
            u_social = torch.sparse.mm(self.socialGraph, u_social)
            _u_comb = self.Graph_Comb(_u, u_social)
            _layer_emb = torch.cat([_u_comb, _i])
            layers_emb.append(_layer_emb)

        _sum = torch.mean(torch.stack(layers_emb, dim=1), dim=1)
        _users, _items = torch.split(_sum, [self.num_users, self.num_items])
        return _users, _items

    def getUsersRating(self, users_):
        all_users, all_items = self.computer()
        users_emb = all_users[users_.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users_, pos_items, neg_items):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        all_users, all_items = self.computer()
        users_emb = all_users[users_].to(device)
        pos_emb = all_items[pos_items].to(device)
        neg_emb = all_items[neg_items].to(device)
        users_emb_ego = self.embedding_user(users_)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, _users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(_users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(_users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users_, items_):
        all_users, all_items = self.computer()
        users_emb = all_users[users_]
        items_emb = all_items[items_]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class ANewFusionModel(nn.Module):
    def __init__(self, embed_dim):
        super(ANewFusionModel, self).__init__()
        # Idea Comes from "Graph Attention Networks for Neural Social Recommendation"
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(embed_dim * 3, embed_dim * 3).to(self.device)
        self.mish = nn.Mish()
        # self.tanh = F.tanh()
        self.fc2 = nn.Linear(embed_dim * 3, embed_dim).to(self.device)
        self.fc3 = nn.Linear(embed_dim, embed_dim).to(self.device)
        # self._init_weight()

    # def _init_weight(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Linear):
    #             torch.nn.init.normal_(module.weight, std=0.01)
    #             torch.nn.init.constant_(module.bias, 0.0)
    #         elif isinstance(module, nn.Conv2d):
    #             torch.nn.init.kaiming_normal_(module.weight, mode="fan_out")
    #             torch.nn.init.constant_(module.bias, 0.0)
    #         elif isinstance(module, nn.BatchNorm2d):
    #             torch.nn.init.constant_(module.weight, 1.0)
    #             torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, y):
        x_y = torch.addcmul(torch.zeros_like(x), x, y)
        _cat = torch.cat((x, y, x_y), dim=1)
        # tmp1 = self.mish(self.fc1(_cat))
        # tmp2 = self.mish(self.fc2(tmp1))
        tmp1 = F.tanh(self.fc1(_cat))
        tmp2 = F.tanh(self.fc2(tmp1))
        tmp3 = self.fc3(tmp2)
        out = tmp3 / tmp3.norm(2)
        return out


if __name__ == "__main__":
    config = {'latent_dim': 64,
              'layers': 3,
              'l2': 1e-3,
              'personalities': 2}
    dataset = SocialGraphDataset('lastfm')
    pagcn = PAGCN(configs=config, datasets=dataset)
    users, items = pagcn.computer()
    # print("users: \n", users)
    # print("items: \n", items)
