import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


# Sentence-TransformersのMultiple Negatives Ranking Lossに手を加えただけ
class WeightedMultipleNegativesRankingLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim):
        super(WeightedMultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # 元の実装で何もしていなかったlabelsに重みを与えて、それを扱うように変更
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        weight = torch.cat((torch.ones(labels.shape).to(scores.device), labels.to(scores.device)), 0)
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores * weight, labels)

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}





