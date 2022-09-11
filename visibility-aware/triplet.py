import torch
import numpy as np


def positive_mask(labels):
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_equal

    return mask


def negative_mask(labels):
    label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

    return label_not_equal


def batch_hard_mine(features, labels):

    with torch.no_grad():
        distance_matrix = torch.cdist(features, features, p=2)

        pm = positive_mask(labels)
        nm = negative_mask(labels)

        hardest_positive_dist = distance_matrix + torch.where(~pm, -np.inf, 0.0)
        hardest_positive_dist = torch.argmax(hardest_positive_dist, dim=1)

        hardest_negative_dist = distance_matrix + torch.where(~nm, np.inf, 0.0)
        hardest_negative_dist = torch.argmin(hardest_negative_dist, dim=1)

    positive = torch.index_select(features, 0, hardest_positive_dist)
    negative = torch.index_select(features, 0, hardest_negative_dist)

    return features, positive, negative


def main():
    f = torch.tensor([[0,0.1],[0.1,0.2],[0.2,0.3],[1,1.1],[1.1,1.2],[1.2,1.3],[2,2.1],[2.1,2.2]])
    l = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

    print(batch_hard_mine(f, l))


if __name__ == '__main__':
    main()

