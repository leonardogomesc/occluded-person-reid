import torch
import numpy as np

'''
def calculate_distance_matrix(occlusion_labels, local_feat_list, global_feat):
    occlusion_labels = occlusion_labels.detach()

    occlusion_labels_matrix = torch.zeros((occlusion_labels.size(0), occlusion_labels.size(0)), device=occlusion_labels.device)
    distance_matrix = torch.zeros((global_feat.size(0), global_feat.size(0)), device=global_feat.device)

    for stripe in range(len(local_feat_list)):
        stripe_occlusion_labels = occlusion_labels[:, stripe]
        stripe_occlusion_labels_matrix = stripe_occlusion_labels.unsqueeze(1) * stripe_occlusion_labels.unsqueeze(0)
        stripe_distance_matrix = torch.cdist(local_feat_list[stripe], local_feat_list[stripe], p=2)

        occlusion_labels_matrix += stripe_occlusion_labels_matrix
        distance_matrix += stripe_occlusion_labels_matrix * stripe_distance_matrix

    distance_matrix += torch.cdist(global_feat, global_feat, p=2)
    distance_matrix /= occlusion_labels_matrix + 1

    return distance_matrix
'''


def calculate_distance_matrix(occlusion_labels, occlusion_labels_2, local_feat_list, local_feat_list_2, global_feat, global_feat_2):
    occlusion_labels = occlusion_labels.detach()
    occlusion_labels_2 = occlusion_labels_2.detach()

    occlusion_labels_matrix = torch.zeros((occlusion_labels.size(0), occlusion_labels_2.size(0)), device=occlusion_labels.device)
    distance_matrix = torch.zeros((global_feat.size(0), global_feat_2.size(0)), device=global_feat.device)

    for stripe in range(len(local_feat_list)):
        stripe_occlusion_labels = occlusion_labels[:, stripe]
        stripe_occlusion_labels_2 = occlusion_labels_2[:, stripe]

        stripe_occlusion_labels_matrix = stripe_occlusion_labels.unsqueeze(1) * stripe_occlusion_labels_2.unsqueeze(0)
        stripe_distance_matrix = torch.cdist(local_feat_list[stripe], local_feat_list_2[stripe], p=2)

        occlusion_labels_matrix += stripe_occlusion_labels_matrix
        distance_matrix += stripe_occlusion_labels_matrix * stripe_distance_matrix

    distance_matrix += torch.cdist(global_feat, global_feat_2, p=2)
    distance_matrix /= occlusion_labels_matrix + 1

    return distance_matrix


def positive_mask(labels):
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_equal

    return mask


def negative_mask(labels):
    label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

    return label_not_equal


def batch_hard_mine_triplet(distance_matrix, labels, margin=1.0):

    with torch.no_grad():
        pm = positive_mask(labels)
        nm = negative_mask(labels)

        hardest_positive_dist = distance_matrix + torch.where(pm, 0.0, -np.inf)
        hardest_positive_dist = torch.argmax(hardest_positive_dist, dim=1)

        hardest_negative_dist = distance_matrix + torch.where(nm, 0.0, np.inf)
        hardest_negative_dist = torch.argmin(hardest_negative_dist, dim=1)

    positive = torch.gather(distance_matrix, 1, hardest_positive_dist.unsqueeze(1))
    negative = torch.gather(distance_matrix, 1, hardest_negative_dist.unsqueeze(1))

    triplet = torch.mean(torch.clamp(positive - negative + margin, min=0))

    return triplet


def main():
    f = torch.tensor([[0,0.1],[0.1,0.2],[0.2,0.3],[1,1.1],[1.1,1.2],[1.2,1.3],[2,2.1],[2.1,2.2]])
    l = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

    print(batch_hard_mine_triplet(torch.cdist(f, f, p=2), l))


if __name__ == '__main__':
    main()

