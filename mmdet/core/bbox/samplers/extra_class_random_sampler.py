import numpy as np
import torch

from .base_sampler import BaseSampler


class ExtraClassRandomSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 fake_pos_fraction=0.5,
                 with_extra_class=False,
                 **kwargs):
        super(ExtraClassRandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals, with_extra_class)
        self.fake_pos_fraction = fake_pos_fraction

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]

    def random_pos_choice(self, gallery, num, pos_labels):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        expected_truly_pos_num = int(num * self.fake_pos_fraction)
        #NB: fake_pos_fraction fake is not fake, just too lazy to change the name

        truly_pos_inds = torch.nonzero(pos_labels.squeeze()%2==0)
        fake_pos_inds = torch.nonzero(pos_labels.squeeze()%2)
        if truly_pos_inds.size(0) < expected_truly_pos_num:
            rand_truly_pos_inds = truly_pos_inds
            rand_fake_pos_inds = \
                    self.random_choice(fake_pos_inds, num-truly_pos_inds.size(0))
        else:
            rand_truly_pos_inds = \
                    self.random_choice(truly_pos_inds, expected_truly_pos_num)
            rand_fake_pos_inds = \
                    self.random_choice(fake_pos_inds, num-expected_truly_pos_num)          
        rand_inds = torch.cat((rand_truly_pos_inds, rand_fake_pos_inds), 0)

        return gallery[rand_inds]

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples. Cause we have 
        fake positive samples, we need to control the truly postive fraction.
        """
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        pos_labels = assign_result.labels[pos_inds]
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            # TODO:if extra class may need adjust@laycoding
            return pos_inds
        else:
            return self.random_pos_choice(pos_inds, num_expected, pos_labels)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
