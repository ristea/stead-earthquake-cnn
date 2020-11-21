import numpy as np


class StatsManager:

    def __init__(self, config):
        self.config = config

    def _normalize(self, vector):
       return vector * 10

    def get_stats(self, pred_depth, pred_distance, pred_magnitude,
                  lbl_depth, lbl_distance, lbl_magnitude):
        pred_depth = np.concatenate(pred_depth)
        pred_distance = np.concatenate(pred_distance)
        pred_magnitude = np.concatenate(pred_magnitude)

        lbl_depth = np.concatenate(lbl_depth)
        lbl_distance = np.concatenate(lbl_distance)
        lbl_magnitude = np.concatenate(lbl_magnitude)

        if self.config['normalize_labels']:
            pred_depth = self._normalize(pred_depth)
            pred_distance = self._normalize(pred_distance)
            pred_magnitude = self._normalize(pred_magnitude)

            lbl_depth = self._normalize(lbl_depth)
            lbl_distance = self._normalize(lbl_distance)
            lbl_magnitude = self._normalize(lbl_magnitude)

        depth_error = self.get_stats_depth(pred_depth, lbl_depth)
        distance_error = self.get_stats_distance(pred_distance, lbl_distance)
        magnitude_error = self.get_stats_magnitude(pred_magnitude, lbl_magnitude)

        return depth_error, distance_error, magnitude_error

    def get_stats_magnitude(self, predictions, labels):
        return np.mean(np.abs(predictions - labels))

    def get_stats_depth(self, predictions, labels):
        return np.mean(np.abs(predictions - labels))

    def get_stats_distance(self, predictions, labels):
        return np.mean(np.abs(predictions - labels))
