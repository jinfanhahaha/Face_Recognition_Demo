from scipy import *
import numpy as np


class SimilarityMeasurement:
    def __init__(self, v_a, v_b):
        self.v_a = v_a
        self.v_b = v_b

    def cos_sim(self):
        v_a = np.mat(self.v_a)
        v_b = np.mat(self.v_b)
        num = float(v_a * v_b.T)
        denom = np.linalg.norm(v_a) * np.linalg.norm(v_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return 1 / sim

    def asymmetricKL(self, P, Q):
        return sum(P * np.log(P / Q))

    def KL(self):
        return (self.asymmetricKL(self.v_a, self.v_b) + self.asymmetricKL(self.v_b, self.v_a)) / 2.00

    def two_D(self):
        return np.sqrt(np.sum(np.square(np.array(self.v_a) - np.array(self.v_b))))

    def one_D(self):
        return np.sum(np.abs(np.array(self.v_a) - np.array(self.v_b)))
