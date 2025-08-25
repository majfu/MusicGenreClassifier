class StandardizationTransform:
    def __init__(self, mean_tensor, std_tensor):
        self.mean = mean_tensor
        self.std = std_tensor

    def __call__(self, feature_tensor):
        return (feature_tensor - self.mean) / self.std
