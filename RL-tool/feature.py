
class FeatureExtractor(object):
    def transform(self, x):
        pass


class IdentityFeatureExtractor(FeatureExtractor):
    def transform(self, x):
        return x
