import mxnet
import pdb

class resnet101_100cls(mxnet.gluon.HybridBlock):
    def __init__(self, prob_drop = 0.3, **kwargs):
        super(resnet101_100cls, self).__init__(**kwargs)
        resnet101_v1 = mxnet.gluon.model_zoo.vision.resnet101_v1(\
                        pretrained = True)
        self.features = resnet101_v1.features
        #self.features.add(mxnet.gluon.nn.Dropout(rate = prob_drop))
        self.output = mxnet.gluon.nn.HybridSequential()
        self.output.add(mxnet.gluon.nn.Dense(100))
        self.output.add(mxnet.gluon.nn.Dense(2))

    def hybrid_forward(self, F, image):
        features = self.features(image)
        output = self.output(features)
        output = output.softmax()
        return output

if __name__ == '__main__':
    testnet = resnet101_100cls()
    image = mxnet.ndarray.random.uniform(shape = (1, 3, 256, 256))

    testnet.initialize()
    cls_output = testnet(image)
    print(cls_output)
