import keras
import sys
from pathlib import Path
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file

from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from keras.applications import ResNet50, InceptionResNetV2
from keras import backend as K


def resnet50_build(input_shape=(224, 224, 3), num_classes=82, model_name="ResNet50"):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")

    prediction = Dense(units=num_classes, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)

    model = Model(inputs=base_model.input, outputs=prediction)

    return model, base_model.output


def efficientnetb3_224_build(input_shape=(224, 224, 3), num_classes=82, weight_file=None):
    def get_model(cfg):
        base_model = getattr(applications, cfg.model.model_name)(
            include_top=False,
            input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
            pooling="avg"
        )
        features = base_model.output
        pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
        pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
        model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
        return model, features

    if not weight_file:
        pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
        weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash="6d7f7b7ced093a8b3ef6399163da6ece", cache_dir=str(Path(__file__).resolve().parent))
    model_name, _ = Path(weight_file).stem.split("_")[:2]
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={input_shape[0]}"])
    model, features = get_model(cfg)
    model.load_weights(weight_file)

    features = model.layers[-3].output
    pred_age = Dense(units=num_classes, activation="softmax", name="pred_age")(features)

    model = Model(inputs=model.input, outputs=pred_age)

    for l in model.layers:
        l.trainable = True
    return model, features


def senet_model_build(input_shape=(224, 224, 3), num_classes=82, weights="imagenet"):
    print("Building senet", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras-squeeze-excite-network')
    from keras_squeeze_excite_network.se_resnet import SEResNet
    m1 = SEResNet(weights=weights, input_shape=input_shape, include_top=True, pooling='avg',weight_decay=0)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, use_bias=True, activation='softmax', name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def mobilenet_224_build(input_shape=(224, 224, 3), num_classes=82, weights="imagenet"):
    print("Building mobilenet v2", input_shape, "- num_classes", num_classes, "- weights", weights)
    m1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape, 1.0, include_top=True, weights=weights)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def vgg16_keras_build(input_shape=(224, 224, 3), num_classes=82, weights="imagenet"):
    # Alpha version
    print("Building vgg16", input_shape, "- num_classes", num_classes, "- weights", weights)
    from keras.applications.vgg16 import VGG16

    # # Uncomment these lines and check the loss
    # input_tensor = keras.layers.Input(shape=input_shape)
    # from keras.applications.vgg16 import preprocess_input
    # input_tensor = keras.layers.Lambda(preprocess_input, arguments={'mode': 'tf'})(input_tensor)
    # m1 = VGG16(include_top=True, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=None)

    m1 = VGG16(include_top=True, weights=weights, input_tensor=None, input_shape=input_shape, pooling=None)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def densenet_121_build(input_shape=(224, 224, 3), num_classes=82, weights="imagenet", lpf_size=1):
    print("Building densenet121bc", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras_vggface')
    from keras_vggface.densenet import DenseNet121
    m1 = DenseNet121(include_top=True, input_shape=input_shape, weights=weights, pooling='avg', lpf_size=lpf_size)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def mobilenet_64_build(input_shape=(64, 64, 3), num_classes=82):
    print("Building mobilenet 64", input_shape, "- num_classes", num_classes)
    from scratch_models.mobile_net_v2_keras import MobileBioNetv2
    m1 = MobileBioNetv2(input_shape=input_shape, width_multiplier=0.5)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def mobilenet_96_build(input_shape=(96,96,3), num_classes=82, weights="imagenet"):
    print("Building mobilenet 96", input_shape, "- num_classes", num_classes, "- weights", weights)
    m1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape, 0.75, include_top=True, weights=weights)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def xception_build(input_shape=(299,299,3), num_classes=82, weights="imagenet", lpf_size=1):
    print("Building xception", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras_vggface')
    from keras_vggface.xception import Xception
    m1 = Xception(include_top=False, input_shape=input_shape, weights=weights, pooling='avg', lpf_size=lpf_size) #emulate include_top through pooling avg
    features = m1.layers[-1].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def squeezenet_build(input_shape=(224, 224, 3), num_classes=82, weights="imagenet"):
    print("Building squeezenet", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras-squeezenet')
    from keras_squeezenet import SqueezeNet
    m1 = SqueezeNet(input_shape=input_shape, weights=weights, include_top=True)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def shufflenet_224_build(input_shape=(224, 224, 3), num_classes=82, weights="imagenet"):
    print("Building shufflenet", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras-shufflenetV2')
    from shufflenetv2 import ShuffleNetV2
    m1 = ShuffleNetV2(input_shape=input_shape, classes=num_classes, include_top=True, scale_factor=1.0, weights=weights)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def vggface_custom_build(input_shape, num_classes=82, weights="vggface2", net="vgg16", lpf_size=1):
    sys.path.append('keras_vggface')
    from keras_vggface.vggface import VGGFace
    return VGGFace(model=net, weights=weights, input_shape=input_shape, classes=num_classes, lpf_size=lpf_size)
    



    



