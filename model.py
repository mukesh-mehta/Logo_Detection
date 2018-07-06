from keras.layers import Dense, Input, Dropout, Activation, Conv2D, MaxPooling2D, Lambda, Flatten, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import RMSprop, nadam, adam
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50

def pre_trained_vgg16(images):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=images.shape[1:], classes=4)
    x = base_model.get_layer('block5_pool').output
    flat = GlobalAveragePooling2D()(x)
    
    FC_box = Dense(128, activation='relu')(flat)
    FC_box = Dropout(0.6)(FC_box)
    preds = Dense(4, activation='relu')(FC_box)
    model = Model(inputs=base_model.input,outputs=preds)
    model.compile(loss='mean_absolute_error',optimizer=adam(lr=0.0001))
    return model

def pre_trained_mobilenet(images):
    base_model = MobileNet(weights='imagenet', include_top=False, 
                 input_shape=images.shape[1:], classes=4)
    x = base_model.get_layer('conv_pw_13_relu').output
    flat = GlobalAveragePooling2D()(x)
    
    FC_box = Dense(128, activation='relu')(flat)
    FC_box = Dropout(0.6)(FC_box)
    preds = Dense(4, activation='relu')(FC_box)
    model = Model(inputs=base_model.input,outputs=preds)
    model.compile(loss='mean_absolute_error',optimizer=adam(lr=0.0001))
    return model

def pre_trained_resnet50(images):
    base_model = ResNet50(weights='imagenet', include_top=False, 
                 input_shape=images.shape[1:], classes=4)
    x = base_model.get_layer('avg_pool').output
    flat = GlobalAveragePooling2D()(x)
    FC_box = Dropout(0.6)(flat)
    FC_box = Dense(128, activation='relu')(FC_box)
    FC_box = Dropout(0.6)(FC_box)
    preds = Dense(4, activation='relu')(FC_box)
    model = Model(inputs=base_model.input,outputs=preds)
    model.compile(loss='mean_absolute_error',optimizer=adam(lr=0.0001))
    return model

def vgg16_mobilenet(images):
    basemodel1=VGG16(weights='imagenet', include_top=False, input_shape=images.shape[1:], classes=4)
    x1=basemodel1.get_layer('block5_pool').output
    x1=GlobalAveragePooling2D()(x1)
    
    basemodel2=MobileNet(weights=None,input_tensor = basemodel1.input,include_top=False, input_shape=images.shape[1:])
    x2 = basemodel2.output
    x2 = GlobalAveragePooling2D()(x2)
    
    merge = concatenate([x1, x2])
    merge = Dropout(0.6)(merge)
    preds = Dense(4, activation='relu')(merge)
    model = Model(inputs=basemodel1.input,outputs=preds)
    model.compile(loss='mean_absolute_error',optimizer=adam(lr=0.0001))
    return model
