import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow_datasets as tfds
from tensorflow.keras.utils import plot_model

# ResNet blokini qurish
def resnet_block(inputs, filters, stride):
    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1:
        inputs = layers.Conv2D(filters, 1, strides=stride, padding='same')(inputs)
        inputs = layers.BatchNormalization()(inputs)

    x = layers.Add()([x, inputs])
    x = layers.Activation('relu')(x)
    return x

# Arxitektura
def resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    x = resnet_block(x, 64, 1)
    x = resnet_block(x, 128, 2)
    x = resnet_block(x, 256, 2)
    x = resnet_block(x, 512, 2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

# model yaratish
input_shape = (32, 32, 3)
num_classes = 10
model = resnet(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# learning rate ni epoch ga to'g'irlash
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train 


input_shape = (32, 32, 3)
num_classes = 12
batch_size = 128
def preprocess_data(example):
    image = example['image']
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 255.0
    label = example['super_class_id']
    return image, label

dataset = tfds.load("stanford_online_products", split = "train")

ds_test = tfds.load("stanford_online_products", split = "test")
# datasetni tozalash


dataset = dataset.map(preprocess_data)
ds_test = ds_test.map(preprocess_data)

dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


epochs = 1

plot_model(model, to_file='resnet.png', show_shapes=True)
history = model.fit(
    dataset,
    epochs=epochs,
    validation_data=ds_test,
)
