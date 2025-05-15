import keras
import tensorflow as tf


TRAIN_SIZE = 60000
TEST_SIZE = 10000
BATCH_SIZE = 64 

(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()

def preprocess_images(images):
    images = images.reshape(images.shape[0], 28, 28, 1)/2550.0 #normalizado 
    return images.astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

#aca abajo defino la red VAE

class CVAE(keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(28, 28, 1)), #variar despues
                keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
                keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),

                keras.layers.Flatten(),

                keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(latent_dim,)),
                keras.layers.Dense(units= 7*7*32, activation = tf.nn.relu),
                keras.layers.Reshape(target_shape=(7, 7, 32)),

                keras.layers.Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding= 'same', activation='relu'),
                keras.layers.Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, padding= 'same', activation='relu'),

                keras.layers.Conv2DTranspose(filters=1, kernel_size=3, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(BATCH_SIZE, self.latent_dim))

        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape= mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits