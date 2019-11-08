import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

classifier = load_model('malaria-9688.h5')

test_datagen = ImageDataGenerator(rescale=1./255)


predict_gen = test_datagen.flow_from_directory(directory='img',
                                                              target_size=[64,64],
                                                              batch_size=4123,
                                                              class_mode='categorical')

X_val_sample, res = next(predict_gen)
y_pred = classifier.predict(X_val_sample)

print((np.argmax(y_pred)+1)%2)