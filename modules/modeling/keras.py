from keras.models import Sequential
from keras.layers import Dense, Dropout


model = Sequential()
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32)