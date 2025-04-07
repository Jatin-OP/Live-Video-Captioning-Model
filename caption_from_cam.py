import cv2
import time
import numpy as np
from PIL import Image
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import load_model

# === Load Tokenizer ===
tokenizer = load(open("tokenizer.p", "rb"))
vocab_size = len(tokenizer.word_index) + 1
max_length = 32

# === Define Captioning Model ===
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# === Load Model ===
model = define_model(vocab_size, max_length)
model.load_weights('D:/Image Captioning/models2/model_7.h5')

# === Load Feature Extractor ===
xception_model = Xception(include_top=False, pooling="avg")

# === Feature Extraction ===
def extract_features_from_frame(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((299, 299))
    image = np.array(image) / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)
    return model.predict(image)

# === Generate Caption ===
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None or word == 'end':
            break
        in_text += ' ' + word
    return in_text.replace("start", "").strip()

# === Webcam Stream ===
cap = cv2.VideoCapture(0)
last_capture_time = time.time()
caption = ""

print("📸 Press 'q' to quit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_capture_time >= 5:
        features = extract_features_from_frame(frame, xception_model)
        caption = generate_desc(model, tokenizer, features, max_length)
        last_capture_time = current_time
        print("Caption:", caption)

    # Display the caption
    display_frame = frame.copy()
    cv2.putText(display_frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Live Captioning', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
