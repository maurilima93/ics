import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# Carregando o modelo VGG16 pré-treinado
vgg = VGG16(weights='imagenet')
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

def extract_features(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# Função para carregar legendas e processar texto
def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def load_descriptions(doc):
    mapping = {}
    for line in doc.split('\n'):
        tokens = line.split()
        if len(tokens) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(image_desc)
    return mapping

def to_vocabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = []
    for key in descriptions.keys():
        [lines.append(d) for d in descriptions[key]]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = []
    for key in descriptions.keys():
        [lines.append(d) for d in descriptions[key]]
    return max(len(d.split()) for d in lines)

# Função para criar sequências de entrada para o modelo
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Modelo de legenda de imagem
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Carregando e prepararando dados
filename = 'descriptions.txt'
doc = load_doc(filename)
descriptions = load_descriptions(doc)
vocabulary = to_vocabulary(descriptions)
tokenizer = create_tokenizer(descriptions)
vocab_size = len(tokenizer.word_index) + 1
max_len = max_length(descriptions)

# Extraindo features das imagens
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
features = {}
for img_path in image_paths:
    image_id = img_path.split('/')[-1].split('.')[0]
    features[image_id] = extract_features(img_path, vgg)

# Criando sequências de treinamento
X1train, X2train, ytrain = create_sequences(tokenizer, max_len, descriptions, features, vocab_size)

# Definindo o modelo
model = define_model(vocab_size, max_len)

# Treinando o modelo
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2)

# Salvando o modelo
model.save('image_captioning_model.h5')
