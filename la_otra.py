def botito():   
    import random
    import json
    import pickle
    import numpy as np 

    #Importar librerias necesarias para lematizar
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt')

    #Importar librerias necesarias para la red neuronal
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation, Dropout
    from tensorflow.keras.optimizers import SGD

    lemmatizer = WordNetLemmatizer()
    sentimientos = json.loads(open('sentimientos.json').read())

    words = []
    classes = []
    documents = []
    ignore_letters = ['?','!','.',',']

    for sentimiento in sentimientos ['sentimientos']:
        for pattern in sentimiento['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list,sentimiento['tag']))
            if  sentimiento['tag'] not in classes:
                classes.append(sentimiento['tag'])

    #Eliminar duplicados y lematizar
    nltk.download('wordnet')
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

    words = sorted(set(words))
    classes = sorted(set(classes))

    pickle.dump(words, open('words.pk1','wb'))
    pickle.dump(classes, open('classes.pk1','wb'))

    #Creamos una bolsa de palabras
    training = []
    output_empty = [0]*len(classes)
    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    #Entrenando el modelo
    random.shuffle(training)
    training = np. array(training)

    #Definiendo valores de enternamiento
    train_x = list(training[:,0])
    train_y = list(training[:,1])

    #Creando una red neural

    model = Sequential()
    model.add(Dense(128,input_shape=(len(train_x[0]),),activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(68, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]),activation = 'softmax'))

    sgd = SGD(lr=0.01, decay= 1e-6,momentum = 0.9, nesterov = True)

    #Defininendo la compilacion y la parte que se entrenará
    model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(np.array(train_x),np.array(train_y),epochs=200, batch_size =5)
    model.save('chatbot_model.model')
    print('hecho')

    #Importar librerias para el modelo de chatbot
    import random
    import json
    import pickle
    import numpy as np
    import nltk
    from nltk.stem import WordNetLemmatizer
    from tensorflow.keras.models import load_model

    lemmatizer = WordNetLemmatizer()
    sentimientos = json.loads(open('sentimientos.json').read())

    #Cargando archivos de forma binaria
    words = pickle.load(open('words.pk1','rb'))
    classes = pickle.load(open('classes.pk1','rb'))

    #Cargamos el modelo
    model = load_model('chatbot_model.model')

    #Funcion para limpiar oraciones
    def clean_up_sentences(sentece):
        sentece_words = nltk.word_tokenize(sentece)
        sentece_words = [lemmatizer.lemmatize(word) for word in sentece_words]
        return sentece_words

    def bag_of_words(sentece):
        sentece_words = clean_up_sentences(sentece)
        bag = [0]* len(words)
        for w in sentece_words:
            for i, word in enumerate (words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    #Funcion para predecir respuesta
    def predict_class(sentece):
        bow = bag_of_words(sentece)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        result = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD ]

        result.sort(key =lambda x: x[1],reverse= True)
        return_list=[]
        for r in result:
            return_list.append({'sentimiento':classes[r[0]],'probability':str(r[1])})
        return return_list

    #Funcion de respuesta e interfaz del bot
    import pandas as pd
    def get_response(sentimiento_list,sentimiento_json):
        tag = sentimiento_list[0]['sentimiento']
        list_of_sentimientos = sentimiento_json['sentimientos']
        for i in  list_of_sentimientos:
            if i['tag'] == tag:
                result =random.choice(i['responses'])
                break
        return result 
    def chalala(sentimiento_list,sentimiento_json):
        tag= sentimiento_list[0]['sentimiento']
        list_of_sentimientos = sentimiento_json['sentimientos']
        for i in  list_of_sentimientos:
            if i['tag'] == tag:
                result =random.choice(i['responses'])
                break
        return tag
    print("Hola soy BOTITO")


    message = "hola"
    
    ints = predict_class(message)
    res = chalala(ints,sentimientos)
    print(res)
    
 
   
    return res
