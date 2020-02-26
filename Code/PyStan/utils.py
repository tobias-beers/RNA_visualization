import pickle
import pystan

def loadStan(file):
    try:
        model = pickle.load(open('pickled_models/'+file+'.pkl', 'rb'))
        print('Model loaded succesfully from cache.')
    except:
        model = pystan.StanModel(file = 'StanModels/'+file+'.stan')
        print('Model compiled succesfully.')
        with open('pickled_models/'+file+'.pkl', 'wb') as f:
            pickle.dump(model, f)
        print('Model saved succesfully to cache.')
    
    return model