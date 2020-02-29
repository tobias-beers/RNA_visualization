import pickle
import pystan

def loadStan(file, recompile=False, automatic_pickle = True):
    if recompile:
        try:
            model = pystan.StanModel(file = 'StanModels/'+file+'.stan')
            print('Model compiled succesfully.')
            if automatic_pickle:
                with open('pickled_models/'+file+'.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print('Model saved succesfully to cache.')
        except FileNotFoundError:
            print(file+'.stan not found!')
        except ValueError:
            print('Could not compile! Error in code maybe!')
    else:
        try:
            model = pickle.load(open('pickled_models/'+file+'.pkl', 'rb'))
            print('Model loaded succesfully from cache.')
        except:
            try:
                model = pystan.StanModel(file = 'StanModels/'+file+'.stan')
                print('Model compiled succesfully.')
                if automatic_pickle:
                    with open('pickled_models/'+file+'.pkl', 'wb') as f:
                        pickle.dump(model, f)
                    print('Model saved succesfully to cache.')
            except FileNotFoundError:
                print(file+'.stan not found!')
            except ValueError:
                print('Could not compile! Error in code maybe!')

    return model