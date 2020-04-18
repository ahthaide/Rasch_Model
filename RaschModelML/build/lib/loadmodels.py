import pickle


def loaddiffi(data):
    diffmodel = "pickle_model.pkl"
    print(" predicted difficulties are:")
    with open(diffmodel, 'rb') as file:
        pickle_model = pickle.load(file)
        newP = pickle_model.predict(data)
        print(newP)


def arraydif(data):
    diffmodel = "pickle_model.pkl"
    with open(diffmodel, 'rb') as file:
        pickle_model = pickle.load(file)
        diff = pickle_model.predict(data)
        return diff


def loadlogit(data):
    logitmodel = "logit_model.pkl"
    print(" predicted logits are:")
    with open(logitmodel, 'rb') as file:
        pickle_model = pickle.load(file)
        newP = pickle_model.predict(data)
        print(newP)


def arraylogit(data):
    logitmodel = "logit_model.pkl"
    with open(logitmodel, 'rb') as file:
        pickle_model = pickle.load(file)
        log = pickle_model.predict(data)
        return log

def loadtheta(data):
        thetamodel = "theta_model.pkl"
        print(" predicted thetas are:")
        with open( thetamodel, 'rb') as file:
            pickle_model = pickle.load(file)
            newP = pickle_model.predict(data)
            print(newP)


def arraytheta(data):
        thetamodel = "theta_model.pkl"
        with open(thetamodel, 'rb') as file:
            pickle_model = pickle.load(file)
            theta = pickle_model.predict(data)
            return theta

