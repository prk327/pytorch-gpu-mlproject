import copy
import torch

class LeastSEModel:
    '''
    Note: The below library must be imported before importing this class
    import torch
    import copy
    '''
    
    # Instance attribute
    def __init__(self, DataFrame, Feature, Target):
        self._dataframe = copy.deepcopy(DataFrame)
        self._feature = copy.deepcopy(Feature)
        self._target = copy.deepcopy(Target)
        # determine the supported device
        self._device = self.get_device()
        
    # helper function
    # determine the supported device
    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu') # don't have GPU 
        return device

    # convert a df to tensor to be used in pytorch
    def df_to_tensor(self, DataFrame):
        return torch.from_numpy(DataFrame.values).float().to(self._device)
    
    # Initialize Input Parameters
    def InitInput(self):
         # Now lets add the Homogeneous coordinates
        self._dataframe["HCoordinates"] = self._dataframe.apply(lambda x: 1, axis = 1)
        self._feature = ["HCoordinates"] + self._feature
        # lets create X as feature set
        X = self._dataframe[self._feature]
        # lets create y as target set
        y = self._dataframe[self._target]
        # lets convert the dataframe to torch.tensor
        self._X_train = self.df_to_tensor(X)
        self._y_train = self.df_to_tensor(y)
        return (self._X_train, self._y_train)
    
    # now define the lease square extimators using normal equation
    def LSE_Model(self):
        ''' As per the normal quuation: 𝜷̂ =[𝑋𝑇.𝑋]−1.𝑋𝑇.𝑦 '''
        X = copy.deepcopy(self._X_train)
        Xt = self._X_train.transpose_(0,1)
        y = self._y_train
        Xt_X = torch.matmul(Xt, X)
        Xt_X_inv = torch.linalg.inv(Xt_X)
        XT_y = torch.matmul(Xt, y)
        lse = torch.matmul(Xt_X_inv, XT_y)
        return lse