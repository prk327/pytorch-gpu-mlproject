import copy
import torch
import numpy as np
import scipy.stats

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
        # lets get the length of rows and column
        self._observations = self._X_train.size()[0]
        self._parameters = self._X_train.size()[1]
        self._y_train = self.df_to_tensor(y)
        return (self._X_train, self._y_train)
    
    # now define the lease square extimators using normal equation
    def LSE_Model(self):
        ''' As per the normal quuation: 𝜷̂ =[𝑋𝑇.𝑋]−1.𝑋𝑇.𝑦 '''
        X = copy.deepcopy(self._X_train)
        Xt = self._X_train.transpose_(0,1)
        self._X_train = copy.deepcopy(X)
        y = self._y_train
        Xt_X = torch.matmul(Xt, X)
        self._Xt_X_inv = torch.linalg.inv(Xt_X)
        XT_y = torch.matmul(Xt, y)
        self._lse = torch.matmul(self._Xt_X_inv, XT_y)
        return self._lse
    
    # now we needto calculate our model prediction yhat
    def LSE_Prediction(self):
        # yhat = X .𝜷_hat
        self._prediction = torch.matmul(self._X_train, self._lse)
        return self._prediction
    
    def Compute_P_Value(self, feature_vec):
        # find p-value for two-tailed test
        return scipy.stats.t.sf(np.absolute(feature_vec), (self._observations - self._parameters))*2
    
    # now we will calculate the Hypothesis Test for our model
    def Generate_Model_Stats(self, alpha = 0.05):
        Residual = self._y_train - self._prediction
        Residual_square = torch.square(Residual)
        # SSE = sum of (y - yhat)^2
        self._SS_Residual = Residual_square.sum()
        self._Standard_Error = torch.sqrt(self._SS_Residual / (self._observations - self._parameters))
        self._SE_Power2 = torch.pow(self._Standard_Error,2)
        self._c_matrix_diag = torch.diag(self._Xt_X_inv, 0)
        self._SE_Of_Regression_Coefficient = torch.sqrt(torch.mul(self._SE_Power2, self._c_matrix_diag))
        y_mean = torch.mean(self._y_train)
        S2 = torch.square(self._y_train - y_mean)
        self._SST = S2.sum()
        self._SSR = self._SST - self._SS_Residual 
        self._tStats = self._lse.flatten() / self._SE_Of_Regression_Coefficient
        # we first need to convert tstats to numpy
        tstats = self._tStats.detach().cpu().numpy()
        v_P_Value = np.vectorize(self.Compute_P_Value)
        self._pValue = v_P_Value(tstats)
        filterArray = self._pValue < alpha
        InsigfilterArray = self._pValue >= alpha
        featureVector = np.array(self._feature)
        self._Significant_Regression_Variable = featureVector[filterArray]
        self._InSignificant_Regression_Variable = featureVector[InsigfilterArray]
        self._t_inv_2 = scipy.stats.t.ppf(1-(alpha/2), (self._observations - self._parameters))
        self._Lower_Confidence_Interval = self._lse.flatten() - torch.mul(self._t_inv_2, self._SE_Of_Regression_Coefficient)
        self._Upper_Confidence_Interval = self._lse.flatten() + torch.mul(self._t_inv_2, self._SE_Of_Regression_Coefficient)
        return None
    
    def Mean_Response_CI(self, x_knot):
        input_X = [1] + x_knot
        X_Knot = torch.tensor([input_X]).float().to(self._device)
        # Average at Y 𝜇 ̂_(𝑌|𝑥_0 )
        self._Mean_Response_Y = torch.matmul(X_Knot,self._lse)
        # 𝒙_𝟎(𝑿^′ 𝑿)^(−1)𝒙_𝟎'
        X0Inv_XTX_Inv = torch.matmul(torch.matmul(X_Knot,self._Xt_X_inv), X_Knot.transpose_(0,1))
        # √(𝜎 ̂^2 𝒙_𝟎(𝑿^′ 𝑿)^(−1) 𝒙_𝟎' )
        SE2_X0Inv_XTX_Inv = torch.sqrt(torch.mul(self._SE_Power2, X0Inv_XTX_Inv))
        # 𝜇 ̂_(𝑌|𝑥_0 )−𝑡_(𝛼∕2,𝑛−𝑝)
        twoTailTset = torch.mul(self._t_inv_2, SE2_X0Inv_XTX_Inv)
        self._Lower_Mean_Confidence = self._Mean_Response_Y - twoTailTset
        self._Upper_Mean_Confidence = self._Mean_Response_Y + twoTailTset
        return
        
    def Prediction_On_Future_OBS(self, x_knot):
        input_X = [1] + x_knot
        X_Knot = torch.tensor([input_X]).float().to(self._device)
        # Average at 𝑦 ̂_0=𝒙_𝟎′𝜷 ̂
        self._Mean_Response_On_Future_Y = torch.matmul(X_Knot,self._lse)
        # 1+𝒙_𝟎(𝑿^′ 𝑿)^(−1)𝒙_𝟎'
        OnePlusX0Inv_XTX_Inv = 1+torch.matmul(torch.matmul(X_Knot,self._Xt_X_inv), X_Knot.transpose_(0,1))
        # √(𝜎 ̂^2 (1+𝒙_𝟎(𝑿^′ 𝑿)^(−1) 𝒙_𝟎' ))
        SE2_OnePlusX0Inv_XTX_Inv = torch.sqrt(torch.mul(self._SE_Power2, OnePlusX0Inv_XTX_Inv))
        # 𝜇 ̂_(𝑌|𝑥_0 )−𝑡_(𝛼∕2,𝑛−𝑝)
        twoTailTsetPrediction = torch.mul(self._t_inv_2, SE2_OnePlusX0Inv_XTX_Inv)
        self._Lower_Mean_Confidence_Future = self._Mean_Response_On_Future_Y - twoTailTsetPrediction
        self._Upper_Mean_Confidence_Future = self._Mean_Response_On_Future_Y + twoTailTsetPrediction
        return
        
        
        
    