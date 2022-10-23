import copy
import torch
import numpy as np
import scipy.stats

class RegressionAnalysis:
    '''
    The simple linear regression with some basic example and explanation
    of the math behind the code

    Attributes
    ----------
    DataFrame : Pandas DataFrame
        The sample data for which the regression analysis needs to be performed
    Feature : list
        The list of feature column in the dataset represented as X
    Target : list
        The observation column for the datasets represented as Y

    Methods
    -------
    get_device()
        return the torch device based on current enviournment
    df_to_tensor(DataFrame)
        Convert the pandas dataframe into torch tensor on the respective device
    InitInput()
        This will return the feature and target tensor as a tuple
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
        '''
        This will convert the pandas dataframe into torch.tensor and also add the Homogenous cordinate
        which is also represented as y-Intercept and initialize it to 1

        The Instance:
            class._X_train - is the final tensor with all the feature
            class._y_train - is the target tensor
            class._observations - is the number of rows (n)
            class._parameters - is the model parameter p=(k+1) where k is the number of regressor variable and 1 is the bias term

        Return
        ------
        Tuple of class._X_train and class._y_train
        '''
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
        # convert the target dataframe to torch.tensor
        self._y_train = self.df_to_tensor(y)
        return (self._X_train, self._y_train)
    
    # now define the lease square extimators using normal equation
    def LSE_Model(self):
        '''
        This will calculate the least square estimates (𝜷̂) using the matrix multiplication
        The Normal Equation for LSE is: 𝜷̂ =[𝑋𝑇.𝑋]−1.𝑋𝑇.𝑦
        \\begin{equation}
            \hat{𝜷}=\\begin{bmatrix}
                X{^T}.X
            \\end{bmatrix} ^{-1}.X{^T}.y
        \\end{equation}

        The Instance:
            class._Xt_X_inv - Is [𝑋^𝑇.𝑋]^−1 which also repsesented as Cij tensor
            class._lse - Is the Lease square estimates

        Return
        ------
        the Lease square estimates (𝜷̂) with the first beta represent the beta for the y-Intercept
        '''
        X = copy.deepcopy(self._X_train)
        Xt = self._X_train.transpose_(0,1)
        self._X_train = copy.deepcopy(X)
        y = self._y_train
        # X transpose . X
        Xt_X = torch.matmul(Xt, X)
        # Inverse of X transpose . X, which is also called C
        self._Xt_X_inv = torch.linalg.inv(Xt_X)
        # X transpose . y
        XT_y = torch.matmul(Xt, y)
        self._lse = torch.matmul(self._Xt_X_inv, XT_y)
        return self._lse
    
    # now we need to calculate our model prediction yhat
    def LSE_Prediction(self):
        # \\begin{equation}
        #     \hat{y}=X.\hat{𝜷}
        # \\end{equation}
        self._prediction = torch.matmul(self._X_train, self._lse)
        return self._prediction
    
    def Compute_P_Value(self, feature_vec):
        '''
        This will compute the p-value for two tailed test from t-dist with some degree of freedom (n-p)

        Parameters
        ----------
        feature_vec : numpy array
            It is the vector of t-test statistic computed by divining Least Square Estimates by Standard Error of the Regressor Coefficient
        '''
        # find p-value for two-tailed test
        return scipy.stats.t.sf(np.absolute(feature_vec), (self._observations - self._parameters))*2
    
    # now we will calculate the Hypothesis Test for our model
    def Generate_Model_Stats(self, alpha = 0.05):
        # The Sum Of Square Error Formula also known as Residual:
        # \\begin{equation}
        #     SS_{E}= \sum^{n}_{i=1} (y_{i} - \hat{y}_{i})^2 
        # \\end{equation}
        Residual = self._y_train - self._prediction
        Residual_square = torch.square(Residual)
        self._SS_Residual = Residual_square.sum()
        # The Standard error Formula:
        # \\begin{equation}
        #     \sigma^{2} = \\frac{\sum^{n}_{i=1} e_{i}^{2}}{n-p}
        # \\end{equation}
        self._Standard_Error = torch.sqrt(self._SS_Residual / (self._observations - self._parameters))
        # The Standard Error formula for Beta coefficient:
        # \\begin{equation}
        #     se(\hat{𝜷}_{j}) = \sqrt{\hat{\sigma}^{2}C_{jj}}
        # \\end{equation}
        self._SE_Power2 = torch.pow(self._Standard_Error,2)
        self._c_matrix_diag = torch.diag(self._Xt_X_inv, 0)
        self._SE_Of_Regression_Coefficient = torch.sqrt(torch.mul(self._SE_Power2, self._c_matrix_diag))
        # The ANOVA Test | Significance of Regression - Is the regression model significant as a whole
        # The SST formula | Sum of Square for Total Variability
        # \\begin{equation}
        #     SS_{T}= \sum^{n}_{i=1} (y_{i} - \overline{y})^2 
        # \\end{equation}
        y_mean = torch.mean(self._y_train)
        S2 = torch.square(self._y_train - y_mean)
        self._SST = S2.sum()
        # The formula for SSR | Variablity due to Model
        # \\begin{equation}
        #     SS_{R}  = SS_{T} - SS_{E}
        # \\end{equation}
        self._SSR = self._SST - self._SS_Residual
        # Hypothesis Test on Individual Regression Coefficients
        # \\begin{equation}
        # ? \\\\
        #    |t_{0}| > t_{\\alpha / 2, n-p}
        # \\end{equation}
        # First We need to calculate the t statistic using below formula at beta coefficient is equal to 0:
        # \\begin{equation}
        #     T_{0}=\\frac{\hat{𝜷}_{j}}{se(\hat{𝜷}_{j})}
        # \\end{equation}
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
        # Confidence Interval on slope and intercept
        # \\begin{equation}
        #    \hat{𝜷}_{j} - t_{\\alpha / 2, n-p} . se(\hat{𝜷}_{j}) \leq 𝜷_{j} \leq \hat{𝜷}_{j} + t_{\\alpha / 2, n-p} . se(\hat{𝜷}_{j})
        # \\end{equation}
        self._t_inv_2 = scipy.stats.t.ppf(1-(alpha/2), (self._observations - self._parameters))
        self._Lower_Confidence_Interval = self._lse.flatten() - torch.mul(self._t_inv_2, self._SE_Of_Regression_Coefficient)
        self._Upper_Confidence_Interval = self._lse.flatten() + torch.mul(self._t_inv_2, self._SE_Of_Regression_Coefficient)
        return None
    
    def Mean_Response_CI(self, x_knot):
        # Confidence Interval on the Mean Response
        # \\begin{equation}
        #     \hat{𝜇}_{𝑌|𝑥_0}−𝑡_{𝛼∕2,𝑛−𝑝} \sqrt{𝜎 ̂^2 𝒙_𝟎′(𝑿^′ 𝑿)^{−1} 𝒙_𝟎} ≤𝜇_{𝑌|𝑥_0}≤\hat{𝜇}_{𝑌|𝑥_0}+𝑡_{𝛼∕2,𝑛−𝑝} \sqrt{𝜎 ̂^2 𝒙_𝟎′(𝑿^′ 𝑿)^{−1} 𝒙_𝟎}
        # \\end{equation}
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
        # Prediction Interval on a Future Observation (Y0)
        # \\begin{equation}
        #     \hat{𝑦}_{0}±𝑡_{𝛼∕2,𝑛−𝑝} \sqrt{𝜎 ̂^2 (𝟏+𝒙_𝟎′(𝑿^′𝑿)^{−1} 𝒙_{𝟎} ) }
        # \\end{equation}
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
        
        
        
    