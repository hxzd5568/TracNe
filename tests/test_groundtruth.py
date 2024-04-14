import numpy as np


def MSE(y_true, y_pred,):
    if np.isinf(y_true).any() == 1 or np.isinf(y_pred).any() == 1:
        print('y_true, y_pred have inf')
        print(y_true, y_pred)
        return 0
    elif np.isnan(y_true).any() == 1 or np.isnan(y_pred).any() == 1:
        print('y_true, y_pred have nan')
        print(y_true, y_pred)
        return 0
    else:
        relative_error = np.average(np.abs(y_true - y_pred).astype(np.float128)
                                    / (np.abs(y_true).astype(np.float128) + 1e-38))
    relative_error = np.average(np.abs(y_true - y_pred).astype(np.float128)
                                / (np.abs(y_true).astype(np.float128) + 1e-38))
    return relative_error
#______________case 1 ____________
# p01 = np.array([[ 0.877  , -0.6206 ,  0.649  ],
#        [ 0.688  ,  2.254  , -0.2369 ],
#        [ 1.602  , -0.5967 ,  1.07   ],
#        [ 0.05652, -0.6504 , -0.6147 ]],dtype='float16')
# p1= np.array([[ 1.4795,  5.    ,  5.    ],
#        [ 1.367 ,  5.    , -5.    ],
#        [ 2.752 , -5.    , -5.    ],
#        [-5.    , -5.    , -1.932 ]], dtype='float16')


# x = np.abs(p01.astype('float128'))
# x = np.sqrt(x)
# y = np.divide(p1.astype('float128'),x)
# y = np.sum(y,axis=1)
# y = y.astype('float16')
# opt_precision = np.array([ 14.133657,-5.293185,-9.131336,-29.695225] ,dtype='float16')
# just_precision = np.array([ 14.13 ,  -5.297  ,-9.14  ,-29.69 ] ,dtype='float16')
# just_opt = np.array( [ 14.13   ,-5.297  ,-9.14  ,-29.69 ] ,dtype='float16')
# unopt = np.array( [ 14.14   ,-5.293  ,-9.13  ,-29.7  ]  ,dtype='float16')

# print('opt_precision eror is',MSE(y, opt_precision),'\n reult is',opt_precision,'\n')
# print('just_precision eror is',MSE(y, just_precision),'\n result is',just_precision,'\n')
# print('just_opt eror is',MSE(y, just_opt),'\n result is',just_opt,'\n')
# print('unopt eror is',MSE(y, unopt),'\n result is',unopt,'\n')
