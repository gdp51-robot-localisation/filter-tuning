# -*- coding: utf-8 -*-
"""
Created on Sun Feb 03 11:35:04 2019

@author: Carol
"""
import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
#import Beacon as beacon
#import Fetch_Optic as opflow
import pdb
import array

X = np.array([[129], [800], [578], [0]])


def get_obs(X):
    
    with open('observations.txt') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i][0:-3]
        lines[i] = lines[i].split(',')
        #pdb.set_trace()
        lines[i] = [-1 * float(lines[i][0])+X[0][0], 0.0, -1 * float(lines[i][2])+X[2][0], 0.0]
    return lines

def get_testmeth():
    with open('pts_SURF_H2700_20190416-123838.csv') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i][0:-2]
        lines[i] = lines[i].split(',')
        lines[i] = [float(lines[i][1]), float(lines[i][2])]
    testx = np.zeros(len(lines))
    testy = np.zeros(len(lines))
    #pdb.set_trace()
    count = 0
    for i in lines:
        #pdb.set_trace()
        testx[count] = i[0]
        testy[count] = i[1]
        count = count + 1
    #pdb.set_trace()
    return (testx, testy)

def prediction(x, xdot, y, ydot, t, a):
    A = np.array([[1, t, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, t],
                  [0, 0, 0, 1]])
    X = np.array([[x],
                  [xdot],
                  [y],
                  [ydot]])
    B = np.array([[0.5 * t ** 2],
                  [t],
                  [0.5 * t ** 2],
                  [t]])
    
    X_prime = A.dot(X) + B.dot(a)
    return X_prime


def covariance4d(sigma1, sigma2, sigma3, sigma4):
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1
    cov1_3 = sigma1 * sigma3
    cov3_1 = sigma3 * sigma1
    cov1_4 = sigma1 * sigma4
    cov4_1 = sigma4 * sigma1
    cov2_3 = sigma2 * sigma3
    cov3_2 = sigma3 * sigma2
    cov2_4 = sigma2 * sigma4
    cov4_2 = sigma4 * sigma2
    cov3_4 = sigma3 * sigma4
    cov4_3 = sigma4 * sigma3
    
    cov_matrix = np.array([[sigma1 ** 2, cov1_2, cov1_3, cov1_4],
                           [cov2_1, sigma2 ** 2, cov2_3, cov2_4],
                           [cov3_1, cov3_2, sigma3 ** 2, cov3_4],
                           [cov4_1, cov4_2, cov4_3, sigma4 ** 2]])
    
    return np.diag(np.diag(cov_matrix))

def covariance2d(sigma1, sigma2):
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1
    
    cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                           [cov2_1, sigma2 ** 2],
                           ])
    
    return np.diag(np.diag(cov_matrix))


def kalman(X, z):
    Xf = np.array([[0.0], [0.0], [0.0], [0.0]])
    
    # ICs
    a = 0
    t = 0.2


    # Process / Estimation Errors
    error_est_x = 5
    error_est_xdot = 1
    error_est_y = 5
    error_est_ydot = 1

    # Observation Errors
    #error_obs_x_b = 1  # Uncertainty in the measurement
    #error_obs_x_o = 1
    error_obs_x = 1
    #error_obs_y_b = 1
    #error_obs_y_o = 1
    error_obs_y = 1
    
    #Initial State
    #X = np.array([0, 1, 0, 1])
    #pdb.set_trace()
    
    #Initial Estimation Covariance Matrix
    P = covariance4d(error_est_x, error_est_xdot, error_est_y, error_est_ydot)

    A = np.array([
        
            [1, t, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1]

            ])
    

    n = len(z[0])
    #pdb.set_trace()
    count = 0
    posx = np.zeros(len(z[1:]))
    posy = np.zeros(len(z[1:]))
    for data in z[1:]:
        X = prediction(X[0][0], X[1][0], X[2][0], X[3][0], t, a)
        P = np.diag(np.diag(A.dot(P).dot(A.T)))
        H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
            ])
        
        R = covariance2d(error_obs_x, error_obs_y)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(inv(S))
        #pdb.set_trace()
        Y = H.dot(data).reshape(2, -1)  #makes 2 row col matrix, measuring 2 vals (x, y)
        X = X + K.dot(Y - H.dot(X))
        P = (np.identity(len(K)) - K.dot(H)).dot(P)
        #pdb.set_trace()
        
        posx[count] = X[0][0]
        posy[count] = X[2][0]
        
        Xf = np.hstack((Xf, X))
        count = count + 1
    #print("Kalman Filter State Matrix:\n", X)
    return posx, posy

opflow = get_obs(X)
(posx, posy) = kalman(X, opflow)
(testx, testy) = get_testmeth()

plt.plot(posx, posy)
plt.plot(testx, testy)