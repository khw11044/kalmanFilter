#https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
import cv2
import numpy as np
import numpy.linalg as la



def kalman(x_esti,P,A,Q,B,u,z,H,R):

    x_pred = A @ x_esti + B @ u;         # B : controlMatrix -->  B @ u : gravity
    #  x_pred = A @ x_esti or  A @ x_esti - B @ u : upto
    P_pred  = A @ P @ A.T + Q;

    zp = H @ x_pred

    # si no hay observación solo hacemos predicción 
    if z is None:
        return x_pred, P_pred, zp

    epsilon = z - zp

    k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T +R)

    x_esti = x_pred + k @ epsilon;
    P  = (np.eye(len(P))-k @ H) @ P_pred;
    return x_esti, P, zp


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)                               # : H
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)    # : A 
    

    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()       # 예측
        statePre = self.kf.statePre
        statePost = self.kf.statePost
        Q = self.kf.processNoiseCov
        measurementNoiseCov = self.kf.measurementNoiseCov
        errorCovPre = self.kf.errorCovPre
        errorCovPost = self.kf.errorCovPost
        # B = self.kf.controlMatrix
        H = self.kf.measurementMatrix
        
        x, y = int(predicted[0]), int(predicted[1])
        return (x, y), statePre.T[0], statePost.T[0], errorCovPre #, errorCovPost

    def kal(self, mu, P, B, u, z):
        A = self.kf.transitionMatrix
        statePre = self.kf.statePre
        # B = self.kf.controlMatrix
        Q = self.kf.processNoiseCov
        # P = self.kf.errorCovPre                     # self.kf.errorCovPost
        H = self.kf.measurementMatrix
        R = self.kf.measurementNoiseCov

        # x_pred = A @ statePre.T[0] + B @ u        # B @ u 가 아래로 떨어트림
        x_pred = A @ mu + B @ u
        P_pred = A @ P @ A.T + Q / 4 
        zp = H @ x_pred

        if z is None:
            return x_pred, P_pred

        epsilon = z - zp

        k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T +R)

        x_esti = x_pred + k @ epsilon
        P  = (np.eye(len(P))-k @ H) @ P_pred
        return x_esti, P

# statePre : predicted state x'_k
# statePost : corrected state, x_k 
# transitionMatrix : 상태 변환행렬 : A 
# controlMatrix : 제어행렬 : B 
# measurementMatrix : 측정행렬 : H
# processNoiseCov : 프로세스 잡음 공분사 : Q 
# measurementNoiseCov : 측정 잡음 공분산 : R 
# errorCovPre : 사전 오차 추정 공분산 행렬 : P'_k
# gain : 칼만이득 : K_k
# errorCovPost : 사후 오차 추정 공분산 행렬 : P_k 