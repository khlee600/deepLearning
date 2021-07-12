# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 실재 값 생성
X = np.random.rand(100)
Y = 0.2 * X + 0.5
# 실재 값들의 분포 출력
plt.figure(figsize=(8,6))
plt.scatter(X, Y)
plt.show()


def drawGraph(x, pred, y) : # 분포 그래프를 그리기
# pred : 예측치  y : 실제 값
    plt.figure( figsize=(8,6))
    plt.scatter(x, y)               # 실재 값 분포
    plt.scatter(x, pred)            # 예측 값 분포
    plt.show()

# 예측 목표 함수 구성
# 1) 초기 기울기 예측
W = np.random.uniform(-1, 1)
b = np.random.uniform(-1, 1)
# learning rate 학습율 결정
LR = 0.7

# 2) 초기 예측치를 이용한 값 추정 및 도해
Y_pred = W * X + b
drawGraph(X, Y_pred, Y)

# 3) 경사 하강법에 의한 실재 값으로 접근하는 W, b 값 계산
'''
  하강시키기 위한 값 계산
  W_grad = 1/N * 'W에 대한 미분'((Y_hat - Y)**2) * LR
         = 2 * 1/N * (Y_pred - Y) * 'W에 대한 미분'(Y_hat - Y) * LR
         2과 LR모두 상수이므로 통합하여 LR에 녹여 넣는다면
         = LR * 1/N(Y_pred - Y) * 'W에 대한 미분'(W*x + d - Y)   <=Y는 상수
         = LR * (Y_pred - Y).mean() * x
  같은 방법으로 
  b_grad = LR * (Y_pred - Y)
'''
# 반복하여 수행하되 두 값들의 에러가 0.001 이면 중도 정지
for epoch in range(100):
    err = np.abs(Y_pred - Y).mean()
    if err < 0.001 :
        break
    W_grad = LR * ((Y_pred - Y)*X).mean()
    b_grad = LR * (Y_pred - Y).mean()
    
    W -= W_grad
    b -= b_grad
    
    Y_pred = W * X + b
    if (epoch % 10 == 0):
        drawGraph(X, Y_pred, Y)
        