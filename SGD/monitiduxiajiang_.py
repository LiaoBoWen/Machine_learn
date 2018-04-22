import numpy as np
import  matplotlib.pyplot as plt

plot_x = np.linspace(-1,6,14)
print(plot_x)
plot_y = (plot_x-2.5)**2-1
print(plot_y)
# print(help(plt.plot))

def dJ(theta):
    return 2*(theta-2.5)

def J(theta):
    try:
        return (theta-2.5)**2-1
    except:
        return float("inf")

#梯度下降部分
def gradient_decent(initial_theta,eta,n_iters=1e4,epsilon=1e-8):
    theta = initial_theta
    theta_history.append(theta)
    i = 0
    while i < n_iters:
        gradient = dJ(theta)
        last_theta = theta
        theta = theta -eta*gradient
        theta_history.append(theta)

        if abs(J(theta)-J(last_theta)) <= epsilon:
            break
        i += 1

def plot_theta_history():
    plt.plot(plot_x, plot_y, "+")
    plt.plot(np.array(theta_history),J(np.array(theta_history)),color="r",marker='*',linewidth=0.1,markersize=3)
    plt.show()

eta = 0.01
theta = 0.0
epsilon = 1e-8
theta_history = []
gradient_decent(theta,eta)
plot_theta_history()

