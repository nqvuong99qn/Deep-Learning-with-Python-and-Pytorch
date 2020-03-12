import torch
import matplotlib.pylab as plt
import torch.nn.functional as F

#create a tensor X

x = torch.tensor(2.0, requires_grad= True)
print("The tensor X: ", x)

y = x ** 2
print("the result of y = x ^ 2: ", y)

#take the derivative and try to print out the derivative at the value x  = 2
y.backward()
print("The derivative at x = 2: ", x.grad)

# calculate the y = x^2 + 2*x + 1 then find the derivative

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
print("The result of y", y)
y.backward()
print("the dervative at x = 2", x.grad)


#PRACTICE

x = torch.tensor(1.0, requires_grad=True)
y = 2 * x ** 3 + x
y.backward()
print("The derivative of y at x = 1: ", x.grad)


# PARTIAL DERIVATIVES
# calculate f(u, v) = v * u + u^2 at u =1, v = 2

u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u * v + u ** 2
print("The result of f at u=1, v=2: ", f)

#calculate the derivative with respect to u
f.backward()
print("The partial derivative with respect to u: ", u.grad)
print("The partial derivative with respect to v: ", v.grad)


#calculate the derivative with multiple values

x = torch.linspace(-10, 10, 10, requires_grad=True)
Y = x ** 2
y = torch.sum(x ** 2)

y.backward()
print(x.grad.numpy())
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

"""  take the derivative of RELU and plot out   """
# x = torch.linspace(-3, 3, 100, requires_grad=True)
# Y = F.relu(x)
# y = Y.sum()
# y.backward()
# plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
# plt.plot(x.detach().numpy(), x.grad.numpy(), label = 'derivative')
# plt.xlabel('x')
# plt.legend()
# plt.show()

## PRACTICE

#calculate the derivative of f = u * v + (u * v) ** 2 at u = 2, v = 1

u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(1.0, requires_grad=True)

f = u * v + (u + v) ** 2
f.backward()

print("The result of f to u:", u.grad)