import torch
from torch.optim import SGD
x = torch.tensor([0.1, 0.1, 0.1, 0.1]) # the input
w = torch.tensor([1], requires_grad=True, dtype=torch.float32) # the learned variance
t = torch.tensor([10], requires_grad=True, dtype=torch.float32) # the predicted values
o = SGD([w, t], lr=1)

for i in range(3000):

    prediction = torch.multiply(t, x)
    p_var = torch.multiply(w, prediction)
    sig = torch.nn.Sigmoid()
    p_var = sig(p_var) * 0.5
    p = torch.distributions.Normal(torch.ones(4), p_var)
    p_val = - torch.mean(p.log_prob(prediction))
    o.zero_grad()
    p_val.backward()
    o.step()
    
    if i % 100 == 0:
        print("p_value", p_val.item())
        print("Predicted Variance", p_var)
        print("Predicted Values", prediction)
        print("")