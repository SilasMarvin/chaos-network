import torch as torch
import torch.nn as nn


nll = nn.NLLLoss()
log_softmax = nn.LogSoftmax(dim=0)
mish = nn.Mish()


def test_forward_batch():
    w1 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    w3 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    w4 = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    in1 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    in2 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    in3 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    in4 = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    out1 = w1 * in1 + w3 * in2 
    out2 = w2 * in1 + w4 * in2 
    output = torch.cat((out1.reshape(1), out2.reshape(1)))
    output = nll(log_softmax(output), torch.tensor(0))
    print("NLL Output1: ", output)
    output.backward()
    # output.backward(torch.ones_like(output))
    print("In1 Grad: ", in1.grad)
    print("In2 Grad: ", in2.grad)
    
    w1_first = w1.grad
    w2_first = w2.grad
    w3_first = w3.grad
    w4_first = w4.grad
    w1.grad = None
    w2.grad = None
    w3.grad = None
    w4.grad = None
    
    out1 = w1 * in3 + w3 * in4 
    out2 = w2 * in3 + w4 * in4 
    output = torch.cat((out1.reshape(1), out2.reshape(1)))
    output = nll(log_softmax(output), torch.tensor(1))
    print("NLL Output2: ", output)
    output.backward()
    print("In3 Grad: ", in3.grad)
    print("In4 Grad: ", in4.grad)
    
    w1_second = w1.grad
    w2_second = w2.grad
    w3_second = w3.grad
    w4_second = w4.grad
    
    print("W1: ", (w1_first + w1_second) / 2)
    print("W2: ", (w2_first + w2_second) / 2)
    print("W3: ", (w3_first + w3_second) / 2)
    print("W4: ", (w4_first + w4_second) / 2)
test_forward_batch()
