import torch
import torch.nn as nn

class Neuron(nn.Module):
    def __init__(self, response_func, threshold, leak, _scale=1, pool=False, out=False, name="neuron") -> None:
        super(Neuron, self).__init__()
        self.response_func = response_func
        self.threshold = threshold
        self.leak = leak
        self.name = name
        self._scale = torch.nn.parameter.Parameter(torch.tensor(_scale), requires_grad=False)
        self.pool = pool
        self.out = out
    
    def reset(self, soma_in, mem, soma_out):
        # torch.autograd.Variable(torch.zeros((batch_size, *self.in_shape), device=device), requires_grad=False)  # will not add in optimizer
        self.soma_in = soma_in
        self.mem = mem
        self.soma_out = soma_out

    def snn_forward(self, response_func_input):
        soma_in = self.response_func(response_func_input)
        self.soma_in += soma_in
        self.mem += soma_in
        # 后面再修改threshold， gt成一个函数
        ex_membrane = nn.functional.threshold(self.mem, self.threshold, 0)
        self.mem = self.mem - ex_membrane # hard reset
        # generate spike
        soma_out = ex_membrane.gt(0).type(torch.cuda.FloatTensor)
        if not self.pool:
            self.mem = self.leak * self.mem
        self.soma_out += soma_out
        return soma_out

    def forward(self, sum_x):
        sum_y = self.response_func(sum_x)
        out = torch.mul(sum_y, self.scale)
        return out - out.detach() + self.soma_out.detach()

    def updateScale(self, sign):
        if sign:
            index = self.soma_in >= self.soma_out
            new_scale = self.soma_out[index].sum()/self.soma_in[index].sum() if self.soma_in[index].sum() > 1e-3 else 1
            self._scale.data = 0.6 * new_scale + 0.4 * self._scale.data
        self.scale = self._scale.data * self.soma_in.gt(0).type(torch.cuda.FloatTensor)

        