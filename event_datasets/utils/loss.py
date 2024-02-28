import torch


def cross_entropy_loss_and_accuracy(prediction:torch.Tensor, target:torch.Tensor):
    loss = torch.nn.CrossEntropyLoss()(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()*100
    return loss, accuracy

def mse_loss_and_accuracy(prediction:torch.Tensor, target:torch.Tensor):
    prediction, target = prediction.cpu(), target.cpu()
    labels = torch.zeros(prediction.shape).scatter_(1, target.unsqueeze(1), 1)
    loss = torch.nn.MSELoss(reduction='sum')(prediction, labels)
    accuracy = (prediction.argmax(1) == target).float().mean()*100
    return loss, accuracy
