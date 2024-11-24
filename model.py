from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for 2D input

        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)   # Dropout with a probability of 0.1
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.dropout2 = nn.Dropout(0.1)   # Dropout with a probability of 0.1
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.conv6 = nn.Conv2d(16, 16, 3)
        self.conv7 = nn.Conv2d(16, 10, 3)
        self.fc = nn.Conv2d(10, 10, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Output size: (1, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.bn1(x)
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.dropout1(x)
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.dropout2(x)
        x = F.relu(self.conv7(x))
        x = self.fc(x)
        x = self.gap(x)  # Using GAP instead of view
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    val_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return val_acc

def main():
    # CUDA setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Model setup
    torch.manual_seed(1)
    model = Net().to(device)
    
    # Print model summary
    summary(model, input_size=(1, 28, 28))
    
    # Data loader setup
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    # Load pre-trained model
    if device.type == 'cuda':
        model.load_state_dict(torch.load('best_model.pth'))
        
    else:
        model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
       
    
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Test the model
    val_acc = test(model, device, test_loader)
    print(f'Model accuracy: {val_acc:.2f}%')

if __name__ == '__main__':
    main()

