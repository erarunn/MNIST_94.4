import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import Net
from torchsummary import summary

def test_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = Net().to(device)
    
    try:
        # Load the pre-trained weights with proper device mapping
        if device.type == 'cuda':
            model.load_state_dict(torch.load('best_model.pth'))
        else:
            model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
            
        print("Successfully loaded pre-trained model")
        
        # Print model summary
        print("\nModel Architecture:")
        print("="*50)
        summary(model, input_size=(1, 28, 28))
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Load test dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                         ])),
            batch_size=1000, shuffle=True)
        
        # Test the model
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 * correct / total
        print(f'\nTest Accuracy: {accuracy:.2f}%')
        
        # Architecture checks
        has_batchnorm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
        has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
        
        print("\nArchitecture Checks:")
        print("="*50)
        print(f"✓ BatchNorm: {'Yes' if has_batchnorm else 'No'}")
        print(f"✓ Dropout: {'Yes' if has_dropout else 'No'}")
        print(f"✓ Global Average Pooling: {'Yes' if has_gap else 'No'}")
        print(f"✓ Parameter Count: {'Pass' if total_params < 20000 else 'Fail'}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    test_model() 