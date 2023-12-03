
def plot_accuracies(time, train_acc, test_acc, val_acc=None, lr=None):
  plt.plot(time, train_acc, c='red', label='training accuracy', marker='x')
  plt.plot(time, test_acc, c='green', label='test accuracy', marker='x')
  if val_acc:
    plt.plot(time, val_acc, c='blue', label='validation accuracy', marker='x')
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  if lr:
    plt.suptitle("param = "+str(lr))
  plt.legend()
  plt.show()

def save_plots(train_err, valid_err, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train error'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion error'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('/content/drive/MyDrive/CPSC559/final_project/outputs/error.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('/content/drive/MyDrive/CPSC559/final_project/outputs/loss.png')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, '/content/drive/MyDrive/CPSC559/final_project/outputs/model.pth')

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    else:
        return data.to(device, non_blocking=True)