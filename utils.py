import matplotlib.pyplot as plt


def display_data(train_loader):
  batch_data, batch_label = next(iter(train_loader)) 

  fig = plt.figure()

  for i in range(12):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])
    
        
def display_model_stats(train_loss, train_accuracy, test_loss, test_accuracy):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_loss)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_accuracy)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_loss)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_accuracy)
  axs[1, 1].set_title("Test Accuracy")


def plot_test_incorrect_predictions(incorrect_pred):

    fig = plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        plt.imshow(incorrect_pred["images"][i].cpu().squeeze(0), cmap="gray")
        plt.title(
            repr(incorrect_pred["predicted_vals"][i])
            + " vs "
            + repr(incorrect_pred["ground_truths"][i])
        )
        plt.xticks([])
        plt.yticks([])
