import matplotlib.pyplot as plt
import numpy as np

# This components plots the accuracies over different epochs.
def plot_accuracies(accuracy_train_epochs, accuracy_test_epochs):
    epochs = np.arange(1, len(accuracy_train_epochs) + 1)

    plt.figure(figsize=(10, 6))

    for epoch in epochs:
        plt.axvline(x=epoch, color='grey', alpha=0.3, linestyle='--', linewidth=0.8)

    plt.plot(epochs, accuracy_train_epochs, label="Training Accuracy", marker='o', linewidth=2)
    plt.plot(epochs, accuracy_test_epochs, label="Testing Accuracy", marker='s', linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy Over Epochs")
    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xticks(epochs)
    plt.ylim(0.0, 1.0)
    # Display the plot
    plt.tight_layout()
    plt.show()

# plot_accuracies(x,y) - used to call it
