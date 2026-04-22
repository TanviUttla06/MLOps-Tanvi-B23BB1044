import matplotlib.pyplot as plt


def plot_metrics(loss, iou, dice):
    plt.figure()
    plt.plot(loss)
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(iou)
    plt.savefig("iou.png")

    plt.figure()
    plt.plot(dice)
    plt.savefig("dice.png")
