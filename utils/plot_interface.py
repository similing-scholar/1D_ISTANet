import matplotlib.pyplot as plt


def plot_metric(df_history, metric):
    train_metrics = df_history["train_"+metric]
    val_metrics = df_history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)

    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.yscale('log')  # 设置纵轴为对数坐标
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'go-')
    min_val_loss = min(val_metrics)
    min_val_loss_index = val_metrics[val_metrics == min_val_loss].index[0]
    plt.scatter(min_val_loss_index + 1, min_val_loss, marker='*', color='r', label='best model', s=150)
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric, 'best model'])

    last_num = 20
    if len(epochs) >= last_num:
        plt.subplot(1, 2, 2)
        last_num_epochs = epochs[-last_num:]
        last_num_train_metrics = train_metrics[-last_num:]
        last_num_val_metrics = val_metrics[-last_num:]
        plt.plot(last_num_epochs, last_num_train_metrics, 'bo--')
        plt.plot(last_num_epochs, last_num_val_metrics, 'go-')
        min_val_loss_last_num = min(last_num_val_metrics)
        min_val_loss_index_last_num = list(last_num_val_metrics).index(min_val_loss_last_num)
        plt.scatter(last_num_epochs[min_val_loss_index_last_num], min_val_loss_last_num, marker='*', color='r',
                    label='best model', s=150)
        plt.title(f'Last {last_num} Epochs ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_"+metric, 'val_'+metric, 'best model'])

    plt.tight_layout()

    return fig
