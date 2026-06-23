import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

def metric_calc(Y_test, Y_pred, Y_pred_classes):

    confusion_mtx = confusion_matrix(Y_test, Y_pred_classes)

    accuracy = accuracy_score(Y_test, Y_pred_classes)
    precision = precision_score(Y_test, Y_pred_classes)
    recall = recall_score(Y_test, Y_pred_classes)
    f1 = f1_score(Y_test, Y_pred_classes)

    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    # Metrikleri kaydet
    with open("metrics_result/metrics.txt", "w") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")
        f.write(f"ROC AUC  : {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix\n")
        f.write(str(confusion_mtx))

    # Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(
        confusion_mtx,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Red Win", "Blue Win"],
        yticklabels=["Red Win", "Blue Win"]
    )

    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.savefig(
        "metrics_result/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8,6))

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.4f})"
    )

    plt.plot([0,1], [0,1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(
        "metrics_result/roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nKaydedilen dosyalar:")
    print("metrics.txt")
    print("confusion_matrix.png")
    print("roc_curve.png")