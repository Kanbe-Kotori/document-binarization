def calc(pred, target, delta=1e-5):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * inter + delta) / (union + delta)
    return 1 - dice
