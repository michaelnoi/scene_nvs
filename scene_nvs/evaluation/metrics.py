def calculate_psnr(pred, target):
    b = pred.size(0)
    mse_err = (pred - target).pow(2).view(b, -1).mean(dim=1)
    psnr = 10 * (1 / mse_err).log10()
    return psnr
