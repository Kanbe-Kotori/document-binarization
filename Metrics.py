import numpy
import torch
import torchvision
import DataLoader as D
import Net0_CNN as N0
import Net1_MSCNN as N1
import Net2_Unet as N2
import Net3_ViT as N3
import skimage.metrics as metrics


path = './dataset/2017'
model = N0.CNN()
# model = N1.MSCNN()
# model = N2.UNet()
# model = N3.ViTAE()
model.load_state_dict(torch.load('./pretrain/2016/net0-1000-dice.pkl'))
model.eval()

listMSE, listPSNR = [], []
for index in range(1, 21):
    listO = D.executeSingle(path + '/o/{}.bmp'.format(index), size=128, step=128)
    listT = D.executeSingle(path + '/t/{}_gt.bmp'.format(index), size=128, step=128, mode='L')

    transform = torchvision.transforms.ToTensor()
    listO = [model(transform(imgO).unsqueeze(dim=0)).squeeze(dim=0).detach() for imgO in listO]
    listT = [transform(imgT) for imgT in listT]
    imgT = torch.cat(listO, dim=2)
    imgReconT = torch.cat(listT, dim=2)

    mse = metrics.mean_squared_error(imgT.numpy(), imgReconT.numpy())
    psnr = metrics.peak_signal_noise_ratio(imgT.numpy(), imgReconT.numpy())
    if not numpy.isfinite(psnr):
        psnr = 50
    listMSE.append(mse)
    listPSNR.append(psnr)
print(numpy.mean(listMSE))
print(numpy.mean(listPSNR))
