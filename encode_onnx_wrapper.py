import argparse
import torch
import torch.nn as nn
import torchac
import pickle
from torchvision.transforms import ToTensor, CenterCrop, ToPILImage
import PIL

from sasic.utils import *
from sasic.model import StereoEncoderDecoder


def cdf(x, loc, scale):
    return 0.5 - 0.5 * (x - loc).sign() * torch.expm1(-(x - loc).abs() / scale)


class Encoder(nn.Module):
    """Encode image with learned image compression model
    """

    def __init__(self, model: nn.Module, device: torch.device, L: int = 10):
        """
        Args:
            model (nn.Module): torch model on device
            device (torch.device): torch.device
            L (int, optional): Vocabulary size parameter: Half of the bin number for quantisation. 
                               Example: L = 2 means that only 5 different symbols are possible for every pixel. 
                               Defaults to 10.
        """
        super().__init__()
        model.eval()
        self.device = device
        self.L = 50
        self.model = model
 
    def forward(self, left: torch.tensor, right: torch.tensor): 
        """encode a tensor with the model m.

        Encoding process:
            1) Get the latent tensor "l" through applying the autoencoder encoder.
            2) Apply the hyperprior encoder on the latent tensor to obtain the hyperlatent "hl".
            3) Obtain the latent entropy parameters "l_loc" and "l_scale" by applying the 
               hyperprior decoder on the quantised hyperlatent.
            4) Quantise the latent.
            5) Encode hyperlatent and latent with the arithmetic encoder using the parameters from step 3
               for the latent and the entropy paramters stored in the model for the hyperlatent.
            6) Combine the encoded latent and hyperlatent with the vocabulary size "L" and
               the shape of the hyperlatent and save.

        Args:
            x (torch.tensor): tensor that will be decoded. Expects tensor to have the shape [B, C, H, W].
            filename (str): filename of the file where the encoded bytestream is saved.

        Returns:
            dict: a dictionary containing the encoded latent and hyperlatent as well as the hyperlatent shape
                  as well as the vocabulary parameter L.
        """
        xl = left.to(self.device)
        xr = right.to(self.device)

        #with torch.no_grad():
            # Left
        y_left = self.model.encoder(xl)
        out_left = self.model.model_left(y_left, training=False) # left image is compressed as always
        y_left_hat = out_left.y_hat

            # Right
        y_right = self.model.encoder(xr)
        shift = self.model.get_shift(y_left, y_right)                     # compute shift for left -> right
        y_right_from_left = left_to_right(out_left.y_hat, shift)    # warp latent left -> right
        y_right_residual = y_right - y_right_from_left              # compute latent residual
        out_right = self.model.model_right(y_right_residual, y_right_from_left, training=False)    # apply model to residual
        y_right_hat = out_right.y_hat + y_right_from_left           # Add missing information



        # Encode y
        ol_y, or_y = out_left.latents.y, out_right.latents.y
        yl_quantised, yl_loc, yl_scale = ol_y.value_hat, ol_y.loc, ol_y.scale
        yr_quantised, yr_loc, yr_scale = or_y.value_hat, or_y.loc, or_y.scale
        
        #yl_bytes = self._enc(yl_quantised.cpu(), yl_loc.cpu(), yl_scale.cpu())
        #yr_bytes = self._enc(yr_quantised.cpu(), yr_loc.cpu(), yr_scale.cpu())

        # Encode z
        ol_z, or_z = out_left.latents.z, out_right.latents.z
        zl_quantised, zl_loc, zl_scale = ol_z.value_hat, ol_z.loc, ol_z.scale
        zr_quantised, zr_loc, zr_scale = or_z.value_hat, or_z.loc, or_z.scale
        
        #zl_bytes = self._enc(zl_quantised.cpu(), zl_loc.cpu(), zl_scale.cpu())
        #zr_bytes = self._enc(zr_quantised.cpu(), zr_loc.cpu(), zr_scale.cpu())


        #byte_dict = {
        #    'yl': yl_bytes,
        #    'zl': zl_bytes,
        #    'yr': yr_bytes,
        #    'zr': zr_bytes,
        #    'shift': shift,
        #    'z_shape': zl_quantised.shape,
        #    'L': self.L
        #}
        
        #return byte_dict
        
        shift = torch.tensor(shift)

        return yl_quantised, yr_quantised, zl_quantised, zr_quantised, shift

'''
    def encode_PIL_Image(self, left: PIL.Image, right: PIL.Image, filename: str) -> dict:
        """encode a PIL image stereo pair with the model m

        Args:
            left/right (PIL.Image): images that will be compressed
            filename (str): filename of the file where the encoded bytestream is saved

        Returns:
            dict: a dictionary containing the encoded latent and hyperlatent as well as the hyperlatent shape
                  as well as the vocabulary parameter L.
        """
        left_tensor = ToTensor()(left).unsqueeze(0)
        right_tensor =  ToTensor()(right).unsqueeze(0)
        
        return self.encode_tensor(left_tensor, right_tensor, filename)

    def encode_file(self, left: str, right: str, output_filename: str) -> dict:
        left_image = PIL.Image.open(left)
        right_image = PIL.Image.open(right)

        return self.encode_PIL_Image(left_image, right_image, output_filename)
'''

if __name__ == '__main__':
    device = torch.device(f'cuda:{get_free_gpu()}')
    print(f'  ## Using device: {device}')

    model = StereoEncoderDecoder().to(device)
    enc = Encoder(model=model, device=device, L=50)
    checkpoint_model = torch.load( "/sasic/experiments/cityscapes_lambda0.01_500epochs/model_encoder_wrapper.pt")
    enc.load_state_dict(checkpoint_model)
    
    enc.eval()


    dummy_input_left = torch.randn(1,3,352,896).cuda()
    dummy_input_right = torch.randn(1,3,352,896).cuda()
    torch.onnx.export(
        enc, 
        args=(dummy_input_left, dummy_input_right),
        f="model_encode_wrapper.onnx",
        input_names=["left", "right"],
    )
    

             
