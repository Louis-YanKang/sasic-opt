import argparse
import torch
import torch.nn as nn
import torchac
import pickle
from torchvision import transforms
import PIL

from sasic.utils import *
from sasic.model import StereoEncoderDecoder


def cdf(x, loc, scale):
    return 0.5 - 0.5 * (x - loc).sign() * torch.expm1(-(x - loc).abs() / scale)


class Decoder(nn.Module):
    """Decode image with learned image compression model
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model (nn.Module): torch model on device
            device (torch.device): torch.device
        """
        super().__init__()

        self.device = device

        #self.hd_left = model.model_left.hyper_decoder
        #self.hd_right = model.model_right.hyper_decoder

        #self.zl_loc = model.model_left.z_loc
        #self.zl_scale = model.model_left.z_scale
        #self.zr_loc = model.model_right.z_loc
        #self.zr_scale = model.model_right.z_scale

        self.sam1 = model.sam1
        self.sam2 = model.sam2
        self.sam3 = model.sam3

        self.decoder_left1 = model.decoder_left1
        self.decoder_left2 = model.decoder_left2
        self.decoder_left3 = model.decoder_left3        
        self.decoder_right1 = model.decoder_right1
        self.decoder_right2 = model.decoder_right2
        self.decoder_right3 = model.decoder_right3
 
    #def forward(self,zl_quant, yl_quant, zr_quant, yr_quant_res, shift) -> torch.tensor:
    def forward(self,yl_quant, yr_quant_res, shift) -> torch.tensor:
        """decode a compressed image file as a torch.tensor with the model m

        Args:
            filename (str): filename of encoded image

        Returns:
            torch.tensor: decoded torch.tensor
        """

        
        with torch.no_grad():

            #z_shape = zl_quant.shape
            #L = 50

            #zl_loc = self.zl_loc.expand(z_shape)
            #zr_loc = self.zr_loc.expand(z_shape)
            #zl_scale = self.zl_scale.expand(z_shape)
            #zr_scale = self.zr_scale.expand(z_shape) 
            #yl_probs = self.hd_left(zl_quant)
            #yl_loc, yl_scale = torch.chunk(yl_probs, 2, dim=1)
            
            y_right_from_left = left_to_right(yl_quant, shift)

            #zr_upscaled = nn.functional.interpolate(zr_quant, size=yl_quant.shape[-2:], mode='nearest')
            #hd_right_in = torch.cat([y_right_from_left, zr_upscaled], dim=1)
            #yr_probs = self.hd_right(hd_right_in)
            #yr_loc, yr_scale = torch.chunk(yr_probs, 2, dim=1)     

            yr_quant = yr_quant_res + y_right_from_left

            #yl_quant = yl_quant.to(device)
            #yr_quant = yr_quant.to(device)

            # decode left and right
            l_left, l_right = self.sam1(yl_quant, yr_quant)
            l_left = self.decoder_left1(l_left)
            l_right = self.decoder_right1(l_right)

            l_left, l_right = self.sam2(l_left, l_right)
            l_left = self.decoder_left2(l_left)
            l_right = self.decoder_right2(l_right)

            l_left, l_right = self.sam3(l_left, l_right)
            x_hat_left = self.decoder_left3(l_left)
            x_hat_right = self.decoder_right3(l_right)

        return torch.clamp(x_hat_left, 0, 1), torch.clamp(x_hat_right, 0, 1), shift
        #return zl_quant, yl_quant, zr_quant, yr_quant_res, shift
'''
    def decode_PIL_Image(self, filename: str) -> PIL.Image:
        """decode a compressed image file as a PIL.Image with the model m

        Args:
            filename (str): filename of encoded image

        Returns:
            PIL.Image: decoded PIL.Image
        """
        xl_hat, xr_hat = self.decode_tensor(filename)
        left = transforms.ToPILImage()(xl_hat[0])
        right = transforms.ToPILImage()(xr_hat[0])

        return left, right
'''

#if __name__ == '__main__':

'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_filename', type=str, required=True,
                        help='Compressed image file')
    parser.add_argument('--output_left', type=str, help='output filename left', default='left')
    parser.add_argument('--output_right', type=str, help='output filename left', default='right')
    parser.add_argument(
        '--model', type=str, help='A trained pytorch compression model.', required=True)
    parser.add_argument("--gpu", action='store_true', help="Use gpu?")
    args = parser.parse_args()

    if args.gpu:
        device = torch.device(f'cuda:{get_free_gpu()}')
    else:
        device = torch.device('cpu')
    print(f'  ## Using device: {device}')
'''
if __name__ == '__main__':

    device = torch.device(f'cuda:{get_free_gpu()}')
    print(f'  ## Using device: {device}')

    model = StereoEncoderDecoder().to(device)
    #model.load_state_dict(torch.load(args.model))
    #model.eval()
    dec = Decoder(model=model, device=device)
    
    checkpoint_model = torch.load( "/sasic/experiments/cityscapes_lambda0.01_500epochs/model_decoder_wrapper.pt")
    dec.load_state_dict(checkpoint_model)
    
    dec.eval()


    #dummy_input_z_hat_for_entropy = torch.randn(1,12,22,56).cuda()
    dummy_input_y_hat_for_entropy = torch.randn(1,12,88,224).cuda()
    #dummy_input_z_hat_for_entropy3 = torch.randn(1,12,22,56).cuda()
    dummy_input_y_hat_for_entropy3 = torch.randn(1,12,88,224).cuda()
    dummy_input_1906 = torch.randint(9,(12,)).cuda()

    #print("dummy_input_1906 shape ==============", dummy_input_1906.shape)

    #decode_inputs = (dummy_input_y_hat_for_entropy, dummy_input_y_hat_for_entropy3,dummy_input_1906)
    

    torch.onnx.export(
        dec, 
        args=(dummy_input_y_hat_for_entropy, dummy_input_y_hat_for_entropy3, dummy_input_1906),
        f="model_decode_wrapper.onnx",
        input_names=["yl","yr", "shift"],
        #do_constant_folding = False,
        #input_names=["y_hat_for_entropy","y_hat_for_entropy.3","z_hat_for_entropy","z_hat_for_entropy.3","1906"],
    )
 
'''
    print(f'  ## Decode image {args.image_filename}')
    with open(args.image_filename, "rb") as f:
        res = pickle.load(f)
    with torch.amp.autocast(device_type="cuda"):
        xl_hat, xr_hat = dec.decode_tensor(res[2].to(device),res[0].to(device), res[3].to(device), res[1].to(device), res[4])
        left = transforms.ToPILImage()(xl_hat[0])
        right = transforms.ToPILImage()(xr_hat[0])

    print(f'  ## Save image as {args.output_left}, {args.output_right}')
    left.save(args.output_left+'.png', 'PNG')
    right.save(args.output_right+'.png', 'PNG')
'''
