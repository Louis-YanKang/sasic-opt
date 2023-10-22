import argparse
import torch
import torch.nn as nn
import torchac
import pickle
from torchvision.transforms import ToTensor, CenterCrop, ToPILImage
from torchvision import transforms
import PIL



from sasic.utils import *
from sasic.model import StereoEncoderDecoder
from decode_onnx_wrapper import Decoder as decoder
from encode_onnx_wrapper import Encoder as encoder


class all_model(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device(f'cuda:{get_free_gpu()}')
        print(f'  ## Using device: {device}')

        model = StereoEncoderDecoder().to(device)
        checkpoint_model = torch.load( "/sasic/experiments/cityscapes_lambda0.01_500epochs/model.pt")
        model.load_state_dict(checkpoint_model)
    
        self.enc = encoder(model=model, device=device, L=50)
        self.dec = decoder(model=model, device=device)

    def forward(self, left: torch.tensor, right: torch.tensor):
        enc_yl, enc_yr, enc_zl, enc_zr, enc_shift = self.enc(left, right)
        dec_left, dec_right, shift = self.dec(enc_yl, enc_yr, enc_shift)

        return dec_left, dec_right
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=str, required=True)
    parser.add_argument('--right', type=str, required=True)
    parser.add_argument('--output_filename', type=str, default='out.sasic', help='output filename')
    parser.add_argument(
        '--model', type=str, help='A trained pytorch compression model.', required=True)
    parser.add_argument('--L', type=int, help='Vocabulary size.', default=50)
    parser.add_argument("--gpu", action='store_true', help="Use gpu?")
    args = parser.parse_args()

    if args.gpu:
        device = torch.device(f'cuda:{get_free_gpu()}')
    else:
        device = torch.device('cpu')
    print(f'  ## Using device: {device}')

    model = all_model()
    model.eval()

    dummy_input_left = torch.randn(1,3,352,896).cuda()
    dummy_input_right = torch.randn(1,3,352,896).cuda()
    torch.onnx.export(
        model, 
        args=(dummy_input_left, dummy_input_right),
        f="model_wrapper.onnx",
        input_names=["left", "right"],
        output_names = ["output_left", "output_right"]
    )

'''
    print(
        f'  ## Encode images {args.left} and {args.right} and save as {args.output_filename}')
    with torch.amp.autocast(device_type="cuda"):
        #left_image = PIL.Image.open(args.left).resize((352,896))
        left_image = PIL.Image.open(args.left).resize((896,352))
        #right_image = PIL.Image.open(args.right).resize((352,896))
        right_image = PIL.Image.open(args.right).resize((896,352))
        left_tensor = ToTensor()(left_image).unsqueeze(0)
        right_tensor =  ToTensor()(right_image).unsqueeze(0)

    xl_hat, xr_hat  = model.forward(left_tensor, right_tensor)


    print(f'  ## Decode image image_filename')
    with torch.amp.autocast(device_type="cuda"):
        left = transforms.ToPILImage()(xl_hat[0])
        right = transforms.ToPILImage()(xr_hat[0])

    print(f'  ## Save image as output_left, output_right')
    left.save('output_left.png', 'PNG')
    right.save('output_right.png', 'PNG')
'''

    
             
