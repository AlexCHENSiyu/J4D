import torch
import torch.nn.functional as F
from torchaudio.functional import convolve
import math
import numpy as np
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.special import entr
import HDQ
import rate_calc
import cv2
# from matplotlib.pyplot import figure

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smallest_float32 = torch.finfo(torch.float32).tiny
largest_float32 = torch.finfo(torch.float32).max


# Define the custom module
class CustomModel(nn.Module):
    def __init__(self, jpeg_layer, underlying_model):
        super(CustomModel, self).__init__()
        self.jpeg_layer = jpeg_layer
        self.underlying_model = underlying_model

    def forward(self, x, validation=False, *args, **kwargs):
        if validation:
            x, bbp, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy = self.jpeg_layer(x, validation)
            self.underlying_model.eval()
            x = self.underlying_model(x, *args, **kwargs)
            return x, bbp, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy
        else:
            x, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy = self.jpeg_layer(x, validation)
            self.underlying_model.eval()
            x = self.underlying_model(x, *args, **kwargs)
            return x, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy

def colorSpace_machineSenstivity():
    ColorSpaceW =  np.array( [
                                [  0.299, 0.587, 0.114 ],
                                [ -0.168736, -0.331264, 0.5  ],
                                [ 0.5, -0.418688, -0.081312]
                             ], dtype=float)
    InvColorSpaceW = np.linalg.inv(ColorSpaceW).astype(float)
    
    eps = 1e-5
    InvColorSpaceW[np.abs(InvColorSpaceW) < eps] = 0 # adjust the weight matrix in the color matrix
    sen_map = np.ones((3,64), dtype=float)
    BiasPerImageFlag = False
    return ColorSpaceW, InvColorSpaceW, sen_map, BiasPerImageFlag


def PSNR_cal_RGB(s, r, max_value=255):
    s = np.array(s, dtype=np.float64) 
    r = np.array(r, dtype=np.float64)
    
    height, width, channel = np.shape(s)
    size = height*width 

    sb,sg,sr = cv2.split(s)
    rb,rg,rr = cv2.split(r)
    
    
    mseb = ((sb-rb)**2).sum()
    mseg = ((sg-rg)**2).sum()
    mser = ((sr-rr)**2).sum()
    mse = (mseb+mseg+mser)
    
    if mse == 0:
        return float('inf')
    
    MSE = mse/(3*size)
    psnr = 10*math.log10(255**2/MSE)
    return round(psnr,2)

class resize_compression(object):
    def __init__(self, compressor, Resize_value=256, centerCrop_value=224, interpolation_method=transforms.InterpolationMode.BILINEAR):
        super().__init__()
        self.compressor = compressor
        self.Resize_value = Resize_value
        self.centerCrop_value = centerCrop_value
        self.interpolation_method = interpolation_method

    def  __call__(self, sample_org):
        sample_org = transforms.Resize(self.Resize_value, interpolation=self.interpolation_method)(sample_org)
        sample_org = transforms.CenterCrop([self.centerCrop_value, self.centerCrop_value])(sample_org)
        sample = self.compressor(sample_org)
        BPP = sample['BPP']
        sample = sample['image']
        return sample, BPP


class HDQ_transforms(object):
    def __init__(self, QF_Y, QF_C, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ):
        self.mean = mean
        self.std = std
        self.QF_Y = QF_Y
        self.QF_C = QF_C
        self.J = 4
        self.a = 4
        self.b = 4
        # Sensivity is ignored in this case as we do not need in the HDQ
        self.ColorSpaceW, self.InvColorSpaceW, _ , self.BiasPerImageFlag = colorSpace_machineSenstivity()
        self.comparsionRunner = HDQ.__call__
    def __call__(self, sample):
        sample = np.asarray(sample, dtype=float)
        sample = np.transpose(sample, (2,0,1))
        compressed_img, BPP = self.comparsionRunner(    sample, 
                                                        self.ColorSpaceW,
                                                        self.InvColorSpaceW,
                                                        self.BiasPerImageFlag,
                                                        self.J, self.a, self.b,
                                                        self.QF_Y, self.QF_C)
        compressed_img = np.round(compressed_img)    
        compressed_img = np.uint8(compressed_img) 
        compressed_img = np.transpose(compressed_img, (1,2,0))
        compressed_img = transforms.ToTensor()(compressed_img)
        compressed_img = transforms.Normalize(mean=self.mean, std=self.std)(compressed_img)
        return {'image': compressed_img, 'BPP': BPP}

    def __repr__(self):
        return 'HDQ_transforms!'

class JPEG_layer_transforms(object):
    def __init__(self, opt , img_shape=(224, 224, 3), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.jpeg_layer = JPEG_layer(opt, img_shape, mean, std, validation=True)
        self.jpeg_layer.lum_qtable = self.jpeg_layer.lum_qtable.squeeze(0)
        self.jpeg_layer.chrom_qtable = self.jpeg_layer.chrom_qtable.squeeze(0)

        self.jpeg_layer.alpha_lum = self.jpeg_layer.alpha_lum.squeeze(0)
        self.jpeg_layer.alpha_chrom = self.jpeg_layer.alpha_chrom.squeeze(0)
        self.comparsionRunner = rate_calc.__call__

    def __call__(self, sample):
        sample = np.asarray(sample, dtype=float)
        sample = np.transpose(sample, (2,0,1))
        sample = torch.tensor(sample, dtype=torch.float32)
        compressed_img, sampled_reconstructed_space = self.jpeg_layer(sample, validation=True)
        BPP = self.comparsionRunner(sampled_reconstructed_space)
        return {'image': compressed_img, 'BPP': BPP}



# Define your neural network
class JPEG_layer(nn.Module):
    def construct(self, opt):
        self.num_bit = opt.num_bit
        self.Q_inital = opt.Q_inital
        self.batch_size = opt.batch_size
        self.block_size = opt.block_size
        # self.num_channels = self.img_shape[-1]
        self.num_channels = 3
        self.num_block = int((self.img_shape[0]*self.img_shape[1])/(self.block_size**2))
        self.min_Q_Step = opt.min_Q_Step
        self.max_Q_Step = opt.max_Q_Step
        self.analysis = False
        self.low_level  = -2**(self.num_bit-1) + 1
        self.high_level =  2**(self.num_bit-1)

        self.num_non_zero_q = opt.num_non_zero_q # 5
        self.num_non_zero_q_on_1_side = math.floor(self.num_non_zero_q/2) # 2
        self.q_idx = torch.arange(0, self.num_non_zero_q) #[0, 1, ... , 4]

        if (not self.validation) and torch.cuda.is_available():
            self.q_idx = self.q_idx.cuda()
    
    
    def __init__(self, opt, img_shape, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], validation=False):
        super(JPEG_layer, self).__init__()
        self.JPEG_alpha_trainable = opt.JPEG_alpha_trainable
        self.img_shape = img_shape
        self.validation = validation
        
        self.construct(opt)

        # Define two q table
        self.lum_qtable = torch.ones((1, 1, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32) # shape: [1,1,1,8,8,1]
        self.chrom_qtable = torch.ones((1, 1, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32) # shape: [1,1,1,8,8,1]
        self.alpha_lum = torch.ones((1, 1, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32) # shape: [1,1,1,8,8,1]
        self.alpha_chrom = torch.ones((1, 1, 1, opt.block_size, opt.block_size, 1), dtype=torch.float32) # shape: [1,1,1,8,8,1]

        nn.init.constant_(self.lum_qtable , self.Q_inital)
        nn.init.constant_(self.chrom_qtable , self.Q_inital)
        nn.init.constant_(self.alpha_lum, opt.JPEG_alpha)
        nn.init.constant_(self.alpha_chrom, opt.JPEG_alpha)

        # QT_Y = torch.tensor(quantizationTable(QF=50, Luminance=True))
        # QT_C = torch.tensor(quantizationTable(QF=50, Luminance=False))
        # self.initial_values = torch.stack([QT_Y, QT_C, QT_C], dim=0)

        self.block_dct = block_dct_callable(self.lum_qtable, validation)
        self.block_idct = block_idct_callable(self.lum_qtable, validation)
        

        # self.rgb_to_ycbcr = rgb_to_ycbcr_batch()
        # self.ycbcr_to_rgb = ycbcr_to_rgb_batch()

        # Define alpha
        
        if not validation:
            self.lum_qtable = nn.Parameter(self.lum_qtable)
            self.chrom_qtable = nn.Parameter(self.chrom_qtable)
            
            if opt.JPEG_alpha_trainable:
                print("==> Alpha is trainable per quantization step")
                self.alpha_lum = nn.Parameter(self.alpha_lum)
                self.alpha_chrom = nn.Parameter(self.alpha_chrom)
            else:
                print("==> Alpha is not trainable")
                if torch.cuda.is_available():
                    self.alpha_lum = self.alpha_lum.cuda()
                    self.alpha_chrom = self.alpha_chrom.cuda()

        self.Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.register_forward_pre_hook(self.reinitialize_q_table_alpha)

    def forward(self, x, validation=False):
        if validation:
            return self.val(x)
        else:
            return self.train(x)

        
    def val(self, input_RGB):
        # mean_per_image = input.mean(dim=(2, 3), keepdim=True)  # Shape (N, C, 1, 1)
        # print("lum max:", self.lum_qtable.max().item(), "lum min:", self.lum_qtable.min().item(), "chrom max:", self.chrom_qtable.max().item(), "chrom max", self.chrom_qtable.min().item())
        # print("lum_qtable:", self.lum_qtable, "chrom_qtable", self.chrom_qtable)
        input = input_RGB - 128
        input_DCT_block_batch = self.block_dct(blockify(rgb_to_ycbcr(input), self.block_size)).unsqueeze(-1) # shape: [bs,ch,B,size,size,1]

        input_lum   = input_DCT_block_batch[0:1, ...] 
        input_chrom = input_DCT_block_batch[1:3, ...] 

        # Quantization
        idx_lum   = torch.round(input_lum / self.lum_qtable) # shape: [bs,ch,B,size,size,1]
        idx_chrom = torch.round(input_chrom / self.chrom_qtable.expand(2, 1, self.block_size, self.block_size, 1)) # shape: [bs,ch,B,size,size,1]
        
        idx_lum   =  torch.clamp(idx_lum   - self.num_non_zero_q_on_1_side, min=self.low_level, max=self.high_level - self.num_non_zero_q) # shape: [bs,ch,B,size,size,1]
        idx_chrom =  torch.clamp(idx_chrom - self.num_non_zero_q_on_1_side, min=self.low_level, max=self.high_level - self.num_non_zero_q) # shape: [bs,ch,B,size,size,1]
    
        idx_lum = idx_lum.expand(-1, -1, -1, -1, self.num_non_zero_q) + self.q_idx # shape: [bs,ch,B,size,size,5]
        idx_chrom = idx_chrom.expand(2, -1, -1, -1, self.num_non_zero_q) + self.q_idx  # shape: [bs,ch,B,size,size,5]

        iq_lum = idx_lum * self.lum_qtable
        iq_chrom = idx_chrom * self.chrom_qtable
        
        distortion_MSE_mask_lum = F.mse_loss(iq_lum, input_lum.expand(-1, -1, -1, -1, self.num_non_zero_q), reduction='none')  # shape: [bs,ch,B,size,size,5]
        distortion_MSE_mask_chrom = F.mse_loss(iq_chrom, input_chrom.expand(-1, -1, -1, -1, self.num_non_zero_q), reduction='none') # shape: [bs,ch,B,size,size,5] 

        # clamp due to softmax will generate Nan in certain cases.
        cmpf_mask_lum = torch.clamp(F.softmax(-self.alpha_lum * distortion_MSE_mask_lum, dim=-1), min=1e-10) # shape: [bs,ch,B,size,size,5]
        cmpf_mask_chrom = torch.clamp(F.softmax(-self.alpha_chrom.expand(2, -1, -1, -1, 1) * distortion_MSE_mask_chrom , dim=-1), min=1e-10) # shape: [bs,ch,B,size,size,5] 
        
        # Q_p
        Y_shape = cmpf_mask_lum.shape
        Y_indices_sampled = torch.multinomial(cmpf_mask_lum.view(-1, Y_shape[-1]), num_samples=1, replacement=False).view(Y_shape[:-1])
        Y_indices_sampled = Y_indices_sampled.unsqueeze(-1).expand(*Y_shape[:-1], Y_shape[-1])
        sampled_reconstructed_space_lum = iq_lum.gather(-1, Y_indices_sampled).squeeze(-1)[..., 0]


        CbCr_shape = cmpf_mask_chrom.shape
        CbCr_indices_sampled = torch.multinomial(cmpf_mask_chrom.view(-1, CbCr_shape[-1]), num_samples=1, replacement=False).view(CbCr_shape[:-1])
        CbCr_indices_sampled = CbCr_indices_sampled.unsqueeze(-1).expand(*CbCr_shape[:-1], CbCr_shape[-1])
        sampled_reconstructed_space_chrom = iq_chrom.gather(-1, CbCr_indices_sampled).squeeze(-1)[..., 0]

        # Q_d
        # estimated_reconstructed_space_lum = torch.sum(cmpf_mask_lum * iq_lum , -1)
        # estimated_reconstructed_space_chrom  = torch.sum(cmpf_mask_chrom * iq_chrom, -1)
        
        
        sampled_reconstructed_space = torch.cat((sampled_reconstructed_space_lum, sampled_reconstructed_space_chrom), 0)
        
        norm_img =  ycbcr_to_rgb(deblockify(self.block_idct(sampled_reconstructed_space), (self.img_shape[0], self.img_shape[1])))
        norm_img += 128

        sampled_reconstructed_space = deblockify(sampled_reconstructed_space, (self.img_shape[0], self.img_shape[1]))
        
        # print(self.lum_qtable.squeeze((0,1,-1)))
        # print(self.chrom_qtable.squeeze((0,1,-1)))
        # print(self.alpha_lum.squeeze((0,1,-1)))
        # print(self.alpha_chrom.squeeze((0,1,-1)))
        
        # Here I am doing the same effect of a tensor by using Scale2One then normalize using the standard normalization
        norm_img = self.normalize(self.Scale2One(norm_img))

        
        assert torch.all((idx_lum >= -2**10+1) & (idx_lum <= 2**10-1)), \
            f"Indexes out of bounds: invalid values found {idx_lum[~((idx_lum >= -2**10+1) & (idx_lum <= 2**10-1))]}"
        assert torch.all((idx_chrom >= -2**10+1) & (idx_chrom <= 2**10-1)), \
            f"Indexes out of bounds: invalid values found {idx_chrom[~((idx_chrom >= -2**10+1) & (idx_chrom <= 2**10-1))]}"
        
        
        return norm_img, sampled_reconstructed_space



        
    def train(self, input_RGB):
        # mean_per_image = input.mean(dim=(2, 3), keepdim=True)  # Shape (N, C, 1, 1)
        # print("lum max:", self.lum_qtable.max().item(), "lum min:", self.lum_qtable.min().item(), "chrom max:", self.chrom_qtable.max().item(), "chrom max", self.chrom_qtable.min().item())
        # print("lum_qtable:", self.lum_qtable, "chrom_qtable", self.chrom_qtable)
        
        input = input_RGB - 128
        input_DCT_block_batch = self.block_dct(blockify(rgb_to_ycbcr(input), self.block_size)).unsqueeze(-1) # shape: [bs,ch,B,size,size,1]

        input_lum   = input_DCT_block_batch[:, 0:1, ...] 
        input_chrom = input_DCT_block_batch[:, 1:3, ...] 

        # Quantization
        idx_lum   = torch.round(input_lum / self.lum_qtable) # shape: [bs,ch,B,size,size,1]
        idx_chrom = torch.round(input_chrom / self.chrom_qtable.expand(1, 2, 1, self.block_size, self.block_size, 1)) # shape: [bs,ch,B,size,size,1]
        
        idx_lum   =  torch.clamp(idx_lum   - self.num_non_zero_q_on_1_side, min=self.low_level, max=self.high_level - self.num_non_zero_q) # shape: [bs,ch,B,size,size,1]
        idx_chrom =  torch.clamp(idx_chrom - self.num_non_zero_q_on_1_side, min=self.low_level, max=self.high_level - self.num_non_zero_q) # shape: [bs,ch,B,size,size,1]
    
        idx_lum = idx_lum.expand(-1, -1, -1, -1, -1, self.num_non_zero_q) + self.q_idx  # shape: [bs,ch,B,size,size,5]
        idx_chrom = idx_chrom.expand(-1, 2, -1, -1, -1, self.num_non_zero_q) + self.q_idx  # shape: [bs,ch,B,size,size,5]
        
        idx_lum = idx_lum.detach()
        idx_chrom = idx_chrom.detach()

        iq_lum = idx_lum * self.lum_qtable
        iq_chrom = idx_chrom * self.chrom_qtable
        
        distortion_MSE_mask_lum = F.mse_loss(iq_lum, input_lum.expand(-1, -1, -1, -1, -1, self.num_non_zero_q), reduction='none')  # shape: [bs,ch,B,size,size,5]
        distortion_MSE_mask_chrom = F.mse_loss(iq_chrom, input_chrom.expand(-1, -1, -1, -1, -1, self.num_non_zero_q), reduction='none') # shape: [bs,ch,B,size,size,5] 

        # clamp due to softmax will generate Nan in certain cases.
        cmpf_mask_lum = torch.clamp(F.softmax(-self.alpha_lum * distortion_MSE_mask_lum, dim=-1), min=1e-10) # shape: [bs,ch,B,size,size,5]
        cmpf_mask_chrom = torch.clamp(F.softmax(-self.alpha_chrom.expand(-1, 2, -1, -1, -1, 1) * distortion_MSE_mask_chrom , dim=-1), min=1e-10) # shape: [bs,ch,B,size,size,5] 
        
        # Q_d
        estimated_reconstructed_space_lum = torch.sum(cmpf_mask_lum * iq_lum , -1)
        estimated_reconstructed_space_chrom  = torch.sum(cmpf_mask_chrom * iq_chrom, -1)
        
        estimated_reconstructed_space = torch.cat((estimated_reconstructed_space_lum, estimated_reconstructed_space_chrom), 1)
        

        
        norm_img =  ycbcr_to_rgb(deblockify(self.block_idct(estimated_reconstructed_space), (self.img_shape[0], self.img_shape[1])))
        norm_img += 128
        
        # Here I am doing the same effect of a tensor by using Scale2One then normalize using the standard normalization
        norm_img = self.normalize(self.Scale2One(norm_img))

        
        assert torch.all((idx_lum >= -2**10+1) & (idx_lum <= 2**10-1)), \
            f"Indexes out of bounds: invalid values found {idx_lum[~((idx_lum >= -2**10+1) & (idx_lum <= 2**10-1))]}"
        assert torch.all((idx_chrom >= -2**10+1) & (idx_chrom <= 2**10-1)), \
            f"Indexes out of bounds: invalid values found {idx_chrom[~((idx_chrom >= -2**10+1) & (idx_chrom <= 2**10-1))]}"
        

        lum_dc_entropy, chrom_dc_entropy = self.calculate_dc_entropy(idx_lum, idx_chrom, cmpf_mask_lum, cmpf_mask_chrom)
        # print("dc_entropy", weighted_dc_entropy.item())
        
        lum_ac_entropy, chrom_ac_entropy = self.calculate_ac_entropy(idx_lum, idx_chrom, cmpf_mask_lum, cmpf_mask_chrom)
        # print("ac_entropy", weighted_ac_entropy.item())
        
        if self.analysis:
            return norm_img, estimated_reconstructed_space, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy
        else:
            return norm_img, lum_dc_entropy, chrom_dc_entropy, lum_ac_entropy, chrom_ac_entropy


    def reinitialize_q_table_alpha(self, model, input):
        # re-initialize the q table before each forward propagation
        with torch.no_grad():
            if self.min_Q_Step == 0:
                self.lum_qtable.copy_(torch.clamp(self.lum_qtable, min=smallest_float32, max=self.max_Q_Step))
                self.chrom_qtable.copy_(torch.clamp(self.chrom_qtable, min=smallest_float32, max=self.max_Q_Step))
            else:
                self.lum_qtable.copy_(torch.clamp(self.lum_qtable, min=self.min_Q_Step, max=self.max_Q_Step))
                self.chrom_qtable.copy_(torch.clamp(self.chrom_qtable, min=self.min_Q_Step, max=self.max_Q_Step))

            if self.JPEG_alpha_trainable:
                self.alpha_lum.copy_(torch.clamp(self.alpha_lum, min=smallest_float32))
                self.alpha_chrom.copy_(torch.clamp(self.alpha_chrom, min=smallest_float32))
    
    
    def get_diff_indexes(self, indexes, indexes_shifted):
        middle = indexes[:,:,:,0] - indexes_shifted[:,:,:,0] # shape: [bs,ch,B]
        diff_indexes = middle.unsqueeze(-1) # shape: [bs,ch,B,1]
        add_index = torch.arange(-self.num_non_zero_q + 1, self.num_non_zero_q).cuda() # [-4,-3,-2,-1,0,1,2,3,4]
        diff_indexes = diff_indexes.expand(-1, -1, -1, 2*self.num_non_zero_q-1) + add_index # shape: [bs,ch,B,9]
        return diff_indexes
    
    
    def calculate_dc_entropy(self, idx_lum, idx_chrom, cmpf_mask_lum, cmpf_mask_chrom):
        # get the entropy of DC part.
        input_shape = idx_lum.shape # shape: [bs,ch,B,size,size,5]
        num_blocks = input_shape[2] # 748
        batch_size = input_shape[0]
        
        cmpf_lum = cmpf_mask_lum
        cmpf_chrom = cmpf_mask_chrom
        indexes_lum = (idx_lum).to(torch.int64) # [-2^10+1,2^10-1]
        indexes_chrom = (idx_chrom).to(torch.int64) # [-2^10+1,2^10-1]
        
        # get DC part, [bs,ch,B,size,size,5] to shape: [bs,ch,B,5]
        cmpf_lum_dc = cmpf_lum[:,:,:,0,0,:]
        cmpf_chrom_dc = cmpf_chrom[:,:,:,0,0,:]
        indexes_lum_dc = indexes_lum[:,:,:,0,0,:]
        indexes_chrom_dc = indexes_chrom[:,:,:,0,0,:]

        #-----------------------------------------Preparation---------------------------------------------------------
        # shift to right by 1 block and set first to default value. shape: [bs,ch,B,5]
        cmpf_lum_dc_shifted = torch.roll(cmpf_lum_dc, shifts=1, dims=2).detach()
        cmpf_lum_dc_shifted[:,:,0,:] = 1e-10
        cmpf_lum_dc_shifted[:,:,0,0] = 1.0  - (self.num_non_zero_q - 1) * (1e-10) # set first to [1,0,0,0,0]
        
        cmpf_chrom_dc_shifted = torch.roll(cmpf_chrom_dc, shifts=1, dims=2)
        cmpf_chrom_dc_shifted[:,:,0,:] = 1e-10
        cmpf_chrom_dc_shifted[:,:,0,0] = 1.0 - (self.num_non_zero_q - 1) * (1e-10) # set first to [1,0,0,0,0]
        
        indexes_lum_dc_shifted = torch.roll(indexes_lum_dc, shifts=1, dims=2)
        indexes_lum_dc_shifted[:,:,0,:] = self.q_idx # set first to [0,1,2,3,4]
        
        indexes_chrom_dc_shifted = torch.roll(indexes_chrom_dc, shifts=1, dims=2)
        indexes_chrom_dc_shifted[:,:,0,:] = self.q_idx # set first to [0,1,2,3,4]
        
        # reverse first so convolve later is along the right diagonal
        cmpf_lum_dc_shifted_tensor = torch.flip(cmpf_lum_dc_shifted, dims=[-1]) # shape: [bs,ch,B,5]
        cmpf_diff_lum_dc = convolve(cmpf_lum_dc_shifted_tensor, cmpf_lum_dc) # shape: [bs,ch,B,9]
        
        cmpf_chrom_dc_shifted_tensor = torch.flip(cmpf_chrom_dc_shifted, dims=[-1]) # shape: [bs,ch,B,5]
        cmpf_diff_chrom_dc = convolve(cmpf_chrom_dc_shifted_tensor, cmpf_chrom_dc) # shape: [bs,ch,B,9]
        
        indexes_diff_lum_dc = self.get_diff_indexes(indexes_lum_dc, indexes_lum_dc_shifted) # shape: [bs,ch,B,9]
        indexes_diff_chrom_dc = self.get_diff_indexes(indexes_chrom_dc, indexes_chrom_dc_shifted) # shape: [bs,ch,B,9]
        
        # change to shape: [bs,ch,B*9]
        cmpf_diff_lum_dc = cmpf_diff_lum_dc.reshape(input_shape[0], 1, -1)
        cmpf_diff_chrom_dc = cmpf_diff_chrom_dc.reshape(input_shape[0], 2, -1)
        
        indexes_diff_lum_dc = indexes_diff_lum_dc.reshape(input_shape[0], 1, -1)
        indexes_diff_chrom_dc = indexes_diff_chrom_dc.reshape(input_shape[0], 2, -1)
        
        # [-2^11+2,2^11-2] to [0,2^12-4]
        indexes_diff_lum_dc += (2**(self.num_bit)-2)
        indexes_diff_chrom_dc += (2**(self.num_bit)-2)
        
        assert torch.all((indexes_diff_lum_dc >= 0) & (indexes_diff_lum_dc <= 2**12-4)), \
            f"Indexes out of bounds: invalid values found {indexes_diff_lum_dc[~((indexes_diff_lum_dc >= 0) & (indexes_diff_lum_dc <= 2**12-4))]}"
        assert torch.all((indexes_diff_chrom_dc >= 0) & (indexes_diff_chrom_dc <= 2**12-4)), \
            f"Indexes out of bounds: invalid values found {indexes_diff_chrom_dc[~((indexes_diff_chrom_dc >= 0) & (indexes_diff_chrom_dc <= 2**12-4))]}"

        #-----------------------------------------DC entropy---------------------------------------------------------
        
        # get distribution of each dc frequency, shape: [bs,ch,2*(num_bit+1)-3]
        p_lum_dc = torch.zeros((batch_size, 1, 2**(self.num_bit+1)-3), dtype=torch.float32).to(indexes_diff_lum_dc.device).scatter_add_(2, indexes_diff_lum_dc, cmpf_diff_lum_dc)
        p_lum_dc /= num_blocks
        # print("sum lum dc prob:", torch.sum(p_lum_dc, axis=-1, keepdims=True))
        
        # shape: [bs,ch,2*(num_bit+1)-3]
        p_chrom_dc = torch.zeros((batch_size, 2, 2**(self.num_bit+1)-3), dtype=torch.float32).to(indexes_diff_chrom_dc.device).scatter_add_(2, indexes_diff_chrom_dc, cmpf_diff_chrom_dc)
        p_chrom_dc /= num_blocks
        # print("sum chrom dc prob:", torch.sum(p_chrom_dc, axis=-1, keepdims=True))
        
        # get entropy of each dc frequency and average over batch.
        lum_dc_entropy = torch.sum(entr(p_lum_dc), dim=-1).mean()
        chrom_dc_entropy = torch.sum(entr(p_chrom_dc), dim=-1).mean()

        return lum_dc_entropy, chrom_dc_entropy

    
    def calculate_ac_entropy(self, idx_lum, idx_chrom, cmpf_mask_lum, cmpf_mask_chrom):
        # get the entropy of AC part.
        input_shape = idx_lum.shape # shape: [bs,ch,B,size,size,5]
        num_blocks = input_shape[2] # 748
        batch_size = input_shape[0]
       
        cmpf_lum = cmpf_mask_lum
        cmpf_chrom = cmpf_mask_chrom
        indexes_lum = (idx_lum).to(torch.int64) # [-2^10+1,2^10-1]
        indexes_chrom = (idx_chrom).to(torch.int64) # [-2^10+1,2^10-1]

        # [bs,ch,B,size,size,5] to shape: [bs,size,size,ch,B,5]
        cmpf_lum = cmpf_lum.permute(0, 3, 4, 1, 2, 5)
        cmpf_chrom = cmpf_chrom.permute(0, 3, 4, 1, 2, 5)
        indexes_lum = indexes_lum.permute(0, 3, 4, 1, 2, 5)
        indexes_chrom = indexes_chrom.permute(0, 3, 4, 1, 2, 5)
        
        # [bs,block_size,block_size,ch,B,5] to shape: [bs,block_size*block_size,ch*B*5]
        cmpf_lum = cmpf_lum.reshape(input_shape[0], input_shape[3] * input_shape[4], 1 * input_shape[2] * input_shape[5])
        cmpf_chrom = cmpf_chrom.reshape(input_shape[0], input_shape[3] * input_shape[4], 2 * input_shape[2] * input_shape[5])
        indexes_lum = indexes_lum.reshape(input_shape[0], input_shape[3] * input_shape[4], 1 * input_shape[2] * input_shape[5])
        indexes_chrom = indexes_chrom.reshape(input_shape[0], input_shape[3] * input_shape[4], 2 * input_shape[2] * input_shape[5])

        # delete DC to shape: [bs,size*size-1,ch*B*5]
        cmpf_lum = cmpf_lum[:, 1:, :]
        cmpf_chrom = cmpf_chrom[:, 1:, :]
        indexes_lum = indexes_lum[:, 1:, :]
        indexes_chrom = indexes_chrom[:, 1:, :]

        # change value range [-2^10+1,2^10-1] to range [0, 2^11-2]
        indexes_lum -= self.low_level
        indexes_chrom -= self.low_level
        
        assert torch.all((indexes_lum >= 0) & (indexes_lum <= 2**11-2)), \
            f"Indexes out of bounds: invalid values found {indexes_lum[~((indexes_lum >= 0) & (indexes_lum <= 2**11-2))]}"
        assert torch.all((indexes_chrom >= 0) & (indexes_chrom <= 2**11-2)), \
            f"Indexes out of bounds: invalid values found {indexes_chrom[~((indexes_chrom >= 0) & (indexes_chrom <= 2**11-2))]}"

        #----------------------------------------------AC entropy-----------------------------------------------
        
        # get distribution of each ac frequency, shape:[bs,size*size-1,2^11-1]
        p_lum_ac = torch.zeros((batch_size, self.block_size*self.block_size-1, 2**(self.num_bit)-1), dtype=torch.float32).to(indexes_lum.device).scatter_add(2, indexes_lum, cmpf_lum)
        p_lum_ac /= num_blocks 
        # # print("sum lum ac prob:", torch.sum(self.p_lum_ac, axis=-1,keepdims=True))
    
        # shape: [bs,size*size-1,2^num_bit-1]
        p_chrom_ac = torch.zeros((batch_size, self.block_size*self.block_size-1, 2**(self.num_bit)-1), dtype=torch.float32).to(indexes_chrom.device).scatter_add(2, indexes_chrom, cmpf_chrom)
        p_chrom_ac /= (2 * num_blocks)
        # # print("sum chrom ac prob:", torch.sum(p_chrom_ac, axis=-1,keepdims=True))
        
        # get entropy of each ac frequency
        lum_ac_entropy = torch.sum(entr(p_lum_ac), dim=-1) # shape: [bs,size*size-1] type: numpy
        # self.visualize_entropy(lum_ac_entropy)
        
        chrom_ac_entropy = torch.sum(entr(p_chrom_ac), dim=-1) # shape: [bs,size*size-1] type: numpy
        # self.visualize_entropy(chrom_ac_entropy)

        # sum ac entropy and average over batch.
        lum_ac_entropy = lum_ac_entropy.sum(axis=1).mean()
        chrom_ac_entropy = chrom_ac_entropy.sum(axis=1).mean()
        
        return lum_ac_entropy, chrom_ac_entropy


    def visualize_entropy(self, entropy):
        # entropy of shape:
        entropy_np = entropy.detach().cpu().numpy()
        mean_entropy = entropy_np.mean(axis=0)
        new_array = np.insert(mean_entropy, 0, 0)
        reshaped_array = new_array.reshape(8, 8)
        flattened_entropy = zigzag(reshaped_array).flatten()
        
        plt.figure(figsize=(8, 4))
        plt.plot(flattened_entropy)
        plt.title("Entropy over Frequency")
        plt.xlabel("Frequency Index")
        plt.ylabel("Entropy(Bit-Rate)")
        plt.xticks(np.arange(0, len(flattened_entropy)+1, 8))
        plt.grid(True)
        plt.show()
    

def zigzag(matrix):
    zigzag = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                        [2, 4, 7, 13, 16, 26, 29, 42],
                        [3, 8, 12, 17, 25, 30, 41, 43],
                        [9, 11, 18, 24, 31, 40, 44, 53],
                        [10, 19, 23, 32, 39, 45, 52, 54],
                        [20, 22, 33, 38, 46, 51, 55, 60],
                        [21, 34, 37, 47, 50, 56, 59, 61],
                        [35, 36, 48, 49, 57, 58, 62, 63]])

    # Get the shape of the matrix
    matrix_size = np.shape(matrix)

    # Calculate the size of the vector
    vector_size = np.shape(zigzag)

    # Check if the matrix size matches the vector size
    if matrix_size != vector_size:
        raise ValueError("The matrix size does not match the vector size.")

    # Create an empty vector to store the values
    vector = np.zeros(matrix_size[0] * matrix_size[1])

    # Iterate over each element in the matrix and place it in the corresponding position in the vector
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            index = zigzag[i, j]
            vector[index] = matrix[i, j]

    return vector


def inverse_zigzag(vector):
    zigzag = torch.tensor([[0, 1, 5, 6, 14, 15, 27, 28],
                           [2, 4, 7, 13, 16, 26, 29, 42],
                           [3, 8, 12, 17, 25, 30, 41, 43],
                           [9, 11, 18, 24, 31, 40, 44, 53],
                           [10, 19, 23, 32, 39, 45, 52, 54],
                           [20, 22, 33, 38, 46, 51, 55, 60],
                           [21, 34, 37, 47, 50, 56, 59, 61],
                           [35, 36, 48, 49, 57, 58, 62, 63]])

    # Get the shape of the vector
    vector_shape = vector.size()

    # Calculate the size of the 2D matrix
    matrix_size = zigzag.size()

    # Check if the vector size matches the matrix size
    if vector_shape != matrix_size:
        raise ValueError("The vector size does not match the matrix size.")

    # Create an empty matrix to store the values
    matrix = torch.zeros(matrix_size)

    # Iterate over each element in the vector and place it in the corresponding position in the matrix
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            index = zigzag[i, j]
            matrix[i, j] = vector[index]

    return matrix


def get_average_model_magnitude(model):
    total_magnitude = 0.0
    num_parameters = 0
    for param in model.parameters():
        param_magnitude = torch.mean(torch.abs(param)).item()
        total_magnitude += param_magnitude
        num_parameters += 1
    if num_parameters == 0:
        return 0.0
    return total_magnitude / num_parameters


def get_max_model_magnitude(model):
    max_magnitude = 0.0
    for param in model.parameters():
        param_max = torch.max(torch.abs(param)).item()
        if param_max > max_magnitude:
            max_magnitude = param_max
    return max_magnitude


def quantizationTable(QF=50, Luminance=True):
    #  Luminance quantization table
    quantizationTableData = np.ones((8, 8), dtype=np.float32)

    if QF == 100:
        # print(quantizationTableData)
        return quantizationTableData

    if Luminance == True:  # Y channel
        #  Standard
        # * 16 11 10 16 24  40  51  61
        # * 12 12 14 19 26  58  60  55
        # * 14 13 16 24 40  57  69  56
        # * 14 17 22 29 51  87  80  62
        # * 18 22 37 56 68  109 103 77
        # * 24 35 55 64 81  104 113 92
        # * 49 64 78 87 103 121 120 101
        # * 72 92 95 98 112 100 103 99

        quantizationTableData = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                          [12, 12, 14, 19, 26, 58, 60, 55],
                                          [14, 13, 16, 24, 40, 57, 69, 56],
                                          [14, 17, 22, 29, 51, 87, 80, 62],
                                          [18, 22, 37, 56, 68, 109, 103, 77],
                                          [24, 35, 55, 64, 81, 104, 113, 92],
                                          [49, 64, 78, 87, 103, 121, 120, 101],
                                          [72, 92, 95, 98, 112, 100, 103, 99]])
        
    else:
        # Standard Cb Cr channel
        # 17 18  24  47  99  99  99  99
        # 18 21  26  66  99  99  99  99
        # 24 26  56  99  99  99  99  99
        # 47 66  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        
        quantizationTableData = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                          [18, 21, 26, 66, 99, 99, 99, 99],
                                          [24, 26, 56, 99, 99, 99, 99, 99],
                                          [47, 66, 99, 99, 99, 99, 99, 99],
                                          [99, 99, 99, 99, 99, 99, 99, 99],
                                          [99, 99, 99, 99, 99, 99, 99, 99],
                                          [99, 99, 99, 87, 99, 99, 99, 99],
                                          [99, 99, 99, 99, 99, 99, 99, 99]])

    if QF >= 1:
        if QF < 50:
            S = 5000 / QF
        else:
            S = 200 - 2 * QF

        for i in range(8):
            for j in range(8):
                q = (50 + S * quantizationTableData[i][j]) / 100
                q = np.clip(np.floor(q), 1, 255)
                quantizationTableData[i][j] = q
    return quantizationTableData


def get_zigzag():
    zigzag = torch.tensor(( [[0,   1,   5,  6,   14,  15,  27,  28],
                             [2,   4,   7,  13,  16,  26,  29,  42],
                             [3,   8,  12,  17,  25,  30,  41,  43],
                             [9,   11, 18,  24,  31,  40,  44,  53],
                             [10,  19, 23,  32,  39,  45,  52,  54],
                             [20,  22, 33,  38,  46,  51,  55,  60],
                             [21,  34, 37,  47,  50,  56,  59,  61],
                             [35,  36, 48,  49,  57,  58,  62,  63]]))
    return zigzag


def _normalize(N: int) -> torch.Tensor:
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    
    return n @ n.t()


def _harmonics(N: int) -> torch.Tensor:
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)
    

def block_dct(blocks: torch.Tensor) -> torch.Tensor:
    N = blocks.shape[3]

    n = _normalize(N).float().to(blocks.device)
    h = _harmonics(N).float().to(blocks.device)

    # if blocks.is_cuda:
    #     n = n.cuda()
    #     h = h.cuda()
    
    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ blocks @ h)

    return coeff


def block_idct(coeff: torch.Tensor) -> torch.Tensor:
    N = coeff.shape[3]

    n = _normalize(N).to(coeff.device)
    h = _harmonics(N).to(coeff.device)

    # if coeff.is_cuda:
    #     n = n.cuda()
    #     h = h.cuda()

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


class block_dct_callable(nn.Module):
    """Callable class."""
    
    def __init__(self, blocks: torch.Tensor, validation: bool):
        super(block_dct_callable, self).__init__()
        self.N = blocks.shape[3]
        self.n = _normalize(self.N).float()
        self.h = _harmonics(self.N).float()
        if (not validation) and torch.cuda.is_available():
            self.n = self.n.cuda()
            self.h = self.h.cuda()

    def forward(self, blocks):
        coeff = (1 / math.sqrt(2 * self.N)) * self.n * (self.h.t() @ blocks @ self.h)
        return coeff


class block_idct_callable(nn.Module):   
    def __init__(self, blocks: torch.Tensor, validation: bool):
        super(block_idct_callable, self).__init__()
        self.N = blocks.shape[3]
        self.n = _normalize(self.N).float()
        self.h = _harmonics(self.N).float()
    
        if (not validation) and torch.cuda.is_available():
            self.n = self.n.cuda()
            self.h = self.h.cuda()

    def forward(self, coeff):
        im = (1 / math.sqrt(2 * self.N)) * (self.h @ (self.n * coeff) @ self.h.t())
        return im


def rgb_to_ycbcr(image: torch.Tensor,
                 W_r = 0.299,
                 W_g = 0.587,
                 W_b = 0.114) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = W_r * r + W_g * g + W_b * b
    cb: torch.Tensor = (b - y) /(2*(1-W_b)) + delta
    cr: torch.Tensor = (r - y) /(2*(1-W_r)) + delta
    return torch.stack((y, cb, cr), -3)


class rgb_to_ycbcr_batch(object):
    # Define the transformation matrix as a torch tensor
    def __init__(self):
        # W_r = 0.299
        # W_g = 0.587
        # W_b = 0.114
        # self.T = torch.tensor([ [W_r, W_g, W_b],
        #                         [-W_r/2, -W_g/2, (1-W_b)/2],
        #                         [(1-W_r)/2, -W_g/2, -W_b/2]], dtype=torch.float32)

        self.T = torch.tensor([ [0.299, 0.587, 0.114],
                                [-0.168736, -0.331264, 0.5],
                                [0.5, -0.418688, -0.081312]], dtype=torch.float32)

        self.B = torch.tensor([0, 0.5, 0.5], dtype=torch.float32)
        if torch.cuda.is_available():
            self.T = self.T.cuda()
            self.B = self.B.cuda()

    def __call__(self, images: torch.Tensor)-> torch.Tensor:            
        # Reshape the batch of images from (N, 3, H, W) to (N, H*W, 3)
        N, C, H, W = images.shape
        images_reshaped = images.permute(0, 2, 3, 1).reshape(N, -1,  C)
        
        # Perform the matrix multiplication and add the bias
        ycbcr_reshaped = torch.matmul(images_reshaped, self.T.T) + self.B
        
        # Reshape back to (N, H, W, 3) and then permute to (N, 3, H, W)
        ycbcr_images = ycbcr_reshaped.view(N, H, W, C).permute(0, 3, 1, 2)
        
        return ycbcr_images


def ycbcr_to_rgb(image: torch.Tensor,
                 W_r=0.299,
                 W_g=0.587,
                 W_b=0.114) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 2*(1-W_r) * cr_shifted
    g: torch.Tensor = y - 2*(1-W_r)*W_r/W_g * cr_shifted - 2*(1-W_b)*W_b/W_g * cb_shifted
    b: torch.Tensor = y + 2*(1-W_b) * cb_shifted
    return torch.stack([r, g, b], -3)


class ycbcr_to_rgb_batch(object):
    # Define the transformation matrix as a torch tensor
    def __init__(self):
        # W_r=0.299
        # W_g=0.587
        # W_b=0.114
        # self.T_inv = torch.tensor([ [1.0, 0.0,  2*(1-W_r)], 
        #                             [1.0, - 2*(1-W_b)*W_b/W_g,  - 2*(1-W_r)*W_r/W_g], 
        #                             [1.0, 2*(1-W_b), 0.0]], dtype=torch.float32)
        
        self.T_inv = torch.tensor([[1.0, 0.0, 1.402],
                          [1.0, -0.344136, -0.714136],
                          [1.0, 1.772, 0.0]], dtype=torch.float32)
        self.B_inv = torch.tensor([0, 0.5, 0.5], dtype=torch.float32)
        
        
        if torch.cuda.is_available():
            self.T_inv = self.T_inv.cuda()
            self.B_inv = self.B_inv.cuda()

    def __call__(self, images: torch.Tensor)-> torch.Tensor:            
        # Reshape the batch of images from (N, 3, H, W) to (N, H*W, 3)
        N, C, H, W = images.shape
        images_reshaped = images.permute(0, 2, 3, 1).reshape(N, -1, C)
        
        # Subtract the bias from Cb and Cr channels
        images_reshaped -= self.B_inv
        
        # Perform the matrix multiplication
        rgb_reshaped = torch.matmul(images_reshaped, self.T_inv.T)
        
        # Reshape back to (N, H, W, 3) and then permute to (N, 3, H, W)
        rgb_images = rgb_reshaped.view(N, H, W, C).permute(0, 3, 1, 2)
        return rgb_images


def convert_NCWL_to_NWLC(img):
    return torch.transpose(torch.transpose(img,1,2),2,3)


def pad_shape(Num, size=8):
    res = Num%size
    pad = 1
    if(res == 0):
        pad = 0
    n = (Num//size+pad)*size
    return n


def blockify(im: torch.Tensor, size: int) -> torch.Tensor:
    shape = im.shape[-2:]
    padded_shape = [pad_shape(shape[0]),pad_shape(shape[1])]
    paded_im = F.pad(im, (0,padded_shape[1]-shape[1], 0,padded_shape[0]-shape[0]), 'constant',0)
    if  paded_im.dim() == 3 :
        ch = paded_im.shape[0]
        h = paded_im.shape[1]
        w = paded_im.shape[2]
        paded_im = paded_im.reshape(ch, 1, h, w)
        paded_im = F.unfold(paded_im, kernel_size=(size, size), stride=(size, size))
        paded_im = paded_im.transpose(1, 2)
        paded_im = paded_im.reshape(ch, -1, size, size)
    else:
        bs = paded_im.shape[0]
        ch = paded_im.shape[1]
        h = paded_im.shape[2]
        w = paded_im.shape[3]
        paded_im = paded_im.reshape(bs * ch, 1, h, w)
        paded_im = F.unfold(paded_im, kernel_size=(size, size), stride=(size, size))
        paded_im = paded_im.transpose(1, 2)
        paded_im = paded_im.reshape(bs, ch, -1, size, size)
    return paded_im


def deblockify(blocks: torch.Tensor, size) -> torch.Tensor:
    padded_shape = pad_shape(size[0]),pad_shape(size[1])
    if  blocks.dim() == 4 :
        ch = blocks.shape[0]
        block_size = blocks.shape[3]
        blocks = blocks.reshape(ch, -1, int(block_size ** 2))
        blocks = blocks.transpose(1, 2)
        blocks = torch.nn.functional.fold(blocks, output_size=padded_shape, kernel_size=(block_size, block_size), stride=(block_size, block_size))
        blocks = blocks.reshape(ch, padded_shape[0], padded_shape[1])
        blocks = blocks[:,:size[0],:size[1]]
    else:
        bs = blocks.shape[0]
        ch = blocks.shape[1]
        block_size = blocks.shape[3]
        blocks = blocks.reshape(bs * ch, -1, int(block_size ** 2))
        blocks = blocks.transpose(1, 2)
        blocks = torch.nn.functional.fold(blocks, output_size=padded_shape, kernel_size=(block_size, block_size), stride=(block_size, block_size))
        blocks = blocks.reshape(bs, ch, padded_shape[0], padded_shape[1])
        blocks = blocks[:,:,:size[0],:size[1]]
    return blocks


def load_3x3_weight(model_name = "Alexnet"):
    rt_arr = np.zeros((3,3))
    seq_weight = np.genfromtxt("color_conv_W/"+model_name+"_W_OPT.txt")
    for i in range(3):
        for j in range(3):
            rt_arr[i, j] = seq_weight[i*3+j]
    return torch.Tensor(rt_arr)


# def plot_confidence_interval(x, top, bottom, mean, horizontal_line_width=0.25, color='#2187bb',label=None,alpha=1):
#     left = x - horizontal_line_width / 2
#     right = x + horizontal_line_width / 2
#     plt.plot([x, x], [top, bottom], color=color,alpha=0.7*alpha)
#     plt.plot([left, right], [top, top], color=color,alpha=0.7*alpha)
#     plt.plot([left, right], [bottom, bottom], color=color,alpha=0.7*alpha)
#     plt.plot(x, mean, 'o', color=color, label=label,alpha=alpha)
#     return mean