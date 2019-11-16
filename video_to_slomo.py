#!/usr/bin/env python3
import argparse
import os,shutil
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import model
import dataloader
import platform
from tqdm import tqdm
import cv2,skimage
import numpy as np
import os.path as op

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="/usr/bin/", help='path to ffmpeg.exe')
parser.add_argument("--video", type=str, default="/home/sherl/workspaces/git/use_tensorflow/use_tensor/GAN_slomo/testing_gif/car-turn.mp4", help='path of video to be converted')
parser.add_argument("--checkpoint", type=str, default="SuperSloMo.ckpt", help='path of checkpoint for pretrained model')
parser.add_argument("--fps", type=float, default=30, help='specify fps of output video. Default: 30.')
parser.add_argument("--sf", type=int, default=32, help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--output", type=str, default="nvidia_output.mp4", help='Specify output file name. Default: nvidia_output.mp4')
args = parser.parse_args()

def check():
    """
    Checks the validity of commandline arguments.

    Parameters
    ----------
        None

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    if (args.sf < 2):
        error = "Error: --sf/slomo factor has to be atleast 2"
    if (args.batch_size < 1):
        error = "Error: --batch_size has to be atleast 1"
    if (args.fps < 1):
        error = "Error: --fps has to be atleast 1"
    return error

def extract_frames(video, outDir):
    """
    Converts the `video` to images.

    Parameters
    ----------
        video : string
            full path to the video file.
        outDir : string
            path to directory to output the extracted images.

    Returns
    -------
        error : string
            Error message if error occurs otherwise blank string.
    """


    error = ""
    print('{} -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), video, outDir))
    retn = os.system('{} -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), video, outDir))
    if retn:
        error = "Error converting file:{}. Exiting.".format(video)
    return error

def create_video(dir):
    error = ""
    print('{} -r {} -i {}/%d.jpg -qscale:v 2 {}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir, args.output))
    retn = os.system('{} -r {} -i {}/%d.jpg -crf 17 -vcodec libx264 {}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir, args.output))
    if retn:
        error = "Error creating output video. Exiting."
    return error


def main():
    # Check if arguments are okay
    error = check()
    if error:
        print(error)
        exit(1)

    # Create extraction folder and extract frames
    IS_WINDOWS = 'Windows' == platform.system()
    extractionDir = "tmpSuperSloMo"
    
    #这里需要有个文件夹放截出来的帧，其实没必要费力去把这个文件夹搞成隐藏的
    if not IS_WINDOWS:
        # Assuming UNIX-like system where "." indicates hidden directories
        extractionDir = "." + extractionDir
    
    if os.path.isdir(extractionDir):
        rmtree(extractionDir)
    os.mkdir(extractionDir)
    if IS_WINDOWS:
        FILE_ATTRIBUTE_HIDDEN = 0x02
        # ctypes.windll only exists on Windows
        ctypes.windll.kernel32.SetFileAttributesW(extractionDir, FILE_ATTRIBUTE_HIDDEN)

    extractionPath = os.path.join(extractionDir, "input")
    outputPath     = os.path.join(extractionDir, "output")
    os.mkdir(extractionPath)
    os.mkdir(outputPath)
    error = extract_frames(args.video, extractionPath)
    if error:
        print(error)
        exit(1)

    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()]) #添加一个转化函数，后面用于对每个img做filter
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Load data
    videoFrames = dataloader.Video(root=extractionPath, transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    #第一个unet，用于计算光流
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    
    #这里只需要inference，去掉训练bp
    for param in flowComp.parameters():
        param.requires_grad = False
    #第二个UNET，用于合成 
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False
    
    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)
    #加载模型的checkpoint
    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 1

    '''
    Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。 
    '''
    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            
            #!!!!实现细节：在dim1连接起来！！！！
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            #flowout中应该是前0，1维度为0-》1的光流，2，3维度为1-》0光流
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            #保存原始视频帧
            for batchIndex in range(args.batch_size):
                (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".jpg"))
            frameCounter += 1

            # Generate intermediate frames
            for intermediateIndex in range(1, args.sf):
                t = intermediateIndex / args.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
                
                #第一个unet的初步结果，先看看这里的效果和我第一步骤对比
                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)
                
                #将上面一堆参数连接起来，送入下一个预测网络中
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                    
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                    
                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)
                
                wCoeff = [1 - t, t]

                #注意这里将mask加入到两帧合成上的方式，还加入了时间序列
                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                
                #这里看看他的中间结果怎么样?
                #Ft_p = (wCoeff[0] * g_I0_F_t_0 + wCoeff[1] * g_I1_F_t_1)
                #结果虽然时序上感觉有点抖动，但是清晰度上却是相当好，应该是loss函数的问题

                # Save intermediate frame
                #保存中间插入的帧
                for batchIndex in range(args.batch_size):
                    (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".jpg"))
                frameCounter += 1
            
            # Set counter accounting for batching of frames
            frameCounter += args.sf * (args.batch_size - 1)

    # Generate video from interpolated frames
    create_video(outputPath)

    # Remove temporary files
    rmtree(extractionDir)

    exit(0)

extractionDir = "tmpSuperSloMo"
def evaluate_video(invideopath):  
    # Check if arguments are okay
    
    #这里需要有个文件夹放截出来的帧，其实没必要费力去把这个文件夹搞成隐藏的
    if os.path.isdir(extractionDir):
        rmtree(extractionDir)
    os.mkdir(extractionDir)

    extractionPath = os.path.join(extractionDir, "input")

    os.mkdir(extractionPath)

    error = extract_frames(invideopath, extractionPath)
    if error:
        print(error)
        exit(1)
    
    evaluate_frame_dir(extractionPath)
    
    rmtree(extractionDir)


def evaluate_frame_dir(extractionPath):
    outputPath= os.path.join(extractionDir, "output")
    inputframe_dir=os.path.join(extractionDir, "inputframe")
    
    if op.exists(outputPath):rmtree(outputPath)
    if op.exists(inputframe_dir):rmtree(inputframe_dir)
    
    os.makedirs(outputPath, exist_ok=True)
    os.makedirs(inputframe_dir, exist_ok=True)
    
    frames_gt=os.listdir(extractionPath)
    frames_gt.sort()
    
    
    print (frames_gt)
    for ind,i in enumerate(frames_gt):
        if  ind%(args.sf )==0:
            shutil.copyfile(os.path.join(extractionPath, i), os.path.join(inputframe_dir, i))    
        
        

    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device == "cpu"):
        transform = transforms.Compose([transforms.ToTensor()]) #添加一个转化函数，后面用于对每个img做filter
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Load data
    videoFrames = dataloader.Video(root=inputframe_dir, transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    #第一个unet，用于计算光流
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    
    #这里只需要inference，去掉训练bp
    for param in flowComp.parameters():
        param.requires_grad = False
    #第二个UNET，用于合成 
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False
    
    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)
    #加载模型的checkpoint
    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])

    # Interpolate frames
    frameCounter = 0

    '''
    Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。 
    '''
    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            #print (I0.shape)
            
            #!!!!实现细节：在dim1连接起来！！！！
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            #flowout中应该是前0，1维度为0-》1的光流，2，3维度为1-》0光流
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            # Save reference frames in output folder
            #保存原始视频帧
            for batchIndex in range(args.batch_size):
                pass
                #(TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".jpg"))
            frameCounter += 1

            # Generate intermediate frames
            for intermediateIndex in range(1, args.sf):
                t = intermediateIndex / args.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
                
                #第一个unet的初步结果，先看看这里的效果和我第一步骤对比
                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)
                
                #将上面一堆参数连接起来，送入下一个预测网络中
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                    
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                    
                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)
                
                wCoeff = [1 - t, t]

                #注意这里将mask加入到两帧合成上的方式，还加入了时间序列
                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                #print (Ft_p.shape)
                #这里看看他的中间结果怎么样?
                #Ft_p = (wCoeff[0] * g_I0_F_t_0 + wCoeff[1] * g_I1_F_t_1)
                #结果虽然时序上感觉有点抖动，但是清晰度上却是相当好，应该是loss函数的问题

                # Save intermediate frame
                #保存中间插入的帧
                for batchIndex in range(args.batch_size):
                    #ttp="%06d.jpg"%(frameCounter + args.sf * batchIndex)
                    ttp=frames_gt[frameCounter + args.sf * batchIndex]
                    
                    ttp=os.path.join(outputPath, ttp )
                    #print (videoFrames.origDim) #(480, 270)
                    (TP(Ft_p[batchIndex].cpu().detach())).save( ttp)
                frameCounter += 1
            
            # Set counter accounting for batching of frames
            frameCounter += args.sf * (args.batch_size - 1)
    
    ssim_kep=[]
    psnr_kep=[]
    for i in os.listdir(outputPath):
        gt_img=cv2.imread( os.path.join(extractionPath, i) )
        genimg=cv2.imread( os.path.join(outputPath, i) )
        
        scale=0.5
        target_shape=( int(gt_img.shape[1]*scale),   int(gt_img.shape[0]*scale))
        #print (genimg.shape)
        gt_img=cv2.resize(gt_img, target_shape)
        genimg=cv2.resize(genimg, target_shape)
        
        psnr=skimage.measure.compare_psnr(gt_img, genimg, 255)
        ssim=skimage.measure.compare_ssim(gt_img, genimg, multichannel=True)
        
        psnr_kep.append(psnr)
        ssim_kep.append(ssim)
    print ("mean psnr:", np.mean(psnr_kep))
    print ("mean ssim:", np.mean(ssim_kep))
    # Generate video from interpolated frames
    #create_video(outputPath)

    # Remove temporary files
    rmtree(outputPath)
    rmtree(inputframe_dir)

ucf_path=r'/media/sherl/本地磁盘/data_DL/UCF101_results'
middleburey_path=r"/media/sherl/本地磁盘/data_DL/eval-color-allframes"
slowflow_train="/media/sherl/本地磁盘/data_DL/slow_flow/slow_flow_teaser/sequence"  #
MPI_sintel_clean="/media/sherl/本地磁盘/data_DL/MPI_Sintel/MPI-Sintel-complete/training/clean"
#main()
#evaluate_video()

def eval_rootdir(rdir):
    for i in os.listdir(rdir):
        tepdir=op.join(rdir, i)
        evaluate_frame_dir(tepdir)

eval_rootdir(slowflow_train)
    
    
    
    
    
    
    
    
    
    
    
    
    
