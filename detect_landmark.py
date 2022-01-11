from __future__ import division

import sys, time, torch, random, argparse, PIL
from PIL import ImageFile, Image
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import numbers, numpy as np
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from xvision  import transforms, draw_image_by_points
from models   import obtain_model, remove_module_dict
from config_utils import load_configure
from blazeface import FaceExtractor, BlazeFace
from datasets import Point_Meta

def detect_landmark(image):
    cpu = False
    model_path = "./snapshot/sbr_model.pth"
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)


    facedet = BlazeFace().to(device)
    facedet.load_weights("./blazeface/blazeface.pth")
    facedet.load_anchors("./blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)
    im_real_faces = face_extractor.process_image(img=image)

    if len(im_real_faces['detections']) == 0:
        
        return None
    else:
        ymin, xmin, ymax, xmax = im_real_faces['detections'][0, 0:4].astype(int)
    
        face_ls = [xmin, ymin, xmax, ymax]

        if not cpu:
            assert torch.cuda.is_available(), 'CUDA is not available.'
            torch.backends.cudnn.enabled   = True
            torch.backends.cudnn.benchmark = True

        snapshot = Path(model_path)
        assert snapshot.exists(), 'The model path {:} does not exist'
        assert len(face_ls) == 4, 'Invalid face input : {:}'.format(face_ls)
        if cpu: snapshot = torch.load(snapshot, map_location='cpu')
        else  : snapshot = torch.load(snapshot)

        # General Data Argumentation
        mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
        normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        param = snapshot['args']
        eval_transform  = transforms.Compose([transforms.PreCrop(param.pre_crop_expand), 
                                            transforms.TrainScale2WH((param.crop_width, param.crop_height)), 
                                            transforms.ToTensor(), 
                                            normalize])

        model_config = load_configure(param.model_config, None)
        target = Point_Meta(param.num_pts, None, np.array(face_ls), image, param.data_indicator)
        image, target = eval_transform(image, target)
        temp_save_wh = target.temp_save_wh
        cropped_size = torch.IntTensor([temp_save_wh[1], temp_save_wh[0], temp_save_wh[2], temp_save_wh[3]])

        net = obtain_model(model_config, param.num_pts + 1)
        if not cpu: net = net.cuda()

        try:
            weights = remove_module_dict(snapshot['detector'])
        except:
            weights = remove_module_dict(snapshot['state_dict'])
        net.load_state_dict(weights)

        # network forward
        with torch.no_grad():
            if cpu: inputs = image.unsqueeze(0)
            else       : inputs = image.unsqueeze(0).cuda()
            batch_heatmaps, batch_locs, batch_scos = net(inputs)

        # obtain the locations on the image in the orignial size
        cpu = torch.device('cpu')
        np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy(), cropped_size.numpy()
        locations, scores = np_batch_locs[0,:-1,:], np.expand_dims(np_batch_scos[0,:-1], -1)

        scale_h, scale_w = cropped_size[0] * 1. / inputs.size(-2) , cropped_size[1] * 1. / inputs.size(-1)

        locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + cropped_size[3]
        prediction = np.concatenate((locations, scores), axis=1)[:, :2].astype(int)

        return prediction