#!/usr/bin/env python
"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs #
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import os.path as osp
import sys
caffe_path = osp.join('/home/chenxh/YSJ_bottom-up-attention/bottom-up-attention/caffe/python')
sys.path.insert(0, caffe_path)
import caffe
import argparse
import pprint
import time, os, sys
import base64 # Encode and decode
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json
csv.field_size_limit(sys.maxsize)


# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 20  # ！！！！！！！！！！！！！！
MAX_BOXES = 20  # ！！！！！！！！！！！！！！
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']


# Return: {'video_1': [(path1, frame_1), (path2, frame_2), ......] 
#          'video_2': [(path1, frame_1), (path2, frame_2), ......]}
def load_frame_ids2():
    ids = {}
    print('Start loading frame ids!')
    dir_path = '/home/chenxh/YSJ/gif_frames/'
    for di in os.listdir(dir_path):
        ids[di] = []
        frames = os.listdir(dir_path + di) # ['0.jpg', '1.jpg', ...]
        n_frames = len(frames)

        index = [int(frame.split('.')[0]) for frame in frames]
        index.sort(reverse=False)
        max_frames = 32 # ！！！！！！！！！！！！！！
        if n_frames >= max_frames:
            step = n_frames // max_frames
            remain = n_frames % max_frames
            index_20 = index[step - 1: n_frames - remain: step] # Crucial, important and critical !!!!
        else:
            # Pad the last frame until the length is 'n_frames':
			index_20 = index
			for i in range(max_frames - n_frames):
				index_20.append(index[-1])
		
        for frame_id in index_20:
            file_name = str(frame_id) + '.png'
            frame_path = os.path.join(dir_path, di, file_name) # /xx/tumblr_232snw/1.jpg
            ids[di].append((frame_path, frame_id))
        ids[di].sort(key=lambda pair: pair[1], reverse=False)
    print('Finish loading frame ids!')
    print('Num videos in total: ', len(ids))
    return ids

    
def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):

    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
   
    return { 
        'image_id': image_id,  # id in tsv file
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes])
    }   


"""
python generate_tsv_ysj.py  --gpu 0  --def /home/chenxh/YSJ_bottom_up_attention/bottom-up-attention/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --net /home/chenxh/YSJ_bottom_up_attention/bottom-up-attention/data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel  --out out  --cfg /home/chenxh/YSJ_bottom_up_attention/bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml   
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile', ##
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

    
def generate_tsv(gpu_id, prototxt, weights, image_dict, outfile):
    # image_ids: the frame ids for a certain video. It's a dict, not a list:
    #           [(path1, frame_1), (path2, frame_2), ..., (pathN, frame_M)]
    # outfile: the tsv path for a certain video.
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
	
    count = 0
    total = len((image_dict.keys()))
    for key in list(image_dict.keys()):
		# Extract every video as a independent tsv file
        image_ids = image_dict[key]
    	wanted_ids = set([image_id[1] for image_id in image_ids])
        found_ids = set()
		
        # Exclude the frames already extracted
        outfile = outfile + key + '.tsv'
    	if os.path.exists(outfile):
            with open(outfile) as tsvfile:
                reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
            	for item in reader: # row
                	found_ids.add(item['image_id'])
    	missing = wanted_ids - found_ids
    	if len(missing) == 0:
        	print ('GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids)))
    	else:
        	print ('GPU {:d}: missing {:d}/{:d}'.format(gpu_id, len(missing), len(image_ids)))
    
    	# Deal with the images remained:
    	if len(missing) > 0:
            with open(outfile, 'ab') as tsvfile:
                writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
				# One frame - one row   
            	for im_file, image_id in image_ids:
                	if image_id in missing:
						print('Start to extract frame_%s for video[%s]!' % (image_id, im_file.spilt('/')[-1]))
						writer.writerow(get_detections_from_im(net, im_file, image_id))
 		count += 1
		print('########## GPU%d: [%d/%d] ##########', gpu_id, count, total)                 
                                                                          
     
if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN
    caffe.init_log()
    # caffe.log('Using devices %s' % str(gpus))


	
    gpus = [0, 1, 2, 3, 4, 5]
    image_ids = load_frame_ids2()
    image_keys = list(image_ids.keys())

    # Use 6 gpus to work simultaneously

    import cPickle
    keys = cPickle.load(open('videos_extracted_cPickle.pkl', 'rb'))
    num_videos = len(keys)

    key0 = keys[0: int(1*num_videos / 6)]
    dict0 = dict([(key, image_ids[key] for key in key0 if key in image_keys)])

    # dict0 = {key: lst for key, lst in image_ids if key in key0}

    key1 = keys[int(1*num_videos / 6) : int(2*num_videos / 6)]
    dict1 = {key: lst for key, lst in image_ids if key in key1}

    key2 = keys[int(2*num_videos / 6) : int(3*num_videos / 6)]
    dict2 = {key: lst for key, lst in image_ids if key in key2}

    key3 = keys[int(3*num_videos / 6) : int(4*num_videos / 6)]
    dict3 = {key: lst for key, lst in image_ids if key in key3}

    key4 = keys[int(4*num_videos / 6) : int(5*num_videos / 6)]
    dict4 = {key: lst for key, lst in image_ids if key in key4}

    key5 = keys[int(5*num_videos / 6) : ]
    dict5 = {key: lst for key, lst in image_ids if key in key5}

    dicts = [dict0, dict1, dict2, dict3, dict4, dict5]

    procs = []
    out_file = '/home/chenxh/YSJ/video_tsv/'
    # Use every GPU to deal with its own ids.
    for i in gpus:	
        p = Process(target=generate_tsv, args=(i, args.prototxt, args.caffemodel, dicts[i], args.outfile))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()