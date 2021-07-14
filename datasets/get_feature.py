import numpy
import base64
import csv
import sys

csv.field_size_limit(sys.maxsize)

def get_regional_features(self, video_name):
    region_feats = np.zeros((32, 20, 2048), dtype=np.float32)
    region_boxes = np.zeros((32, 20, 4), dtype=np.float32)

    tsv_path = os.path.join(self.tsv_dir, video_name + '.tsv')
    with open(tsv_path, "r+") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t', fieldnames=FIELDNAMES)
        # sortedlist = sorted(reader, key=lambda x: x[5], reverse=True)
        for i, item in enumerate(reader):

            feats = np.frombuffer(base64.decodebytes(bytes(item['features'], encoding='utf8')),
                                  dtype=np.float32).reshape((8, -1))  # 8 × 2048
            boxes = np.frombuffer(base64.decodebytes(bytes(item['boxes'], encoding='utf8')), dtype=np.float32).reshape(
                (8, -1))  # 8 × 4

            frame_id = int(item['image_id'].split('__')[1])  # id: 0 ~ 15

            region_feats[i] = feats
            region_boxes[i] = boxes

    return (region_feats, region_boxes)