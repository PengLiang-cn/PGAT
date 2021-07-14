from PIL import Image
import os

gif_dir = 'gifs/'
i = 0

import cPickle
gifs = cPickle.load(open('videos_extracted_cPickle.pkl', 'rb'))
num_gifs = len(gifs)
print('Total number of gif files: ', num_gifs)

for gif in gifs:
    gif_path = os.path.join(gif_dir, gif + '.gif')
    im = Image.open(gif_path)

    frame_dir = gif_path[5: -4]
    if frame_dir[:8] == '._tumblr':
        continue

    # Create dir for all the frames of this gif file:
    dir_path = os.path.join('gif_frames', frame_dir)
    if os.path.exists(dir_path):
        continue
    else:
        os.mkdir(dir_path)

    try:
        while True:
            current = im.tell()
            img = im.convert('RGB')
            img.save(dir_path + '/' + str(current) + '.jpg')
            im.seek(current + 1)
    except EOFError:
        pass
    i += 1
    print('Save %s success!' % frame_dir)
    print('[%d/%d] ', i, num_gifs)