import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image


class Dataset(object):
    SUBSET_OPTIONS = ['train', 'val', 'test']
    VOID_LABEL = 255

    def __init__(self, root, subset='val', sequences='all'):
        """
        Class to read the dataset
        :param root: Path to the dataset folder that contains JPEGImages, Annotations, etc. folders.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')

        self.task = 'semi-supervised'
        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, 'JPEGImages')
        annotations_folder = 'Annotations'
        self.mask_path = os.path.join(self.root, annotations_folder)
        self.imagesets_path = os.path.join(self.root, 'ImageSets')

        self._check_directories()

        if sequences == 'all':
            with open(os.path.join(self.imagesets_path, f'{self.subset}.txt'), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            if len(masks) == 0:
                raise FileNotFoundError(f'Annotations for sequence {seq} not found.')
            self.sequences[seq]['masks'] = masks
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            filtered_images = []
            for img in images:
                ann = img.replace('jpg', 'png').replace('JPEGImages', 'Annotations')
                if ann not in masks:
                    print(ann)
                else:
                    filtered_images.append(img)
            self.sequences[seq]['images'] = filtered_images

            # images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            # if len(images) == 0:
            #     raise FileNotFoundError(f'Images for sequence {seq} not found.')
            # self.sequences[seq]['images'] = images
            # masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            # masks.extend([-1] * (len(images) - len(masks)))
            # self.sequences[seq]['masks'] = masks

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'Dataset not found in the specified directory')
        if not os.path.exists(os.path.join(self.imagesets_path, f'{self.subset}.txt')):
            raise FileNotFoundError(f'Subset sequences list for {self.subset} not found')
        if self.subset in ['train', 'val'] and not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'Annotations folder not found')

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape), dtype=np.uint8)
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape), dtype=masks.dtype)
            tmp = tmp * np.arange(1, num_objects + 1, dtype=tmp.dtype)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq

