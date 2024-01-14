# Shree KRISHNAya Namaha
# Warps frame using pose info
# Author: Nagabhushan S N
# Last Modified: 27/09/2021

import datetime
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional
import os
import numpy
import skimage.io


class Warper:
    def __init__(self, resolution: tuple = None):
        self.resolution = resolution
        return

    def forward_warp(self, frame1: numpy.ndarray, mask1: Optional[numpy.ndarray], depth1: numpy.ndarray,
                     transformation1: numpy.ndarray, transformation2: numpy.ndarray, intrinsic1: numpy.ndarray,
                     intrinsic2: Optional[numpy.ndarray]) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
                                                                   numpy.ndarray]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        :param frame1: (h, w, 3) uint8 numpy array
        :param mask1: (h, w) bool numpy array. Wherever mask1 is False, those pixels are ignored while warping. Optional
        :param depth1: (h, w) float numpy array.
        :param transformation1: (4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (3, 3) camera intrinsic matrix
        :param intrinsic2: (3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[:2] == self.resolution
        h, w = frame1.shape[:2]
        if mask1 is None:
            mask1 = numpy.ones(shape=(h, w), dtype=bool)
        if intrinsic2 is None:
            intrinsic2 = numpy.copy(intrinsic1)
        assert frame1.shape == (h, w, 3)
        assert mask1.shape == (h, w)
        assert depth1.shape == (h, w)
        assert transformation1.shape == (4, 4)
        assert transformation2.shape == (4, 4)
        assert intrinsic1.shape == (3, 3)
        assert intrinsic2.shape == (3, 3)

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                        intrinsic2)
        trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, 2, 0]

        grid = self.create_grid(h, w)
        flow12 = trans_coordinates - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
        warped_depth2 = self.bilinear_splatting(trans_depth1[:, :, None], mask1, trans_depth1, flow12, None,
                                                is_image=False)[0][:, :, 0]
        return warped_frame2, mask2, warped_depth2, flow12

    def compute_transformed_points(self, depth1: numpy.ndarray, transformation1: numpy.ndarray,
                                   transformation2: numpy.ndarray, intrinsic1: numpy.ndarray,
                                   intrinsic2: Optional[numpy.ndarray]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape == self.resolution
        h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = numpy.copy(intrinsic1)
        transformation = numpy.matmul(transformation2, numpy.linalg.inv(transformation1))

        y1d = numpy.array(range(h))
        x1d = numpy.array(range(w))
        x2d, y2d = numpy.meshgrid(x1d, y1d)
        ones_2d = numpy.ones(shape=(h, w))
        ones_4d = ones_2d[:, :, None, None]
        pos_vectors_homo = numpy.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]

        intrinsic1_inv = numpy.linalg.inv(intrinsic1)
        intrinsic1_inv_4d = intrinsic1_inv[None, None]
        intrinsic2_4d = intrinsic2[None, None]
        depth_4d = depth1[:, :, None, None]
        trans_4d = transformation[None, None]

        unnormalized_pos = numpy.matmul(intrinsic1_inv_4d, pos_vectors_homo)
        world_points = depth_4d * unnormalized_pos
        world_points_homo = numpy.concatenate([world_points, ones_4d], axis=2)
        trans_world_homo = numpy.matmul(trans_4d, world_points_homo)
        trans_world = trans_world_homo[:, :, :3]
        trans_norm_points = numpy.matmul(intrinsic2_4d, trans_world)
        return trans_norm_points

    def bilinear_splatting(self, frame1: numpy.ndarray, mask1: Optional[numpy.ndarray], depth1: numpy.ndarray,
                           flow12: numpy.ndarray, flow12_mask: Optional[numpy.ndarray], is_image: bool = False) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Using inverse bilinear interpolation based splatting
        :param frame1: (h, w, c)
        :param mask1: (h, w): True if known and False if unknown. Optional
        :param depth1: (h, w)
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame2: (h, w, c)
                 mask2: (h, w): True if known and False if unknown
        """
        if self.resolution is not None:
            assert frame1.shape[:2] == self.resolution
        h, w, c = frame1.shape
        if mask1 is None:
            mask1 = numpy.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = numpy.ones(shape=(h, w), dtype=bool)
        grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
        trans_pos_offset[:, :, 0] = numpy.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, 1] = numpy.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, 0] = numpy.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = numpy.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = numpy.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = numpy.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        sat_depth1 = numpy.clip(depth1, a_min=0, a_max=1000)
        log_depth1 = numpy.log(1 + sat_depth1)
        depth_weights = numpy.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
        weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
        weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
        weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        warped_image = numpy.zeros(shape=(h + 2, w + 2, c), dtype=numpy.float64)
        warped_weights = numpy.zeros(shape=(h + 2, w + 2), dtype=numpy.float64)

        numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
        numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
        numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
        numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)

        numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
        numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
        numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
        numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

        cropped_warped_image = warped_image[1:-1, 1:-1]
        cropped_weights = warped_weights[1:-1, 1:-1]

        mask = cropped_weights > 0
        with numpy.errstate(invalid='ignore'):
            warped_frame2 = numpy.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)

        if is_image:
            assert numpy.min(warped_frame2) >= 0
            assert numpy.max(warped_frame2) <= 256
            clipped_image = numpy.clip(warped_frame2, a_min=0, a_max=255)
            warped_frame2 = numpy.round(clipped_image).astype('uint8')
        return warped_frame2, mask

    def bilinear_interpolation(self, frame2: numpy.ndarray, mask2: Optional[numpy.ndarray], flow12: numpy.ndarray,
                               flow12_mask: Optional[numpy.ndarray], is_image: bool = False) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Using bilinear interpolation
        :param frame2: (h, w, c)
        :param mask2: (h, w): True if known and False if unknown. Optional
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame1: (h, w, c)
                 mask1: (h, w): True if known and False if unknown
        """
        if self.resolution is not None:
            assert frame2.shape[:2] == self.resolution
        h, w, c = frame2.shape
        if mask2 is None:
            mask2 = numpy.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = numpy.ones(shape=(h, w), dtype=bool)
        grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
        trans_pos_offset[:, :, 0] = numpy.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, 1] = numpy.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, 0] = numpy.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = numpy.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = numpy.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = numpy.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        weight_nw = prox_weight_nw * flow12_mask
        weight_sw = prox_weight_sw * flow12_mask
        weight_ne = prox_weight_ne * flow12_mask
        weight_se = prox_weight_se * flow12_mask

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        frame2_offset = numpy.pad(frame2, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        mask2_offset = numpy.pad(mask2, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)

        f2_nw = frame2_offset[trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]]
        f2_sw = frame2_offset[trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]]
        f2_ne = frame2_offset[trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]]
        f2_se = frame2_offset[trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]]

        m2_nw = mask2_offset[trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]]
        m2_sw = mask2_offset[trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]]
        m2_ne = mask2_offset[trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]]
        m2_se = mask2_offset[trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]]

        m2_nw_3d = m2_nw[:, :, None]
        m2_sw_3d = m2_sw[:, :, None]
        m2_ne_3d = m2_ne[:, :, None]
        m2_se_3d = m2_se[:, :, None]

        nr = weight_nw_3d * f2_nw * m2_nw_3d + weight_sw_3d * f2_sw * m2_sw_3d + \
             weight_ne_3d * f2_ne * m2_ne_3d + weight_se_3d * f2_se * m2_se_3d
        dr = weight_nw_3d * m2_nw_3d + weight_sw_3d * m2_sw_3d + weight_ne_3d * m2_ne_3d + weight_se_3d * m2_se_3d
        warped_frame1 = numpy.where(dr > 0, nr / dr, 0)
        mask1 = dr[:, :, 0] > 0

        if is_image:
            assert numpy.min(warped_frame1) >= 0
            assert numpy.max(warped_frame1) <= 256
            clipped_image = numpy.clip(warped_frame1, a_min=0, a_max=255)
            warped_frame1 = numpy.round(clipped_image).astype('uint8')
        return warped_frame1, mask1

    @staticmethod
    def create_grid(h, w):
        x_1d = numpy.arange(0, w)[None]
        y_1d = numpy.arange(0, h)[:, None]
        x_2d = numpy.repeat(x_1d, repeats=h, axis=0)
        y_2d = numpy.repeat(y_1d, repeats=w, axis=1)
        grid = numpy.stack([x_2d, y_2d], axis=2)
        return grid

    @staticmethod
    def read_image(path: Path) -> numpy.ndarray:
        if path.suffix in ['.jpg', '.png', '.bmp']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    @staticmethod
    def read_depth(path: Path) -> numpy.ndarray:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        elif path.suffix == '.exr':
            import Imath
            import OpenEXR

            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth

    @staticmethod
    def camera_intrinsic_transform(capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(3)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics


def demo1():
    frame1_path = Path('third-parties/Pose-Warping/Data/frame1.png')
    frame2_path = Path('third-parties/Pose-Warping/Data/frame2.png')
    depth1_path = Path('third-parties/Pose-Warping/Data/depth1.npy')
    transformation1 = numpy.array([
        4.067366123199462891e-01, 9.135454893112182617e-01, 2.251522164442576468e-05, -1.571802258491516113e+00,
        -7.961163669824600220e-02, 3.546993434429168701e-02, -9.961947202682495117e-01, 1.842712044715881348e+00,
        -9.100699424743652344e-01, 4.051870703697204590e-01, 8.715576678514480591e-02, -2.255212306976318359e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00
    ]).reshape(4, 4)
    transformation2 = numpy.array([
        4.067366123199462891e-01, 9.135454893112182617e-01, 2.251522164442576468e-05, -1.616834521293640137e+00,
        -7.961163669824600220e-02, 3.546993434429168701e-02, -9.961947202682495117e-01, 1.848096847534179688e+00,
        -9.100699424743652344e-01, 4.051870703697204590e-01, 8.715576678514480591e-02, -2.275809526443481445e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00
    ]).reshape(4, 4)

    warper = Warper()
    frame1 = warper.read_image(frame1_path)
    frame2 = warper.read_image(frame2_path)
    depth1 = warper.read_depth(depth1_path)
    intrinsic = warper.camera_intrinsic_transform()

    warped_frame2 = warper.forward_warp(frame1, None, depth1, transformation1, transformation2, intrinsic, None)[0]
    skimage.io.imsave('frame1.png', frame1)
    skimage.io.imsave('frame2.png', frame2)
    skimage.io.imsave('frame2_warped.png', warped_frame2)
    return

def demo2():
    kk = 1
    in_path = 'data/00scene_text/text40'
    save_path = 'results_test_on_warping/text40'
    os.makedirs(save_path, exist_ok=True)

    frame1_path = Path(os.path.join(in_path, 'rgbs/00000.png'))
    frame2_path = Path(os.path.join(in_path, 'DIBR_gt/warped/%05d.png'%kk))
    depth1_path = Path(os.path.join(in_path, 'depth/00000.npy'))

    warper = Warper()
    frame1 = warper.read_image(frame1_path)
    frame2 = warper.read_image(frame2_path)
    depth1 = warper.read_depth(depth1_path)
    # intrinsic = warper.camera_intrinsic_transform()
    transformation1 = numpy.linalg.inv(numpy.load(os.path.join(in_path, 'cam/00000_pose.npy')))
    transformation2 = numpy.linalg.inv(numpy.load(os.path.join(in_path, 'cam/%05d_pose.npy'%kk)))
    intrinsic = numpy.load(os.path.join(in_path, 'cam/intrinsic.npy'))

    warped_frame2, mask2, warped_depth2, flow12 = warper.forward_warp(frame1, None, depth1, transformation1, transformation2, intrinsic, None)
    for i in range(3):
        warped_frame2[:,:,i] = numpy.array(warped_frame2[:,:,i]*mask2+255*(1-mask2))
    skimage.io.imsave(os.path.join(save_path, '_frame0.png'), frame1)
    skimage.io.imsave(os.path.join(save_path, '_frame%05d.png'%kk), frame2)
    skimage.io.imsave(os.path.join(save_path, '_mask%05d.png'%kk), mask2)
    skimage.io.imsave(os.path.join(save_path, '_frame%05d_warped.png'%kk), warped_frame2)
    skimage.io.imsave(os.path.join(save_path, '_depth%05d_warped.png'%kk), warped_depth2)
    return


def main():
    # demo1()
    demo2()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

