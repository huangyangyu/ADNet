import os
import sys
import cv2
import math
import argparse
import numpy as np
from skimage import transform
from scipy import interpolate

# pytorch
import torch
import torch.nn as nn
from lib import utility

# onnx
import onnxruntime as rt


class GetCropMatrix():
    """
    from_shape -> transform_matrix
    """
    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def process(self, scale, center_w, center_h):
        if self.align_corners:
            to_w, to_h = self.image_size-1, self.image_size-1
        else:
            to_w, to_h = self.image_size, self.image_size

        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w/2.0, to_h/2.0])
        return matrix


class TransformPerspective():
    """
    image, matrix3x3 -> transformed_image
    """
    def __init__(self, image_size):
        self.image_size = image_size
        
    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)


class TransformPoints2D():
    """
    points (nx2), matrix (3x3) -> points (nx2)
    """
    def process(self, srcPoints, matrix):
        # nx3
        desPoints = np.concatenate([srcPoints, np.ones_like(srcPoints[:, [0]])], axis=1)
        desPoints = desPoints @ np.transpose(matrix)  # nx3
        desPoints = desPoints[:, :2] / desPoints[:, [2, 2]]
        return desPoints.astype(srcPoints.dtype)


class Alignment:
    def __init__(self, config_name, work_dir, model_path, dl_framework, device_ids):
        self.input_size = 256
        self.target_face_scale = 1.0
        self.dl_framework = dl_framework

        # model
        if self.dl_framework == "pytorch":
            # conf
            config = utility.get_config(config_name, work_dir)
            config.device_id = device_ids[0]
            # set environment
            utility.set_environment(config)
            config.init_instance()
            if config.logger is not None:
                config.logger.info("Loaded configure file %s: %s" % (config_name, config.id))
                config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))

            net = utility.get_net(config)
            if device_ids == [-1]:
                checkpoint = torch.load(model_path, map_location="cpu")
            else:
                checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint["net"])
            net.eval()
            self.alignment = net
        elif self.dl_framework == "onnx":
            self.alignment = rt.InferenceSession(model_path)
            self.input_name = self.alignment.get_inputs()[0].name
            #self.output_name = self.alignment.get_outputs()[-1].name
        else:
            assert False

        self.getCropMatrix = GetCropMatrix(image_size=self.input_size, target_face_scale=self.target_face_scale, align_corners=True)
        self.transformPerspective = TransformPerspective(image_size=self.input_size)
        self.transformPoints2D = TransformPoints2D()

    def norm_points(self, points, align_corners=False):
        if align_corners:
            # [0, SIZE-1] -> [-1, +1]
            return points / torch.tensor([self.input_size-1, self.input_size-1]).to(points).view(1, 1, 2) * 2 - 1
        else:
            # [-0.5, SIZE-0.5] -> [-1, +1]
            return (points * 2 + 1) / torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1

    def denorm_points(self, points, align_corners=False):
        if align_corners:
            # [-1, +1] -> [0, SIZE-1]
            return (points + 1) / 2 * torch.tensor([self.input_size-1, self.input_size-1]).to(points).view(1, 1, 2)
        else:
            # [-1, +1] -> [-0.5, SIZE-0.5]
            return ((points + 1) * torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1) / 2

    def preprocess(self, image, landmarks, scale, center_w, center_h):
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        input_tensor = self.transformPerspective.process(image, matrix)
        input_tensor = input_tensor[np.newaxis, :]
        input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.float().permute(0, 3, 1, 2)
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0
        return input_tensor, matrix

    def postprocess(self, srcPoints, coeff):
        #dstPoints = self.transformPoints2D.process(srcPoints, coeff)
        dstPoints = np.zeros(srcPoints.shape, dtype=np.float32)
        for i in range(srcPoints.shape[0]):
            dstPoints[i][0] = coeff[0][0] * srcPoints[i][0] + coeff[0][1] * srcPoints[i][1] + coeff[0][2]
            dstPoints[i][1] = coeff[1][0] * srcPoints[i][0] + coeff[1][1] * srcPoints[i][1] + coeff[1][2]
        return dstPoints

    def analyze(self, image, landmarks, scale, center_w, center_h):
        input_tensor, matrix = self.preprocess(image, landmarks, scale, center_w, center_h)

        if self.dl_framework == "pytorch":
            with torch.no_grad():
                output = self.alignment(input_tensor)
            landmarks = output[-1][0]
        elif self.dl_framework == "onnx":
            output = self.alignment.run([], {self.input_name: input_tensor.numpy()})
            landmarks = torch.from_numpy(output[-1])
        else:
            assert False

        landmarks = self.denorm_points(landmarks)
        landmarks = landmarks.data.cpu().numpy()[0]
        landmarks = self.postprocess(landmarks, np.linalg.inv(matrix))

        return landmarks


def L2(p1, p2):
    return np.linalg.norm(p1 - p2)


def NME(landmarks_gt, landmarks_pv):
    pts_num = landmarks_gt.shape[0]
    if pts_num == 29:
        left_index = 16
        right_index = 17
    elif pts_num == 68:
        left_index = 36
        right_index = 45
    elif pts_num == 98:
        left_index = 60
        right_index = 72
    
    nme = 0
    eye_span = L2(landmarks_gt[left_index], landmarks_gt[right_index])
    for i in range(pts_num):
        error = L2(landmarks_pv[i], landmarks_gt[i])
        nme += error / eye_span
    nme /= pts_num
    return nme


def evaluate(config_name, work_dir, model_path, metadata_path, image_dir, device_ids, mode):
    if model_path.endswith("onnx"):
        dl_framework = "onnx"
    else:
        dl_framework = "pytorch"
    alignment = Alignment(config_name, work_dir, model_path, dl_framework, device_ids)

    nme_sum = 0
    for k, line in enumerate(open(metadata_path)):
        item = line.strip().split("\t")
        image_name, landmarks_5pts, landmarks_gt, scale, center_w, center_h = item[:6]
        image_path = os.path.join(image_dir, image_name)
        landmarks_5pts = np.array(list(map(float, landmarks_5pts.split(","))), dtype=np.float32).reshape(5, 2)
        landmarks_gt = np.array(list(map(float, landmarks_gt.split(","))), dtype=np.float32).reshape(-1, 2)
        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        
        image = cv2.imread(image_path)
        landmarks_pv = alignment.analyze(image, landmarks_5pts, scale, center_w, center_h)
        
        # NME
        if mode == "nme":
            nme = NME(landmarks_gt, landmarks_pv)
            nme_sum += nme
            print("Current NME(%d): %f" % (k+1, (nme_sum / (k+1))))
        else:
            # visualization
            for i in range(landmarks_pv.shape[0]):
                x = int(landmarks_pv[i][0] + 0.5)
                y = int(landmarks_pv[i][1] + 0.5)
                cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
            cv2.imshow("demo", image)
            if cv2.waitKey(0) == 27:
                break
    if mode == "nme":
        print("Final NME: %f" % (nme_sum / k))
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--config_name", type=str, default="alignment", help="set configure file name")
    parser.add_argument("--work_dir", type=str, default="./", help="the directory of workspace")
    parser.add_argument("--model_path", type=str, default="./model/alignment/300W/train.onnx", help="the path of model")
    parser.add_argument("--metadata_path", type=str, default="./data/alignment/300W/test.tsv", help="the path of metadata")
    parser.add_argument("--image_dir", type=str, default=r"", help="the root directory of images")
    parser.add_argument("--device_ids", type=str, default="-1", help="set device ids, -1 means use cpu device, >= 0 means use gpu device")
    parser.add_argument("--mode", type=str, default="nme", help="set the evaluate mode: nme and visualization")
    args = parser.parse_args()

    device_ids = list(map(int, args.device_ids.split(",")))
    evaluate(config_name=args.config_name,
         work_dir=args.work_dir,
         model_path=args.model_path,
         metadata_path=args.metadata_path,
         image_dir=args.image_dir,
         device_ids=device_ids,
         mode=args.mode)
