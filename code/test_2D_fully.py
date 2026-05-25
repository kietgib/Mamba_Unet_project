# import argparse
# import os
# import shutil

# import h5py
# import nibabel as nib
# import numpy as np
# import SimpleITK as sitk
# import torch
# from medpy import metric
# from scipy.ndimage import zoom
# from scipy.ndimage.interpolation import zoom
# from tqdm import tqdm

# # from networks.efficientunet import UNet
# from networks.net_factory import net_factory

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/ACDC', help='Name of Experiment')
# parser.add_argument('--exp', type=str,
#                     default='ACDC/Fully_Supervised', help='experiment_name')
# parser.add_argument('--model', type=str,
#                     default='unet', help='model_name')
# parser.add_argument('--num_classes', type=int,  default=4,
#                     help='output channel of network')
# parser.add_argument('--labeled_num', type=int, default=3,
#                     help='labeled data')


# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     dice = metric.binary.dc(pred, gt)
#     # asd = metric.binary.asd(pred, gt)
#     # hd95 = metric.binary.hd95(pred, gt)
#     return dice
#     # , hd95
#     # , asd


# def test_single_volume(case, net, test_save_path, FLAGS):
#     h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
#     image = h5f['image'][:]
#     label = h5f['label'][:]
#     prediction = np.zeros_like(label)
#     for ind in range(image.shape[0]):
#         slice = image[ind, :, :]
#         x, y = slice.shape[0], slice.shape[1]
#         slice = zoom(slice, (256 / x, 256 / y), order=0)
#         input = torch.from_numpy(slice).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             if FLAGS.model == "unet_urds":
#                 out_main, _, _, _ = net(input)
#             else:
#                 out_main = net(input)
#             out = torch.argmax(torch.softmax(
#                 out_main, dim=1), dim=1).squeeze(0)




#             out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
#             prediction[ind] = pred

#     first_metric = calculate_metric_percase(prediction == 1, label == 1)
#     second_metric = calculate_metric_percase(prediction == 2, label == 2)
#     third_metric = calculate_metric_percase(prediction == 3, label == 3)

#     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#     img_itk.SetSpacing((1, 1, 10))
#     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#     prd_itk.SetSpacing((1, 1, 10))
#     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#     lab_itk.SetSpacing((1, 1, 10))
#     sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
#     sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
#     sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
#     return first_metric, second_metric, third_metric


# def Inference(FLAGS):
#     with open(FLAGS.root_path + '/test.list', 'r') as f:
#         image_list = f.readlines()
#     image_list = sorted([item.replace('\n', '').split(".")[0]
#                          for item in image_list])
#     # snapshot_path = "../model/{}_{}_labeled/{}".format(
#     snapshot_path = "../model/{}_{}/{}".format(        
#         FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
#     # test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
#     test_save_path = "../model/{}_{}/{}_predictions/".format(        
#         FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
#     if os.path.exists(test_save_path):
#         shutil.rmtree(test_save_path)
#     os.makedirs(test_save_path)
#     # net = net_factory(net_type=FLAGS.model, in_chns=1,
#     #                   class_num=FLAGS.num_classes)
#     # new
#     from config import get_config
#     from networks.vision_mamba import MambaUnet as VIM_seg
#     config = get_config(FLAGS)
#     net = VIM_seg(config, img_size=[224, 224], num_classes=FLAGS.num_classes).cuda()
#     # new

#     save_mode_path = os.path.join(
#         snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
#     print(save_mode_path)
#     net.load_state_dict(torch.load(save_mode_path))
#     print("init weight from {}".format(save_mode_path))
#     net.eval()

#     first_total = 0.0
#     second_total = 0.0
#     third_total = 0.0
#     for case in tqdm(image_list):
#         first_metric, second_metric, third_metric = test_single_volume(
#             case, net, test_save_path, FLAGS)
#         first_total += np.asarray(first_metric)
#         second_total += np.asarray(second_metric)
#         third_total += np.asarray(third_metric)
#     avg_metric = [first_total / len(image_list), second_total /
#                   len(image_list), third_total / len(image_list)]
#     return avg_metric


# if __name__ == '__main__':
#     FLAGS = parser.parse_args()
#     metric = Inference(FLAGS)
#     # print(metric)
#     # print((metric[0]+metric[1]+metric[2])/3)

import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from yacs.config import CfgNode as CN

# Import hàm lấy cấu hình hệ thống và Class mô hình gốc
from config import get_config
from networks.vision_mamba import MambaUnet as VIM_seg

# --- CẤU HÌNH CÁC THAM SỐ ĐẦU VÀO VÀ ĐỂ BYPASS HÀM UPDATE_CONFIG ---
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', 
                    help='Đường dẫn gốc đến thư mục chứa dữ liệu ACDC')
parser.add_argument('--num_classes', type=int, default=4, 
                    help='Số phân lớp đầu ra (gồm cả nền)')
parser.add_argument('--model', type=str, default='mambaunet', 
                    help='Tên mô hình')

# Khai báo thêm các tham số hệ thống bắt buộc phải có để get_config không bị lỗi
parser.add_argument('--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", 
                    help='Đường dẫn đến file config yaml')
parser.add_argument("--opts", default=None, nargs='+', help="Modify config options")
parser.add_argument('--zip', action='store_true', help='use zipped dataset')
parser.add_argument('--cache-mode', type=str, default='part')
parser.add_argument('--amp-opt-level', type=str, default='O1')

# Bổ sung các tham số liên quan đến huấn luyện để vượt qua hàm update_config()
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--max_iterations', type=int, default=10000)
parser.add_argument('--patch_size', type=list, default=[224, 224])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--labeled_num', type=int, default=140)
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing")

def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "acc": 1.0,
            "pre": 1.0,
            "sen": 1.0,
            "spe": 1.0,
            "hd95": None,
            "asd": None,
        }
    if pred_sum == 0 or gt_sum == 0:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "acc": 0.0,
            "pre": 0.0,
            "sen": 0.0,
            "spe": 0.0,
            "hd95": None,
            "asd": None,
        }

    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0

    try:
        hd95 = metric.binary.hd95(pred, gt)
    except Exception:
        hd95 = None

    try:
        asd = metric.binary.asd(pred, gt)
    except Exception:
        asd = None

    return {
        "dice": metric.binary.dc(pred, gt),
        "iou": metric.binary.jc(pred, gt),
        "acc": acc,
        "pre": metric.binary.precision(pred, gt),
        "sen": metric.binary.recall(pred, gt),
        "spe": metric.binary.specificity(pred, gt),
        "hd95": hd95,
        "asd": asd,
    }

def test_single_volume(case, net, test_save_path, FLAGS):
    # Đọc dữ liệu từ file cắt lớp hình khối .h5
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    
    # Dự đoán tuần tự trên từng lát cắt 2D (Slice-by-Slice)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        
        # Đưa tầm nhìn ảnh về kích thước chuẩn 256x256
        slice_resized = zoom(slice, (256 / x, 256 / y), order=1) 
        input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
        
        with torch.no_grad():
            out_main = net(input_tensor)
            out_argmax = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out_numpy = out_argmax.cpu().detach().numpy()
            
            # Trả ảnh dự đoán về lại kích thước gốc của ca bệnh
            pred_slice = zoom(out_numpy, (x / 256, y / 256), order=0) 
            prediction[ind] = pred_slice

    # Tính toán chỉ số Dice Score riêng cho từng bộ phận cấu trúc tim
    first_metric = calculate_metric_percase(prediction == 1, label == 1)  # RV - Tâm thất phải
    second_metric = calculate_metric_percase(prediction == 2, label == 2) # MYO - Cơ tim
    third_metric = calculate_metric_percase(prediction == 3, label == 3)  # LV - Tâm thất trái

    # Xuất định dạng ảnh y tế .nii.gz để kiểm tra trực quan trên ITK-SNAP
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    
    return first_metric, second_metric, third_metric

def Inference(FLAGS):
    # Đọc danh sách các ca bệnh trong tập kiểm thử
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    
    # Thiết lập thư mục đầu ra chứa kết quả dự đoán test
    test_save_path = "../model/VIM_140_labeled/mambaunet_predictions/"
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    print("-> Đang nạp và đồng bộ hóa cấu hình hệ thống (Swin + Mamba)...")
    # Tải cấu hình nền tảng từ file YAML hệ thống
    config = get_config(FLAGS)
    
    
    # Sử dụng tính năng mở khóa cấu hình động để tự động vá các thuộc tính lai bị thiếu
    config.defrost()
    
    if not hasattr(config.MODEL, 'VSSM'):
        config.MODEL.VSSM = CN()
    config.MODEL.VSSM.DEPTHS = [2, 2, 2, 2]         
    config.MODEL.VSSM.DEPTHS_DECODER = [1, 2, 2, 2] 
    config.MODEL.VSSM.DROP_PATH_RATE = 0.2
    config.MODEL.VSSM.PATCH_SIZE = 4
    config.MODEL.VSSM.IN_CHANS = 3
    config.MODEL.VSSM.EMBED_DIM = 96
    config.MODEL.VSSM.GMDIM = 0
    config.MODEL.VSSM.SSM_CFG = CN()
    
    # Khóa cấu hình an toàn sau khi hoàn tất quá trình đồng bộ bổ sung
    config.freeze()
    
    # Khởi tạo mô hình Mamba-UNet trực tiếp từ Class gốc
    net = VIM_seg(config, img_size=[224, 224], num_classes=FLAGS.num_classes).cuda()
    
    # Trỏ thẳng đích danh vào file trọng số xịn nhất được tìm thấy trong thư mục huấn luyện
    save_mode_path = "/home/nhutnt/data/hai/Mamba-UNet/model/ACDC/VIM_140_labeled/mambaunet/mambaunet_best_model.pth"
    
    print("-> Đang nạp trọng số từ: {}".format(save_mode_path))
    net.load_state_dict(torch.load(save_mode_path))
    net.eval()

    metric_keys = ["dice", "iou", "acc", "pre", "sen", "spe", "hd95", "asd"]

    def init_metric_sum():
        return {k: 0.0 for k in metric_keys}

    def init_metric_count():
        return {k: 0 for k in metric_keys}

    first_total = init_metric_sum()
    second_total = init_metric_sum()
    third_total = init_metric_sum()

    first_count = init_metric_count()
    second_count = init_metric_count()
    third_count = init_metric_count()
    
    print("-> Bắt đầu quá trình Test mô hình Mamba-UNet...")
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)

        for k in metric_keys:
            if first_metric[k] is not None:
                first_total[k] += float(first_metric[k])
                first_count[k] += 1
            if second_metric[k] is not None:
                second_total[k] += float(second_metric[k])
                second_count[k] += 1
            if third_metric[k] is not None:
                third_total[k] += float(third_metric[k])
                third_count[k] += 1
        
    # Tính toán trung bình kết quả đánh giá cuối cùng
    def safe_avg(total_map, count_map):
        avg = {}
        for k in metric_keys:
            if count_map[k] > 0:
                avg[k] = total_map[k] / count_map[k]
            else:
                avg[k] = 0.0
        return avg

    avg_rv = safe_avg(first_total, first_count)
    avg_myo = safe_avg(second_total, second_count)
    avg_lv = safe_avg(third_total, third_count)
    avg_mean = {k: (avg_rv[k] + avg_myo[k] + avg_lv[k]) / 3.0 for k in metric_keys}
    
    print("\n" + "="*50)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("="*50)
    print("- Dice Tâm thất phải (RV - Class 1): {:.4f}".format(avg_rv["dice"]))
    print("- Dice Cơ tim         (MYO - Class 2): {:.4f}".format(avg_myo["dice"]))
    print("- Dice Tâm thất trái  (LV - Class 3): {:.4f}".format(avg_lv["dice"]))
    print("- IoU  Trung bình (Mean IoU): {:.4f}".format(avg_mean["iou"]))
    print("- ACC  Trung bình (Mean ACC): {:.4f}".format(avg_mean["acc"]))
    print("- Pre  Trung bình (Mean Pre): {:.4f}".format(avg_mean["pre"]))
    print("- Sen  Trung bình (Mean Sen): {:.4f}".format(avg_mean["sen"]))
    print("- Spe  Trung bình (Mean Spe): {:.4f}".format(avg_mean["spe"]))
    print("- HD95 Trung bình (Mean HD95): {:.4f}".format(avg_mean["hd95"]))
    print("- ASD  Trung bình (Mean ASD): {:.4f}".format(avg_mean["asd"]))
    print("-"*50)
    print(">>> DICE TRUNG BINH CHUNG (Mean Dice): {:.4f}".format(avg_mean["dice"]))
    print("="*50)
    print("-> Toàn bộ file kết quả .nii.gz đã được xuất thành công tại:\n   {}".format(test_save_path))
    
    return [avg_rv, avg_myo, avg_lv]

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)