from copy import deepcopy
from pathlib import Path
import pickle, os, glob, imageio, tqdm
from matplotlib import pyplot as plt
import cv2, numpy as np

def parse(path, image_roots, save_root=None, show=False, func=None):
    if save_root:
        os.makedirs(save_root, exist_ok=True)    

    with open(path, "rb") as f:
        image_boxes = pickle.load(f)

    all_w_list = []
    all_h_list = []
    all_pred_list = []
    all_label_list = []
    for instances, image_path in tqdm.tqdm(image_boxes):
        images = []
        if save_root:
            for image_root in image_roots:
                image_full_path = os.path.join(image_root, image_path)
                image = imageio.imread_v2(image_full_path)
                images.append(image)
        
        save_path = None
        
        if save_root:
            save_path = os.path.join(save_root, image_path)
        w_list, h_list, pred_ids, label_ids = draw_boxes(images, instances, save_path, func)
        all_w_list += w_list
        all_h_list += h_list
        all_pred_list += pred_ids
        all_label_list += label_ids


    all_w_list = all_w_list[::4]
    all_h_list = all_h_list[::4]
    all_label_list = all_label_list[::4]
    all_pred_list = all_pred_list[::4]
    
    
    sc = plt.scatter(all_w_list, all_h_list, s=2, c=all_label_list, alpha=0.5, cmap='Set1')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("w")
    plt.ylabel("h")
    
    plt.colorbar(sc, ticks=[0,1,2,3,4])    
    plt.show()
    plt.close()

    sc = plt.scatter(all_w_list, all_h_list, s=2, c=all_pred_list, alpha=0.5, cmap='Set1')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("w")
    plt.ylabel("h")
    
    plt.colorbar(sc, ticks=[0,1,2,3,4])    
    plt.show()
    plt.close()


import math
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
          pos=(0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    x = max(0, x)
    y = math.ceil(max(0, y + text_h + font_scale - 1))
    # print(x, y)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness)

    return text_size


def draw_boxes(images, instances, save_path, func):
    w_list, h_list = [], []
    pred_ids, label_ids = [], []
    ori_images = deepcopy(images)
    for box, label_id, pred_id in instances:
        # print(boxes, label_id, pred_id)
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if func is not None:
            func(ori_images, box, pred_id, label_id, Path(save_path).name)

        line_type = cv2.LINE_4
        box_color = (255, 0, 0)
        texts = []
        if label_id == -1:
            # FP Blue xuxian
            box_color = (0, 0, 255)
            texts = [
                f"{pred_id}",
            ]
            
            # line_type = cv2.LINE_6
            pass
        elif pred_id == -1:
            # FN Red
            texts = [
                f"{label_id}"
            ]
            pass
        else:
            # FG
            if label_id == pred_id:
                # TP Green
                box_color = (0, 255, 0)
                texts = [
                    f"{pred_id}",
                ]
                
                pass
            else:
                texts = [
                    f"{pred_id}",
                    f"{label_id}"
                ]
                
                # Wrong Red
                pass
                w_list.append(w)
                h_list.append(h)
                pred_ids.append(pred_id)
                label_ids.append(label_id)
            pass
        
        outputs = []
        for image in images:
            img_drawed = cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 1, line_type)
            outputs.append(img_drawed)
            if len(texts) > 0:
                # fontFace = cv.FONT_HERSHEY_TRIPLEX
                fontScale = 0.6
                fontcolor = (255, 255, 255) # BGR
                thickness = 1 
                lineType = cv2.LINE_AA
                draw_text(img_drawed, "/".join(texts), pos=(x1, y1), font_scale=fontScale, text_color=fontcolor, font_thickness=thickness, text_color_bg=(0, 0, 0))
        if len(outputs) > 0:
            outputs = np.concatenate(outputs, axis=0)
        


        if save_path is not None:
            imageio.imwrite(save_path, outputs)

    return w_list, h_list, pred_ids, label_ids


def crop_bbox(images, box, pred_id, label_id, file_name):
    output_dir = "/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/cropped_images"
    x1, y1, x2, y2 = box
    scale = 1.5

    w, h = x2 - x1, y2 - y1
    w *= scale
    h *= scale

    w, h = int(w), int(h)
    x1, x2 = int(x1), int(x2)

    center = [(x1 + x2) // 2, (y1 + y2) // 2]

    new_x1 = int(max(center[0] - w // 2, 0))
    new_y1 = int(max(center[1] - h // 2, 0))

    H, W = images[0].shape[:2]

    new_x2 = int(min(new_x1 + w, W))
    new_y2 = int(min(new_y1 + h, H))

    # print(new_x1, new_y1, new_x2, new_y2, images[0].shape)

    cropped1 = images[0][new_y1:new_y2, new_x1:new_x2]
    # crop_images1.append(cropped)

    cropped2 = images[1][new_y1:new_y2, new_x1:new_x2]
    # crop_images2.append(cropped)

    rgb_dir = os.path.join(output_dir, "rgb")
    tir_dir = os.path.join(output_dir, "tir")

    stem = Path(file_name).stem

    img_name = f"{stem}_{new_x1}_{new_y1}_{new_x2}_{new_y2}_{pred_id}_{label_id}.jpg"

    save_path1 = os.path.join(rgb_dir, f"{label_id:02}")
    save_path2 = os.path.join(tir_dir, f"{label_id:02}")

    os.makedirs(save_path1, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)

    save_path1 = os.path.join(save_path1, img_name)
    save_path2 = os.path.join(save_path2, img_name)

    imageio.imwrite(save_path1, cropped1)
    imageio.imwrite(save_path2, cropped2)


if __name__ == "__main__":
    # parse("/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/image_boxes.pk",
    #       ["/root/workspace/data/GAIIC2024/val/rgb", "/root/workspace/data/GAIIC2024/val/tir"],
    #       "/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/outputs")

    # parse("/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/image_boxes.pk",
    #       ["/root/workspace/data/GAIIC2024/val/rgb", "/root/workspace/data/GAIIC2024/val/tir"],
    #       "/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_pianyi/outputs/val", func=crop_bbox)
    
    parse("/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_albu_train/image_boxes.pk",
        ["/root/workspace/data/GAIIC2024/train_more/rgb", "/root/workspace/data/GAIIC2024/train_more/tir"],
        "/root/workspace/data/dual_mmdetection/mmdetection/analysis_results/co_dino_5scale_swin_l_16xb1_16e_gaiic_dual_stream_o365_yang_more_data_albu_train/outputs/train", func=crop_bbox)