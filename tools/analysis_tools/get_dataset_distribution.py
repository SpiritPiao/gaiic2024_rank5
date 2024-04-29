from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

train_annot_path = '/root/workspace/data/GAIIC2024/train.json'
val_annot_path = '/root/workspace/data/GAIIC2024/val.json'

# train_annot_path = '/root/workspace/data/GAIIC2024/merged_coco_new.json'
val_annot_path = '/root/workspace/data/GAIIC2024/val.json'
train_coco = COCO(train_annot_path) # 加载训练集的注释
val_coco = COCO(val_annot_path) # 加载验证集的注释

# 函数遍历一个人的所有数据库并逐行返回相关数据
def get_meta(coco):
    # print(coco.cats)
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # 图像的基本参数
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        # 检索当前图像中所有人的元数据
        anns = coco.loadAnns(ann_ids)

        yield [img_id, img_file_name, w, h, anns]


def statistic(train_coco):
    cats = train_coco.cats
    # 迭代图像
    category_ids = []
    areas = []
    for img_id, img_fname, w, h, meta in get_meta(train_coco):
        # 遍历图像的所有注释
        for m in meta:
            # print(m)
            # m是字典
            area = m['area']
            bbox = m['bbox']
            category_id = cats[m['category_id']]["name"]
            # print(category_id)
            category_ids.append(category_id)
            areas.append(area)


    df = pd.DataFrame({
        'category': category_ids,
        'areas': areas,
    })

    all_cnt = len(df)
    cat_cnt = df.groupby('category').count()
    x = cat_cnt.index.values
    y = [i[0] / all_cnt for i in cat_cnt.values]
    plt.title("categories")
    plt.bar(x, y)
    plt.show()

    # cat_cnt = df.groupby('category').count()
    # print(cat_cnt.index.values, cat_cnt.values)
    # x = cat_cnt.index.values
    # y = [i[0] for i in cat_cnt.values]
    # plt.title("Category count")
    # plt.bar(x, y)
    df['areas'] = df['areas'].apply(lambda x: math.sqrt(x))
    df.hist('areas')
    plt.show()
    
    
    df.boxplot(by='category')
    # print(df.groupby('category').describe())
    # plt.title("Areas by categories")
    plt.show()
    
statistic(train_coco)
# statistic(val_coco)
