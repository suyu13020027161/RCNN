import json
from pathlib import Path

# 指定原始JSON文件路径
json_file_path = 'instances_val2017.json'


# 指定保存新JSON文件的目录
output_directory = Path('')
output_directory.mkdir(exist_ok=True)  # 如果输出目录不存在则创建

def reformat_bbox(bbox):
    """将bbox从[x, y, width, height]转换为[x1, y1, x2, y2]"""
    x1, y1, width, height = bbox
    x2 = x1 + width
    y2 = y1 + height
    return [x1, y1, x2, y2]

def reformat_keypoints(keypoints):
    """将keypoints从一维列表转换为列表的列表，并更改visibility标记为1"""
    formatted_keypoints = []
    # 将keypoints按每3个一组进行分组
    for i in range(0, len(keypoints), 3):
        formatted_keypoints.append([keypoints[i], keypoints[i+1], 1])  # 只取x, y, 将visibility设置为1
    return formatted_keypoints

def process_json(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    images = data['images']
    annotations = data['annotations']

    # 创建image_id到file_name的映射
    image_id_to_file_name = {img['id']: img['file_name'].replace('.jpg', '') for img in images}

    # 按image_id组织annotations
    image_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = {'bboxes': [], 'keypoints': []}
        # 重格式化bbox
        formatted_bbox = reformat_bbox(ann['bbox'])
        image_annotations[image_id]['bboxes'].append(formatted_bbox)
        # 重格式化keypoints
        formatted_keypoints = reformat_keypoints(ann['keypoints'])
        image_annotations[image_id]['keypoints'].append(formatted_keypoints)

    # 保存每个图像的annotations为新的JSON文件
    for image_id, anns in image_annotations.items():
        file_name = image_id_to_file_name[image_id]
        new_file_path = output_directory / (file_name + '.json')
        with new_file_path.open('w') as f:
            json.dump(anns, f)

        print(f'Saved {new_file_path}')

# 调用函数处理JSON文件
process_json(json_file_path)

