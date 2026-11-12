from cv2 import cv2
import numpy as np


def handle_image(path_in, path_depth, dict_config):
    """
    根据左眼图片和深度图，合成右眼视角并生成多种3D输出格式。

    Args:
        path_in: 左眼视角图片路径
        path_depth: 深度图路径 (灰度图, 0~255, 值越大表示距离越近/突出)
        dict_config: 配置字典，包含以下键：
            - list_kind: 输出格式列表 ('ot','bd','rb','lr','rl')
            - rate_distance: 景深强度 (0~1)，控制左右视差大小
            - rate_out_screen: 出屏率 (0~1)，控制画面整体偏移量
            - rate_dilate: 深度图膨胀比例 (0~1)，用于平滑深度边缘

    Returns:
        dict: 包含各格式输出图片的字典；若无需合成右眼则可能不含'ot'等键
    """
    # ========== 读取配置参数 ==========
    list_kind = dict_config['list_kind']          # 需要输出的3D格式列表

    rate_template_width = 0.03   # 黑框出屏效果：边框宽度占图片宽度的比例
    rate_template_height = 0.1   # 黑框出屏效果：边框高度占图片高度的比例
    rate_zoom_25D = 0.3          # 2.5D黑框模式：深度图缩放比例（用于统计深度分布）
    length_min_25D = 10          # 2.5D黑框模式：深度直方图滑动窗口最小长度

    rate_distance = dict_config['rate_distance']      # 景深系数，决定最大像素偏移量
    rate_out_screen = dict_config['rate_out_screen']  # 出屏率，决定零视差平面的位置
    rate_dilate = dict_config['rate_dilate']          # 深度图膨胀系数，消除边缘伪影

    dict_image_out = {}

    # ========== 读取输入图片 ==========
    image_in = cv2.imread(path_in, cv2.IMREAD_COLOR)         # 左眼彩色图
    image_depth = cv2.imread(path_depth, cv2.IMREAD_GRAYSCALE)  # 深度灰度图

    shape = image_in.shape
    height, width = shape[0], shape[1]
    value_unit = max(height, width)                           # 以较长边作为基准单位
    value_distance = int(value_unit * rate_distance)          # 最大视差像素数
    value_dilate = round(value_unit * rate_dilate) + 1        # 膨胀核尺寸基数

    # 确保深度图与原图尺寸一致
    if image_in.shape[:2] != image_depth.shape:
        image_depth = cv2.resize(image_depth, (width, height))

    # ========== 黑框出屏效果 (2.5D) ==========
    # 原理：通过深度阈值分割前景，叠加黑色边框模板，营造物体"跳出"边框的立体感
    if 'bd' in list_kind:
        # 对深度图做腐蚀+模糊，去除噪点并使前景区域更紧凑
        image_depth_25D = cv2.erode(image_depth, np.ones((3, 3), np.uint8), iterations=3)
        image_depth_25D = cv2.GaussianBlur(image_depth_25D, (3, 3), 0)

        # 构建三栏式黑框模板mask：三个白色矩形区域，其余为黑色边框
        mask_template = np.zeros((height, width), dtype=np.uint8)
        width_template = round(width * rate_template_width)
        height_template = round(height * rate_template_height)
        width_box = round((width - 4 * width_template) / 3)  # 每个内容区域的宽度
        for j in range(3):
            y_start = height_template
            y_end = height - height_template
            x_start = width_template + j * (width_box + width_template)
            x_end = (j + 1) * (width_box + width_template)
            mask_template[y_start: y_end, x_start: x_end] = 255

        # 缩小深度图用于快速统计深度值分布（降低计算量）
        image_depth_25D_zoom = cv2.resize(image_depth_25D, (0, 0), fx=rate_zoom_25D, fy=rate_zoom_25D)

        # 统计每个深度值的像素数量（等效于灰度直方图）
        list_num = []
        num_before = 0
        for depth in range(0, 255):
            num_now = (image_depth_25D_zoom < depth + 1).sum()
            num = num_now - num_before
            num_before = num_now
            list_num.append(num)

        # 在后半段深度范围中，找滑动窗口内像素数最少的深度值作为前景分割阈值
        # 目的：避开背景大面积区域，选中前景与背景的过渡分界
        list_num_sum = [sum(list_num[i: i + length_min_25D]) for i in range(0, len(list_num) - length_min_25D)]
        search_start = len(list_num_sum) // 2
        search_end = len(list_num_sum) // 4 * 3
        index_depth_template = list_num_sum.index(min(list_num_sum[search_start: search_end]))
        depth_template = index_depth_template + length_min_25D // 2

        # 低于阈值的深度置零（视为背景），只保留前景
        image_depth_25D[image_depth_25D < depth_template] = 0

        # 将前景mask与黑框模板做OR运算，再掩码提取原图对应区域
        image_out_bd = cv2.bitwise_or(image_depth_25D, mask_template)
        image_out_bd = cv2.bitwise_and(image_in, image_in, mask=image_out_bd)

        dict_image_out['bd'] = image_out_bd

    # ========== 合成右眼视角及3D输出 ==========
    if 'ot' in list_kind or 'rb' in list_kind or 'lr' in list_kind or 'rl' in list_kind:

        # --- Step 1: 深度图预处理 ---
        # 非均匀膨胀：水平方向大于垂直方向，适应人眼水平视差特性，减少纵向伪影
        # 作者注：该处极其关键，一是可以将前景的白色区域扩大，能缓解前景边缘不准确的问题，二是可以减少补图区域的大小，加快效率
        kernel_w = round(value_dilate * 1.67)
        kernel_h = value_dilate
        image_depth_3D = cv2.dilate(image_depth, np.ones((kernel_h, kernel_w), np.uint8), iterations=3)

        # 作者注：深度图预处理还可使用中值滤波，可去除噪点，且不是很损失细节

        # --- Step 2: 边缘自适应平滑 ---
        # 在深度突变处用模糊替代原始值，避免remap时产生锯齿和撕裂
        # 作者注：该处可以减少边缘的瑕疵，但是会有浮雕效果
        diff_smooth_max = 100   # 判定"边缘"的差分阈值
        width_smooth = 31       # 平滑窗口半径

        blur_size = width_smooth * 2 + 1
        image_depth_3D_blur = cv2.GaussianBlur(image_depth_3D, (blur_size, 3), 0)      # 水平高斯模糊
        image_depth_3D_dilate = cv2.dilate(image_depth_3D, np.ones((1, width_smooth), np.uint8), iterations=1)

        # 仅当膨胀值与原值差异超过阈值时，才替换为模糊值（保护平坦区域细节）
        edge_mask = np.abs(image_depth_3D_dilate - image_depth_3D) > diff_smooth_max
        image_depth_3D = np.where(edge_mask, image_depth_3D_blur, image_depth_3D)

        # --- Step 3: 构建重映射坐标表 ---
        # 将深度值转换为水平位移：depth越大 → 偏移越大 → 视觉上越靠前
        image_depth_3D = image_depth_3D.astype(np.float32)
        # 减去出屏基准偏移，使rate_out_screen对应的深度成为零视差平面
        image_depth_3D = image_depth_3D - int((1 - rate_out_screen) * 255)

        # 生成原始像素坐标网格
        map_x = np.tile(np.arange(width), (height, 1))                    # shape=(H,W), 每行=[0,1,...,W-1]
        map_y = np.tile(np.arange(height).reshape((height, 1)), (1, width))  # shape=(H,W), 每列=[0,1,...,H-1]

        # 计算每个目标像素应从源图的哪个x坐标采样（负偏移=向右移=物体更近）
        map_x_before = (map_x - np.round(image_depth_3D * value_distance / 256)).astype(int)

        # 越界坐标标记为-1（无效）
        map_x_before[map_x_before < 0] = -1
        map_x_before[map_x_before >= width] = -1

        # 初始化新映射表，默认-1表示该位置无有效源像素（待修复空洞）
        map_x_new = np.full((height, width), -1, dtype=int)

        # 仅将合法坐标写入映射表
        is_inside = map_x_before >= 0
        map_x_new[map_y[is_inside].flatten(), map_x_before[is_inside].flatten()] = map_x[is_inside].flatten()

        # --- Step 4: 执行重映射 ---
        map_x_new = map_x_new.astype(np.float32)
        map_y_new = map_y.astype(np.float32)

        # 对映射表做小幅膨胀，填补因离散化产生的细小缝隙，减少inpaint负担
        map_x_new = cv2.dilate(map_x_new, np.ones((2, 2), np.uint8), iterations=3)
        map_y_new = cv2.dilate(map_y_new, np.ones((2, 2), np.uint8), iterations=3)

        # 转为BGRA以便通过Alpha通道识别无效区域
        image_in_bgra = cv2.cvtColor(image_in, cv2.COLOR_BGR2BGRA)

        # INTER_NEAREST避免插值引入半透明伪影；BORDER_CONSTANT填充黑色(0,0,0,0)
        image_out_3D = cv2.remap(
            image_in_bgra, map_x_new, map_y_new,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # --- Step 5: 空洞修复 (Inpainting) ---
        # Alpha通道为0的位置即remap未覆盖的空洞
        _, _, _, mask_inpaint = cv2.split(image_out_3D)
        mask_inpaint = cv2.bitwise_not(mask_inpaint)  # 反转：空洞处为白色(255)

        image_out_3D = cv2.cvtColor(image_out_3D, cv2.COLOR_BGRA2BGR)

        # TELEA算法基于快速行进法(FMM)，适合修复小面积不规则空洞
        flags_inpaint = cv2.INPAINT_TELEA
        image_out_3D = cv2.inpaint(image_out_3D, mask_inpaint, inpaintRadius=3, flags=flags_inpaint)

        # --- Step 6: 组装输出 ---
        if 'ot' in list_kind:
            dict_image_out['ot'] = image_out_3D  # 单独输出右眼视角

        # 合并生成红蓝/左右/右左等3D格式
        dict_image_out.update(merge_3D_image(image_in, image_out_3D, list_kind))

        return dict_image_out

def merge_3D_image(image_in, image_ot, list_kind):
    """
    根据左眼原图和合成的右眼视角图，生成指定格式的3D图片。

    Args:
        image_in: 左眼视角原始图片 (BGR)
        image_ot: 合成的右眼视角图片 (BGR)
        list_kind: 需要生成的3D格式列表，支持 'rb'(红蓝), 'lr'(左右平行), 'rl'(右左交叉)

    Returns:
        dict: 键为格式名称，值为对应3D图片的字典
    """
    dict_image_out = {}

    # 红蓝3D：左眼取R通道，右眼取BG通道，合并后需佩戴红蓝眼镜观看
    if 'rb' in list_kind:
        _, _, r = cv2.split(image_in)       # 左眼只保留红色通道
        b, g, _ = cv2.split(image_ot)       # 右眼只保留蓝、绿通道
        image_out_rb = cv2.merge([b, g, r]) # 按BGR顺序合并
        dict_image_out['rb'] = image_out_rb

    # 左右格式（平行眼）：左图在左，右图在右，适用于平行眼观看法或VR设备
    if 'lr' in list_kind:
        image_out_lr = np.concatenate([image_in, image_ot], axis=1)
        dict_image_out['lr'] = image_out_lr

    # 右左格式（交叉眼）：右图在左，左图在右，适用于交叉眼观看法
    if 'rl' in list_kind:
        image_out_rl = np.concatenate([image_ot, image_in], axis=1)
        dict_image_out['rl'] = image_out_rl

    return dict_image_out