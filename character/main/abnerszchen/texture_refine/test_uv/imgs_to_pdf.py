from PIL import Image
from reportlab.lib.pagesizes import A0
from reportlab.pdfgen import canvas
import argparse

import os
import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
from dataset.utils_dataset import parse_objs_json


def images_to_pdf(image_paths, output_path):
    # 创建一个PDF画布
    pdf_canvas = canvas.Canvas(output_path, pagesize=A0)

    # 页面尺寸
    page_width, page_height = A0

    # 初始化图像垂直位置
    y = page_height

    # 遍历图像路径列表
    for image_path in image_paths:
        # 打开图像
        image = Image.open(image_path)

        # 调整图像大小以适应页面宽度
        image_width, image_height = image.size
        scale = page_width / image_width
        new_width = int(image_width * scale)
        new_height = int(image_height * scale)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # 更新图像垂直位置
        y -= new_height

        # 如果当前位置不足以容纳下一张图像，则开始新的一页
        if y < 0:
            pdf_canvas.showPage()
            y = page_height - new_height

        # 计算图像的位置（左上角坐标）
        x = 0

        # 将图像绘制到PDF画布上
        pdf_canvas.drawInlineImage(image, x, y, width=new_width, height=new_height)

    # 结束最后一页
    pdf_canvas.showPage()

    # 保存PDF文件
    pdf_canvas.save()


def parse_paths(in_json, select_key="infer_uv_sdxl"):
    objs_dict, key_pair_list = parse_objs_json(in_json)
    image_paths = []
    for d_, dname, oname in key_pair_list:
        meta = objs_dict[d_][dname][oname]
        if select_key not in meta:
            continue
        image_paths.append(meta[select_key])
    return image_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_json',
                        type=str,
                        default='/aigc_cfs_3/sz/result/compare_c_cxs/vis_mcwy2_test/my/output.json')
    parser.add_argument('out_pdf', type=str, help='')
    parser.add_argument('--select_key', type=str, default="infer_uv_sdxl", help='')
    args = parser.parse_args()

    image_paths = parse_paths(args.in_json, select_key=args.select_key)
    images_to_pdf(image_paths, args.out_pdf)
    print(f"save from {args.in_json} to pdf {args.out_pdf}")
