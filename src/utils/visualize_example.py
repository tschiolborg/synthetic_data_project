import json
import os

from PIL import Image, ImageColor, ImageDraw, ImageFont


def load_annotation(image_key):
    with open(os.path.join("/../../data/MTSD/annotations", f"{image_key}.json"), "r") as fid:
        anno = json.load(fid)
    return anno


def visualize_gt(image_key, anno, color="green", alpha=125, font=None):
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        print("Falling back to default font...")
        font = ImageFont.load_default()

    img_folder = "train"

    with open("data/splits/test.txt") as f:
        if image_key in f.readlines():
            img_folder = "test"

    with open("data/splits/val.txt") as f:
        if image_key in f.readlines():
            img_folder = "val"

    with Image.open(os.path.join(f"data/images/{img_folder}", f"{image_key}.jpg")) as img:
        img = img.convert("RGBA")
        img_draw = ImageDraw.Draw(img)

        rects = Image.new("RGBA", img.size)
        rects_draw = ImageDraw.Draw(rects)

        for obj in anno["objects"]:
            x1 = obj["bbox"]["xmin"]
            y1 = obj["bbox"]["ymin"]
            x2 = obj["bbox"]["xmax"]
            y2 = obj["bbox"]["ymax"]

            color_tuple = ImageColor.getrgb(color)
            if len(color_tuple) == 3:
                color_tuple = color_tuple + (alpha,)
            else:
                color_tuple[-1] = alpha

            rects_draw.rectangle((x1 + 1, y1 + 1, x2 - 1, y2 - 1), fill=color_tuple)
            img_draw.line(
                ((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)),
                fill="black",
                width=1,
            )

            class_name = obj["label"]
            img_draw.text((x1 + 5, y1 + 5), class_name, font=font)

        img = Image.alpha_composite(img, rects)

        img = img.resize((254, 254), Image.ANTIALIAS)

        return img


if __name__ == "__main__":
    image_key = "OENb8BfFyAocFzHHM4Mehg"

    # load the annotation json
    anno = load_annotation(image_key)

    # visualize traffic sign boxes on the image
    vis_img = visualize_gt(image_key, anno)
    vis_img.show()
