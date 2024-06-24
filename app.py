import os
import gdown
from PIL import Image, ImageDraw, ImageEnhance
import cv2
import numpy as np
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from mmocr.apis.inferencers.mmocr_inferencer import MMOCRInferencer


models = {
    "DBNetpp": {
        "dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015": {  # +
            "config": "/mmocr/configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py",
            "weights": "https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth",
        },
    },
    "DRRG": {
        "drrg_resnet50_fpn-unet_1200e_ctw1500": {  # +
            "config": "/mmocr/configs/textdet/drrg/drrg_resnet50_fpn-unet_1200e_ctw1500.py",
            "weights": "https://download.openmmlab.com/mmocr/textdet/drrg/drrg_resnet50_fpn-unet_1200e_ctw1500/drrg_resnet50_fpn-unet_1200e_ctw1500_20220827_105233-d5c702dd.pth",
        },
    },
    "FCENet": {
        "fcenet_resnet50-oclip_fpn_1500e_icdar2015": {  # +
            "config": "/mmocr/configs/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_icdar2015.py",
            "weights": "https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_icdar2015/fcenet_resnet50-oclip_fpn_1500e_icdar2015_20221101_150145-5a6fc412.pth",
        },
    },
    "Mask R-CNN": {
        "mask-rcnn_resnet50-oclip_fpn_160e_ctw1500": {  # +
            "config": "/mmocr/configs/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_ctw1500.py",
            "weights": "https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_ctw1500/mask-rcnn_resnet50-oclip_fpn_160e_ctw1500_20221101_154448-6e9e991c.pth",
        },
    },
    "PANet": {
        "panet_resnet18_fpem-ffm_600e_ctw1500": {  # +
            "config": "/mmocr/configs/textdet/panet/panet_resnet18_fpem-ffm_600e_ctw1500.py",
            "weights": "https://download.openmmlab.com/mmocr/textdet/panet/panet_resnet18_fpem-ffm_600e_ctw1500/panet_resnet18_fpem-ffm_600e_ctw1500_20220826_144818-980f32d0.pth",
        },
    },
    "PSENet": {
        "psenet_resnet50-oclip_fpnf_600e_icdar2015": {  # +
            "config": "/mmocr/configs/textdet/psenet/psenet_resnet50-oclip_fpnf_600e_icdar2015.py",
            "weights": "https://download.openmmlab.com/mmocr/textdet/psenet/psenet_resnet50-oclip_fpnf_600e_icdar2015/psenet_resnet50-oclip_fpnf_600e_icdar2015_20221101_131357-2bdca389.pth",
        },
    },
    "Textsnake": {
        "textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500": {  #
            "config": "/mmocr/configs/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500.py",
            "weights": "https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth",
        },
    },
}


def download_weights(weights_url, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdown.download(weights_url, output_path, quiet=False)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Пришлите мне изображение, и я найду на нем текст. Для выбора модели используйте команду /select_model. Для включения или выключения режима 'slide window' используйте команду /toggle_slide_window."
    )


async def select_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = []
    for model_name, variants in models.items():
        for variant_name in variants.keys():
            keyboard.append(
                [
                    InlineKeyboardButton(
                        f"{model_name} - {variant_name}",
                        callback_data=f"{model_name}|{variant_name}",
                    )
                ]
            )
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите модель:", reply_markup=reply_markup)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    model_name, variant_name = query.data.split("|")
    await query.answer()

    model_info = models[model_name][variant_name]
    config_path = model_info["config"]
    weights_url = model_info["weights"]
    weights_path = f"/mmocr/weights/{os.path.basename(weights_url)}"

    download_weights(weights_url, weights_path)

    # Сохранение выбранной модели в контексте пользователя
    context.user_data["model"] = {"config": config_path, "weights": weights_path}

    await query.edit_message_text(text=f"Выбрана модель: {model_name} - {variant_name}")


async def toggle_slide_window(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    slide_window = context.user_data.get("slide_window", False)
    context.user_data["slide_window"] = not slide_window
    state = "включен" if not slide_window else "выключен"
    await update.message.reply_text(f"Режим 'slide window' теперь {state}.")


async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        model = context.user_data.get("model")
        if not model:
            await update.message.reply_text(
                "Пожалуйста, сначала выберите модель с помощью команды /select_model."
            )
            return

        if update.message.photo:
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
        elif update.message.document:
            file = await context.bot.get_file(update.message.document.file_id)
        else:
            await update.message.reply_text("Пожалуйста, отправьте изображение.")
            return

        file_path = "input" + os.path.splitext(file.file_path)[-1]
        await file.download_to_drive(file_path)
        print(f"Файл загружен по пути: {file_path}")

        # Предобработка изображения для максимизации контрастности
        enhanced_image_path = enhance_contrast(file_path)
        if not enhanced_image_path:
            await update.message.reply_text(
                "Произошла ошибка при предобработке изображения."
            )
            return

        # Инициализация инференсора
        inferencer = MMOCRInferencer(
            det=model["config"], det_weights=model["weights"], device="cpu"
        )

        slide_window = context.user_data.get("slide_window", False)

        if slide_window:
            # Обработка изображения с использованием slide window
            detections = slide_window_detection(enhanced_image_path, inferencer)
        else:
            # Обработка изображения без использования slide window
            detections = process_image(enhanced_image_path, inferencer)

        print(f"Детекции: {detections}")

        if "error" in detections:
            await update.message.reply_text(
                "Произошла ошибка при обработке изображения."
            )
        else:
            output_path = draw_boxes(enhanced_image_path, detections)
            if output_path:
                try:
                    with open(output_path, "rb") as photo:
                        await update.message.reply_photo(photo=photo)
                    print(f"Фото отправлено: {output_path}")
                except Exception as e:
                    print(f"Ошибка при отправке фото: {e}")
                    await update.message.reply_text(
                        "Произошла ошибка при отправке изображения."
                    )
            else:
                await update.message.reply_text(
                    "Произошла ошибка при сохранении изображения."
                )
    except Exception as e:
        print(f"Ошибка в обработчике изображений: {e}")
        await update.message.reply_text(
            f"Произошла ошибка при обработке вашего запроса: {e}"
        )


def slide_window_detection(image_path, inferencer, tile_size=512, overlap=50):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    step = tile_size - overlap
    detections = []

    for y in range(0, height, step):
        for x in range(0, width, step):
            tile = image[y : y + tile_size, x : x + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = cv2.copyMakeBorder(
                    tile,
                    0,
                    tile_size - tile.shape[0],
                    0,
                    tile_size - tile.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )

            tile_path = f"tile_{y}_{x}.jpg"
            cv2.imwrite(tile_path, tile)

            tile_result = inferencer(tile_path)
            if "predictions" in tile_result:
                tile_detections = tile_result["predictions"][0]["det_polygons"]
                for det in tile_detections:
                    box = det
                    for i in range(len(box)):
                        box[i][0] += x
                        box[i][1] += y
                    detections.append({"coordinates": box, "score": det["score"]})

    # Применение NMS для объединения детекций
    final_detections = non_max_suppression(detections, iou_threshold=0.5)
    return final_detections


def non_max_suppression(detections, iou_threshold):
    boxes = np.array([det["coordinates"] for det in detections])
    scores = np.array([det["score"] for det in detections])

    # Преобразуем координаты боксов в формат (x1, y1, x2, y2)
    x1 = boxes[:, :, 0].min(axis=1)
    y1 = boxes[:, :, 1].min(axis=1)
    x2 = boxes[:, :, 0].max(axis=1)
    y2 = boxes[:, :, 1].max(axis=1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=0.5,
        nms_threshold=iou_threshold,
    )
    return [detections[i[0]] for i in indices]


def enhance_contrast(image_path, output_path="enhanced_image.jpg"):
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(2.0)
        enhanced_image.save(output_path)
        print(f"Enhanced image saved as {output_path}")
        return output_path
    except Exception as e:
        print(f"Ошибка при улучшении контраста: {e}")
        return None


def process_image(image_path, inferencer):
    try:
        print("Начало обработки изображения")
        result = inferencer(image_path, print_result=False)
        print(f"Результат: {result}")

        formatted_result = []
        for det_result in result["predictions"]:
            det_polygons = det_result["det_polygons"]
            det_scores = det_result["det_scores"]
            for polygon, score in zip(det_polygons, det_scores):
                formatted_result.append({"coordinates": polygon, "score": score})

        print("Обработка изображения завершена")
        return formatted_result

    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        return {"error": str(e)}


def draw_boxes(image_path, detections):
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image, "RGBA")

        for detection in detections:
            coords = detection["coordinates"]
            if isinstance(coords[0], list) or isinstance(coords[0], tuple):
                polygon = [tuple(point) for point in coords]
            else:
                polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            score = detection["score"]
            draw.polygon(
                polygon, outline="red", fill=(255, 0, 0, 100)
            )  # Полупрозрачная заливка
            draw.line(polygon + [polygon[0]], fill="red", width=3)  # Линии боксов
            draw.text(
                (polygon[0][0], polygon[0][1]), f"{score:.2f}", fill="yellow"
            )  # Цвет текста

        output_path = "output.jpg"
        try:
            image.save(output_path, format="JPEG")
            print("Image saved successfully.")
        except Exception as e:
            print(f"Failed to save image: {e}")
            output_path = None
        return output_path
    except Exception as e:
        print(f"Ошибка при рисовании боксов: {e}")
        return None


def main() -> None:
    application = (
        Application.builder()
        .token("7287622548:AAGBEwjd5nhQS-XhGv4sa6Ihc06LOfZlHM4")
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("select_model", select_model))
    application.add_handler(CommandHandler("toggle_slide_window", toggle_slide_window))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.ALL, image_handler)
    )

    application.run_polling()


if __name__ == "__main__":
    main()
