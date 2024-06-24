import os
import gdown
from PIL import Image, ImageDraw, ImageEnhance
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


def enhance_contrast(image_path, output_path="enhanced_image.jpg"):
    try:
        # Открыть изображение
        image = Image.open(image_path)

        # Преобразовать изображение в режим RGB, если оно не в этом режиме
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Создать объект Enhance для контраста
        enhancer = ImageEnhance.Contrast(image)

        # Максимизировать контраст
        enhanced_image = enhancer.enhance(
            2.0
        )  # 2.0 - значение контраста (можно варьировать)

        # Сохранить результат
        enhanced_image.save(output_path)
        print(f"Enhanced image saved as {output_path}")

        return output_path
    except Exception as e:
        print(f"Ошибка при улучшении контраста: {e}")
        return None


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
        "Привет! Пришлите мне изображение, и я найду на нем текст. Для выбора модели используйте команду /select_model."
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

        # Обработка изображения
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


def process_image(image_path, inferencer):
    try:
        print("Начало обработки изображения")
        result = inferencer(image_path, print_result=False)
        print(f"Результат: {result}")

        # Форматирование результатов
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
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(
        MessageHandler(filters.PHOTO | filters.Document.ALL, image_handler)
    )

    application.run_polling()


if __name__ == "__main__":
    main()
