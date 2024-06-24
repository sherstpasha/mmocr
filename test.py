from PIL import Image, ImageEnhance


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


# Пример использования функции
enhanced_image_path = enhance_contrast(
    r"C:\Users\pasha\OneDrive\Рабочий стол\test\00004-scan_2021-12-03_07-53-53.jpg",
    r"C:\Users\pasha\OneDrive\Рабочий стол\test\00004-scan_2021-12-03_07-53-53_.jpg",
)
