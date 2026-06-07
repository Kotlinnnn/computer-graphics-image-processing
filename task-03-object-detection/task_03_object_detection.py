import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Lille_restaurant_12_rue_d%27arras.jpg/960px-Lille_restaurant_12_rue_d%27arras.jpg"
IMAGE_PATH = "input_image.jpg"
PREPROCESSED_PATH = "preprocessed_image.jpg"
RESULT_PATH = "detection_result.jpg"


def download_image(url, output_path):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request) as response:
        image_data = response.read()

    with open(output_path, "wb") as file:
        file.write(image_data)


def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Nie udało się wczytać obrazu.")

    image = cv2.resize(image, (800, 600))

    image = cv2.GaussianBlur(image, (3, 3), 0)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    improved_lab = cv2.merge((l, a, b))
    improved_image = cv2.cvtColor(improved_lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, improved_image)

    return improved_image


def show_image(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 7))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def detect_objects(image_path):
    model = YOLO("yolov8n.pt")

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False
    )

    result = results[0]

    annotated_image = result.plot()
    cv2.imwrite(RESULT_PATH, annotated_image)

    print("\n--- Wykryte obiekty ---")

    if result.boxes is None or len(result.boxes) == 0:
        print("Nie wykryto obiektów.")
        return annotated_image

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(
            f"Obiekt: {class_name}, "
            f"pewność: {confidence:.2f}, "
            f"ramka: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})"
        )

    return annotated_image


def main():
    download_image(IMAGE_URL, IMAGE_PATH)

    original_image = cv2.imread(IMAGE_PATH)
    show_image(original_image, "Obraz oryginalny")

    preprocessed_image = preprocess_image(IMAGE_PATH, PREPROCESSED_PATH)
    show_image(preprocessed_image, "Obraz po preprocessingu")

    detected_image = detect_objects(PREPROCESSED_PATH)
    show_image(detected_image, "Wynik detekcji obiektów")

    print(f"\nZapisano obraz po preprocessingu jako: {PREPROCESSED_PATH}")
    print(f"Zapisano wynik detekcji jako: {RESULT_PATH}")


if __name__ == "__main__":
    main()