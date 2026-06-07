import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Cat_playing_with_a_lizard.jpg/960px-Cat_playing_with_a_lizard.jpg"


def load_remote_image(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request) as response:
        image_data = response.read()

    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise Exception("Nie udało się wczytać obrazu.")

    return image


def show_image(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_histograms(image, title):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(8, 5))
    plt.hist(gray.ravel(), bins=256, range=(0, 256))
    plt.title(f"Histogram całego zdjęcia - {title}")
    plt.xlabel("Jasność piksela")
    plt.ylabel("Liczba pikseli")
    plt.show()

    colors = ("blue", "green", "red")
    channel_names = ("Blue", "Green", "Red")

    plt.figure(figsize=(8, 5))
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color, label=channel_names[i])

    plt.title(f"Histogram kanałów RGB - {title}")
    plt.xlabel("Wartość koloru")
    plt.ylabel("Liczba pikseli")
    plt.legend()
    plt.show()


def analyze_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    p1 = np.percentile(gray, 1)
    p99 = np.percentile(gray, 99)
    dynamic_range = p99 - p1

    total_pixels = gray.size
    shadow_pixels = np.sum(gray <= 5)
    highlight_pixels = np.sum(gray >= 250)

    shadow_clip_percent = shadow_pixels / total_pixels * 100
    highlight_clip_percent = highlight_pixels / total_pixels * 100

    saturation = np.mean(hsv[:, :, 1])

    channel_means = np.mean(rgb, axis=(0, 1))
    color_balance_difference = np.max(channel_means) - np.min(channel_means)

    score = 100
    problems = []

    if brightness < 90:
        score -= 20
        problems.append("Zdjęcie jest zbyt ciemne.")
    elif brightness > 170:
        score -= 20
        problems.append("Zdjęcie jest zbyt jasne.")
    else:
        problems.append("Jasność zdjęcia jest poprawna.")

    if contrast < 40:
        score -= 20
        problems.append("Zdjęcie ma niski kontrast.")
    else:
        problems.append("Kontrast zdjęcia jest poprawny.")

    if dynamic_range < 130:
        score -= 15
        problems.append("Zdjęcie ma mały zakres tonalny.")
    else:
        problems.append("Zakres tonalny zdjęcia jest dobry.")

    if shadow_clip_percent > 3:
        score -= 10
        problems.append("Na zdjęciu występują mocno niedoświetlone obszary.")

    if highlight_clip_percent > 3:
        score -= 10
        problems.append("Na zdjęciu występują przepalone jasne obszary.")

    if saturation < 35:
        score -= 10
        problems.append("Zdjęcie ma niskie nasycenie kolorów.")
    elif saturation > 170:
        score -= 10
        problems.append("Zdjęcie ma zbyt wysokie nasycenie kolorów.")
    else:
        problems.append("Nasycenie kolorów jest poprawne.")

    if color_balance_difference > 35:
        score -= 10
        problems.append("Możliwa nierównowaga kolorów między kanałami RGB.")
    else:
        problems.append("Balans kolorów wygląda poprawnie.")

    score = max(0, min(100, score))

    if score >= 80:
        verdict = "Dobra jakość zdjęcia"
    elif score >= 60:
        verdict = "Średnia jakość zdjęcia"
    else:
        verdict = "Słaba jakość zdjęcia"

    print("\n--- Analiza jakości zdjęcia ---")
    print(f"Średnia jasność: {brightness:.2f}")
    print(f"Kontrast: {contrast:.2f}")
    print(f"Zakres tonalny: {dynamic_range:.2f}")
    print(f"Niedoświetlone piksele: {shadow_clip_percent:.2f}%")
    print(f"Przepalone piksele: {highlight_clip_percent:.2f}%")
    print(f"Średnie nasycenie: {saturation:.2f}")
    print(f"Różnica między kanałami RGB: {color_balance_difference:.2f}")
    print(f"Wynik jakości: {score}/100")
    print(f"Ocena: {verdict}")

    print("\nUwagi:")
    for problem in problems:
        print("-", problem)

    return brightness, contrast, dynamic_range, saturation


def improve_image(image, brightness, contrast, dynamic_range, saturation):
    result = image.copy()

    if brightness < 100:
        gamma = 0.75
    elif brightness > 160:
        gamma = 1.25
    else:
        gamma = 1.0

    lookup_table = np.array([
        ((i / 255.0) ** gamma) * 255 for i in range(256)
    ]).astype("uint8")

    result = cv2.LUT(result, lookup_table)

    if contrast < 45 or dynamic_range < 140:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if saturation < 45:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


def main():
    image = load_remote_image(IMAGE_URL)

    show_image(image, "Oryginalne zdjęcie")
    show_histograms(image, "oryginał")

    brightness, contrast, dynamic_range, saturation = analyze_quality(image)

    improved_image = improve_image(image, brightness, contrast, dynamic_range, saturation)

    show_image(improved_image, "Zdjęcie po poprawie jakości")
    show_histograms(improved_image, "po poprawie")

    cv2.imwrite("improved_image.jpg", improved_image)
    print("\nZapisano poprawione zdjęcie jako improved_image.jpg")


if __name__ == "__main__":
    main()