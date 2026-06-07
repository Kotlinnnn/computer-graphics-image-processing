import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "deer-3275594_1280-1210x642.jpg"

image = cv2.imread(image_path)

if image is None:
    raise Exception("Nie udało się wczytać obrazu.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Obraz oryginalny")
plt.axis("off")
plt.show()

height, width = image.shape[:2]
new_width = width // 2
new_height = height // 2

resized_image = cv2.resize(image, (new_width, new_height))

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_image, cmap="gray")
plt.title("Obraz zmniejszony i w skali szarości")
plt.axis("off")
plt.show()

rotated_image = cv2.rotate(gray_image, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(rotated_image, cmap="gray")
plt.title("Obraz obrócony o 90 stopni")
plt.axis("off")
plt.show()

plt.imshow(rotated_image, cmap="gray")
plt.title("Obraz wynikowy")
plt.axis("off")
plt.show()

print("Macierz obrazu wynikowego:")
print(rotated_image)

np.savetxt("macierz_obrazu.txt", rotated_image, fmt="%d")

print("Zapisano macierz do pliku macierz_obrazu.txt")