import numpy as np

def add_salt_and_pepper(image, amount):
    """
    Adiciona ruído Sal e Pimenta à imagem na proporção 'amount' (ex: 0.15 = 15%).
    """
    if amount <= 0:
        return image.copy()
    
    noisy = image.copy()
    num_noise = int(np.ceil(amount * image.size))
    
    # Ruído Sal (Pixels Brancos: 255)
    coords = [np.random.randint(0, i - 1, int(num_noise / 2)) for i in image.shape]
    noisy[tuple(coords)] = 255

    # Ruído Pimenta (Pixels Pretos: 0)
    coords = [np.random.randint(0, i - 1, int(num_noise / 2)) for i in image.shape]
    noisy[tuple(coords)] = 0

    return noisy