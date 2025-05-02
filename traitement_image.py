from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import disk, binary_erosion
from skimage.segmentation import watershed 
from skimage.filters import threshold_otsu
from skimage.measure import label,regionprops
import cv2
import tensorflow as tf
import os
from utils import transform_batch, newTorchSign
import torch

def image_processing(generated_images,sample_size,real_sign,CNNmodel):
    
    k=0
    for image_fake in generated_images:

        X_fake_reshaped = (image_fake * 255).astype(np.uint8).reshape(128, 128)
        image = Image.fromarray(X_fake_reshaped, mode='L') 

        name_image = f'image{k}.png'
        output_path = os.path.join('/Users/Bader/Desktop/Mines 2A/Projet 2A/Ines/projet-dep/Projet_InesHafassaMaiza/results', name_image)
        
        image.save(output_path)

        image_path = rf"/Users/Bader/Desktop/Mines 2A/Projet 2A/Ines/projet-dep/Projet_InesHafassaMaiza/results/image{k}.png"
        with open(image_path, "rb") as f:
            image_data = np.asarray(bytearray(f.read()), dtype=np.uint8)

        #On décode l'image
        imageenr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(imageenr, cv2.COLOR_BGR2GRAY)

        smoothed = gaussian_filter(gray,sigma=0.5) #modifier le sigma peut modifier nos séparations

        tresh_value = threshold_otsu(smoothed)
        binary = (smoothed > tresh_value).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        kernel = disk(2) # Attention ! Plus le kernel est petit plus les artefacts seront conservés
        binary_opened = binary_erosion(binary, kernel)

        distance=distance_transform_edt(binary_opened)
        markers=label(distance>0.3*distance.max()) #si on augmente le 0.3 les petits clusters peuvent disparaitre
        local_maxi = watershed(-distance,markers,mask=binary_opened)
        binary_local_maxi = (local_maxi > 0).astype(np.uint8) * 255
        
        image_bw = Image.fromarray(binary_local_maxi, mode='L')

        name_image2 = f'image2_{k}.png'
        output_path = os.path.join('/Users/Bader/Desktop/Mines 2A/Projet 2A/Ines/projet-dep/Projet_InesHafassaMaiza/results', name_image2)

        image_bw.save(output_path)

        image_path = rf"/Users/Bader/Desktop/Mines 2A/Projet 2A/Ines/projet-dep/Projet_InesHafassaMaiza/results/image2_{k}.png"

        with open(image_path, "rb") as f:
            image_data = np.asarray(bytearray(f.read()), dtype=np.uint8)

        #On décode l'image
        imageenr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(imageenr, cv2.COLOR_BGR2GRAY)


        #On applique un seuillage
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # ON évite les petits artefacts
        kernel = np.ones((3, 3), np.uint8) #pareil modifier le kernel modifie nos artefacts
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # On trouve les contours des clusters
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Image sur laquelle on dessine les cercles
        output = np.zeros_like(image)

        # taille fixe des cercles
        fixed_radius =imageenr.shape[0]*3.98942280401433*10**(-2) 
        dpi = 25
        figsize = ( imageenr.shape[0] / dpi, imageenr.shape[1] / dpi)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(output, cmap='gray')

        W=128
        H=128
        
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"] )
                cy = int(M["m01"] / M["m00"] )
        
                # DOn dessine un cercle de taille fixe centré sur le cluster
                #cv2.circle(output, (cx, cy), fixed_radius, (255, 255, 255), -1)
                circle = plt.Circle((cx, cy), fixed_radius, color='white', fill=True, alpha=0.9)
                ax.add_patch(circle)

                translations = [
                    (-W, 0),  # Gauche
                    (W, 0),   # Droite
                    (0, -H),  # Bas
                    (0, H),   # Haut
                    (-W, -H), # Coin Bas-Gauche
                    (-W, H),  # Coin Haut-Gauche
                    (W, -H),  # Coin Bas-Droite
                    (W, H)    # Coin Haut-Droite
                ]

                for dx, dy in translations:
                    shifted_circle = plt.Circle((cx + dx, cy + dy), fixed_radius, color='white', fill=True, alpha=0.9)
                    ax.add_patch(shifted_circle)
        ax.set_position([0,0,1,1])

        ax.set_xticks([])  
        ax.set_yticks([])  
        ax.set_frame_on(False)  
        ax.set_facecolor('black')

        fig.canvas.draw()


        image_array = np.array(fig.canvas.renderer.buffer_rgba())  # Récupération en RGBA
        plt.close(fig)  # Fermer la figure pour libérer la mémoire
        print(image_array.shape)

        # Conversion en niveaux de gris et seuillage
        gray_image = np.mean(image_array[:H, :W, :3], axis=2)  # Convertir en niveaux de gris
        binary_array = gray_image > 100  # Seuil pour obtenir une matrice de 0 et 1
        print(binary_array.shape)
        # Supposons que ton array s'appelle `single_image`
        batch_tensor = []
        binary_array2 = np.expand_dims(binary_array, axis=0)  # Shape devient (1,128,128)
        upscaled_array = np.repeat(np.repeat(binary_array2, 2, axis=1), 2, axis=2)

        tensor = torch.from_numpy(upscaled_array.astype(np.float32)) 
        
        # Remplacement des valeurs
        tensor = torch.where(tensor == 1, 
                            torch.tensor(0.5), 
                            torch.tensor(-0.5))

        batch_tensor.append(tensor)
        batch_tensor = np.array(batch_tensor)
        batch_tensor = torch.from_numpy(batch_tensor.astype(np.float32))
        
        pred_sign1, pred_sign2 = CNNmodel(batch_tensor)
        real_sign1, real_sign2 = newTorchSign(real_sign)
        
        # Affichage du résultat
        plt.figure()
        plt.imshow(binary_array, cmap='gray')
        plt.title("Image en binaire")
        plt.show()
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(real_sign1[0].detach().numpy(), label="Real Sign 1")
        plt.plot(pred_sign1[0].detach().numpy(), label="Pred Sign 1")
        plt.legend()
        plt.title("Premier module")
    
        plt.subplot(1, 2, 2)
        plt.plot(real_sign2[0].detach().numpy(), label="Real Sign 2")
        plt.plot(pred_sign2[0].detach().numpy(), label="Pred Sign 2")
        plt.legend()
        plt.title("Deuxième module")
    
        plt.show()
        k+=1
