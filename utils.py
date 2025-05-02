import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import layers
import torch
import shutil

# Fonction adaptée pour l'algorithme génératif de base
def load_parquet_files(root_folder, test):
    dfs = []
    # Si on veut juste un échantillon de données
    if test :
        k = 0

    # Parcourir tous les sous-dossiers dans le chemin spécifié
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder) # Chemin du sous dossier
        if os.path.isdir(folder_path):
            # Charger les fichiers parquet dans le dossier

            # Parcourir tous les fichiers dans le dossier
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # Vérifier si le fichier est un fichier Parquet
                if filename.endswith(".parquet"):
                    # Charger le fichier Parquet dans un DataFrame
                    df = pd.read_parquet(file_path)

                    # Ajouter le DataFrame à la liste
                    dfs.append(df)
        if test :
            k+=1
            if k > 10000 :
                break
    return dfs

# Fonction adaptée pour l'algorithme conditionnel
def load_parquet_files_cond(root_folder, test):
    """

    :param root_folder: Chemin d'accès aux données
    :param test: Booleen pour effectuer des tests, i.e. charger seulement une fraction des données pour
                tester si la pipeline tourne bien
    :return: Liste contenant les données d'entraînement
    """
    dfs = []
    # Si on veut juste un échantillon de données
    if test :
        k = 0
    # Parcourir tous les sous-dossiers dans le chemin spécifié
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            # Charger les fichiers parquet dans le dossier
            for filename in os.listdir(folder_path):
                if filename.endswith("s12.parquet"):
                    # Etraction de la signature des microstructures du fichier (originale + augmentations)
                    file_path = os.path.join(folder_path, filename)
                    vect = pd.read_parquet(file_path)

            # Parcourir tous les fichiers dans le dossier
            for filename in os.listdir(folder_path):
                # On extrait toutes les structures associées à la signature (en fonction de l'augmentation de données)
                if filename.endswith("preprocessed.parquet") or 'preprocessed' in filename:
                    file_path = os.path.join(folder_path, filename)
                    # Charger le fichier Parquet dans un DataFrame
                    df = pd.read_parquet(file_path)

                    # Ajouter le DataFrame à la liste
                    dfs.append((df, vect))
        if test :
            k+=1
            if k > 10000 :
                break
    return dfs

# Loading for reduced cond_gan
def load_parquet_files_cond_red(root_folder, test):
    dfs = []
    # Si on veut juste un échantillon de données
    if test :
        k = 0
    # Parcourir tous les sous-dossiers dans le chemin spécifié
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            # Charger les fichiers parquet dans le dossier
            for filename in os.listdir(folder_path):
                if filename.endswith("s12.parquet"):
                    file_path = os.path.join(folder_path, filename)
                    #print(file_path)
                    vect = pd.read_parquet(file_path)

            # Parcourir tous les fichiers dans le dossier
            for filename in os.listdir(folder_path):
                if filename.endswith(".parquet") and 'reduced' in filename:
                    file_path = os.path.join(folder_path, filename)
                    # Charger le fichier Parquet dans un DataFrame
                    df = pd.read_parquet(file_path)

                    # Ajouter le DataFrame à la liste
                    dfs.append((df, vect))
        if test :
            k+=1
            if k > 3000 :
                break
    return dfs


def load_parquet_files_cond_red_save(root_folder, destination_root, test=False):
    dfs = []
    if test:
        k = 0

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            # Créer un sous-dossier correspondant dans le dossier de destination
            dest_folder_path = os.path.join(destination_root, folder)
            os.makedirs(dest_folder_path, exist_ok=True)

            vect = None

            # Parcourir tous les fichiers dans le dossier
            for filename in os.listdir(folder_path):
                if filename.endswith("s12.parquet"):
                    file_path = os.path.join(folder_path, filename)
                    vect = pd.read_parquet(file_path)

                    # Copier le fichier dans le nouveau dossier
                    shutil.copy(file_path, os.path.join(dest_folder_path, filename))

            for filename in os.listdir(folder_path):
                if filename.endswith(".parquet") and 'reduced' in filename:
                    file_path = os.path.join(folder_path, filename)

                    # Copier le fichier dans le nouveau dossier
                    shutil.copy(file_path, os.path.join(dest_folder_path, filename))

                    # Charger les données
                    df = pd.read_parquet(file_path)
                    dfs.append((df, vect))

        if test:
            k += 1
            if k > 3000:
                break

    return dfs

# For reduced patterns
def load_parquet_files_red(root_folder, test):
    dfs = []
    # Si on veut juste un échantillon de données
    if test :
        k = 0

    # Parcourir tous les sous-dossiers dans le chemin spécifié
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            # Charger les fichiers parquet dans le dossier

            # Parcourir tous les fichiers dans le dossier
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # Vérifier si le fichier est un fichier Parquet et un fichier de structure réduite
                if filename.endswith(".parquet") and 'reduced' in filename:
                    # Charger le fichier Parquet dans un DataFrame
                    df = pd.read_parquet(file_path)

                    # Ajouter le DataFrame à la liste
                    dfs.append(df)

        if test :
            k+=1
            if k >64 :
                break

    return dfs


def data_image(data):
    data_image = [np.array(df[0]) for df in data] 
    return np.array(data_image)
    
def data_sign(data):
    data_sign = [df[1].to_numpy() for df in data] 
    return np.array(data_sign)

def generate_cond(self, cond):
        """
        Mise en forme de la condition avant passage dans le modèle
        :param cond: Liste des conditions du batch correspondant lors de l'entraînement
        :return: les conditions mises en forme
        """
        x_input = pd.concat(cond, axis=1)
        x_input = x_input.T
        x_input = np.array(x_input)
        return x_input


# Fonction adaptée pour la génération de batchs pour le WGAN
def generate_batches(data, batch_size):
    data_np = [df.to_numpy() for df in data]
    np.random.shuffle(data_np)
    batches = [data_np[i:i+batch_size] for i in range(0, len(data_np), batch_size)]

    if len(batches[-1]) != batch_size :
        batches.pop()

    return batches

# Fonction adaptée pour la génération de batchs pour le WCGAN
def generate_batches_cond(data, batch_size):
    """
    Génère les batchs pour l'entraînement (gan conditionnel)
    :param data: données d'entraînements (wcgan.data)
    :param batch_size: taille des batchs
    :return: une liste de batchs d'images et de conditions associées
    """
    data_np = [(df[0].to_numpy(), df[1]) for df in data] # Liste (dataframe/signature)
    np.random.shuffle(data_np) # mélange des données
    batches = [data_np[i:i+batch_size] for i in range(0, len(data_np), batch_size)] # Création des batchs

    # Si dernier batch plus petit que batchsize, on l'enlève
    if len(batches[-1]) != batch_size :
        batches.pop()

    return batches

def generate_batches_image_cond(data, batch_size):
    """
    Génère les batchs pour l'entraînement (gan conditionnel)
    :param data: données d'entraînements (wcgan.data)
    :param batch_size: taille des batchs
    :return: une liste de batchs d'images et de conditions associées
    """
    data_np = [df[0].to_numpy() for df in data]# Liste (dataframe/signature)
    sign_np = [df[1].to_numpy() for df in data]
    np.random.shuffle(data_np) # mélange des données
    batches = [data_np[i:i+batch_size] for i in range(0, len(data_np), batch_size)] # Création des batchs

    # Si dernier batch plus petit que batchsize, on l'enlève
    if len(batches[-1]) != batch_size :
        batches.pop()

    return batches


def interpolate_data(real_data, fake_data, batch_size):
    """
    Interpolation entre les vraies et les fausses données.
    :param real_data: Données réelles (images ou signatures).
    :param fake_data: Données fictives (images ou signatures générées).
    :param batch_size: Taille du batch.
    :return: Données interpolées.
    """
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    return alpha * real_data + (1 - alpha) * fake_data

def interpolate_signatures(real_sign, fake_sign, batch_size):
    """
    Interpolation pour les signatures.
    :param real_sign: Signatures réelles.
    :param fake_sign: Signatures fictives.
    :param batch_size: Taille du batch.
    :return: Signatures interpolées.
    """
    alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
    return alpha * real_sign + (1 - alpha) * fake_sign

def wasserstein_loss(y_true, y_pred):
    """
    Retourne une approximation de la perte de wasserstein
    :param y_true: étiquettes réelles
    :param y_pred: étiquettes prédites
    :return: score de différences entre échantillons réels et générés
    """
    # return tf.reduce_mean(y_true * y_pred)
    return tf.keras.backend.mean(y_true * y_pred)


def generator_loss(y_true, y_pred):
    # Perte adversarial classique (Wasserstein)
    adversarial_loss = -tf.reduce_mean(y_pred)
    return adversarial_loss
    
def generator_withCNN_loss(y_true, y_pred, real_sign, fake_images, CNNmodel, criterion,lambda_sign=0.01):
    # Perte adversarial classique (Wasserstein)
    adversarial_loss = generator_loss(y_true,y_pred)
    
    # Conversion des images générées (TensorFlow) en tenseur PyTorch
    fake_images_torch = transform_batch(fake_images)
    
    # Prédiction de la signature avec le CNN-RNN
    pred_sign1, pred_sign2 = CNNmodel(fake_images_torch)
    real_sign1, real_sign2 = newTorchSign(real_sign)
    
    # Calcul de la perte de signature (MSE)
    loss1 = criterion(pred_sign1, real_sign1)
    loss2 = criterion(pred_sign2, real_sign2)
    sign_loss = loss1 + loss2
    sign_loss = sign_loss.item()  # Conversion en float
    print(sign_loss,adversarial_loss)
    # Perte totale pondérée
    total_loss = adversarial_loss + lambda_sign * sign_loss
    return total_loss

def transform_batch(batch):
    """
    Fonction destiné à transformer la donnée pour le CNNRNN
    Transforme un array numpy binaire (0/1) en tensor PyTorch avec valeurs -0.5/0.5
    Transforme la forme (batch_size, 1, 128, 128) la forme originale (batch_size, 1, 256, 256)
    """
    batch_tensor = []
    # Conversion numpy array -> tensor PyTorch
    for array in batch :
        array = np.squeeze(array, axis=-1)
        array = np.expand_dims(array,axis=0)

        if array.shape != (1,128, 128):
            raise ValueError("Le tableau d'entrée doit avoir la shape (1,128,128).")
    
        # Répéter les valeurs sur les deux premières dimensions
        upscaled_array = np.repeat(np.repeat(array, 2, axis=1), 2, axis=2)

        tensor = torch.from_numpy(upscaled_array.astype(np.float32)) 
        
        # Remplacement des valeurs
        tensor = torch.where(tensor == 1, 
                            torch.tensor(0.5), 
                            torch.tensor(-0.5))
        

        batch_tensor.append(tensor)

    batch_tensor = np.array(batch_tensor)
    batch_tensor = torch.from_numpy(batch_tensor.astype(np.float32))
    
    return  batch_tensor

def newTorchSign(oldSign):
        newSign = [np.delete(oldSign[i],30) for i in range(len(oldSign))]
        newSign = [np.delete(newSign[i],60) for i in range(len(newSign))]
        newSign1 = np.array([newSign[i][0:30] for i in range(len(newSign))])
        newSign2 = np.array([newSign[i][30:60] for i in range(len(newSign))])

        newSign1 = torch.from_numpy(newSign1.astype(np.float32)) 
        newSign2 = torch.from_numpy(newSign2.astype(np.float32)) 
        
        return newSign1/2,newSign2/2


class ClipConstraint(Constraint):
    """
    Utiliser pour le clipping du gradient, ne fonctionne pas avec Conv2DCircularPadding
    """
    def __init__(self, clip_value):
        """
        Renseigner la valeur du clipping du discriminateur
        :param clip_value: valeur du clipping
        """
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return{'clip_value': self.clip_value}

class Conv2DCircularPadding(layers.Layer):
    """
    Permet le Padding circulaire pour la prise en compte de la périodicité des échantillons
    """
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="valid", activation=None, **kwargs):
        """

        :param filters: taille du filtre
        :param kernel_size: taille du kernel
        :param strides:
        :param activation: fonction d'activation de la couche
        :param kwargs:
        """
        super(Conv2DCircularPadding, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding='valid', activation=activation)

    def call(self, input_tensor):
        # Taille du padding basée sur la taille du kernel
        pad_size = self.conv.kernel_size[0] - 1
        half_pad = pad_size // 2

        # Padding circulaire
        # Ajout des bords d'un côté de l'autre côté
        padded_input = tf.concat([input_tensor[:, -half_pad:, :], input_tensor, input_tensor[:, :half_pad, :]], axis=1)
        padded_input = tf.concat([padded_input[:, :, -half_pad:], padded_input, padded_input[:, :, :half_pad]], axis=2)

        # Application de la convolution
        return self.conv(padded_input)

    def get_config(self):
        config = super(Conv2DCircularPadding, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    



