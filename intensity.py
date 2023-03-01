import numpy as np
import SimpleITK as sitk
from scipy.stats import rice
from .rng import rng
from .augmentation import choose_random_scale_factor


def rician_noise(signal, sigma=0.):
    signal_shape = signal.shape
    signal = signal.flatten()
    b = signal/sigma
    output = rice.rvs(b, scale=sigma)
    output = output.reshape(signal_shape)
    return output


# def alt_rician_noise(signal, SNR=1.):
#     signal_shape = signal.shape
#     # print(signal_shape)
#     signal = signal.flatten()
#     b = SNR
#     output = rice.rvs(b, scale=signal/SNR)
#     # print(output.shape)
#     # print(signal_shape)
#     output = output.reshape(signal_shape)
#     # print(output.shape)
#     return output


# def alt_rician_noise(signal, sigma=0.):
#     signal_shape = signal.shape
#     signal = signal.flatten()
#     b = signal / sigma
#     output = np.empty_like(signal)
#     for i in range(len(b)):
#         output[i] = rice.rvs(b[i], scale=sigma)
#     # output = rice.rvs(b, scale=sigma)
#     output = output.reshape(signal_shape)
#     return output


def apply_noise(image, sigma):
    if sigma == 0:
        return image
    image_array = sitk.GetArrayFromImage(image)
    noisy_image_array = rician_noise(image_array, sigma)
    noisy_image = sitk.GetImageFromArray(noisy_image_array)
    noisy_image.CopyInformation(image)
    return noisy_image


# def alt_apply_noise(image, sigma):
#     image_array = sitk.GetArrayFromImage(image)
#     noisy_image_array = alt_rician_noise(image_array, sigma)
#     noisy_image = sitk.GetImageFromArray(noisy_image_array)
#     noisy_image.CopyInformation(image)
#     return noisy_image


def scale_image_intensity(image, scale=1.):
    scale_filter = sitk.ShiftScaleImageFilter()
    scale_filter.SetScale(scale)
    scaled_image = scale_filter.Execute(image)
    return scaled_image


def randomly_augment_image_intensity(image, scale_factor_bounds=1., noise_sigma=0.):
    # TODO decide if scaling or noise should go first
    scale_factor = choose_random_scale_factor(scale_factor_bounds)
    scaled_image = scale_image_intensity(image, scale_factor)
    noisy_image = apply_noise(scaled_image, noise_sigma)
    return noisy_image
