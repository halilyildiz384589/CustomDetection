# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:34:05 2023

@author: halil
"""

import glob
import shutil
import os

source_folder = "C:\\Users\\halil\\Desktop\\OpenCV_python\\customdetection2\\OIDv4_ToolKit\\OID\\Dataset\\train\\Human hand"  # Kaynak klasör yolu
destination_folder = "C:\\Users\\halil\\Desktop\\OpenCV_python\\customdetection2\\OIDv4_ToolKit\\OID\\Dataset\\train\\Human hand\\labels\\train"  # Hedef klasör yolu

txt_files = glob.glob(source_folder + "/*.txt")

for file_path in txt_files:
    # Dosyayı hedef klasöre taşı
    shutil.move(file_path, destination_folder)
