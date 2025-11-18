"""
Dataset Manager for nnUNet v2

Bu modül dataset yönetimi ve otomatik ID sistemi sağlar.
Kullanıcı deneyimini iyileştirmek için tasarlanmıştır.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger('dataset_manager')

class DatasetManager:
    """Dataset yönetimi ve otomatik ID sistemi için sınıf."""
    
    def __init__(self, nnunet_raw_dir: str):
        """
        DatasetManager'ı başlat.
        
        Args:
            nnunet_raw_dir (str): nnUNet_raw dizin yolu
        """
        self.nnunet_raw_dir = nnunet_raw_dir
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "dataset_config.json")
        self.load_config()
    
    def load_config(self):
        """Konfigürasyon dosyasını yükle."""
        default_config = {
            'last_dataset_id': 0,
            'default_dataset_name': 'dataset',
            'dataset_history': []
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.warning(f"Konfigürasyon dosyası yüklenemedi, varsayılan ayarlar kullanılıyor: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Konfigürasyonu dosyaya kaydet."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Konfigürasyon dosyası kaydedilemedi: {e}")
    
    def get_next_dataset_id(self) -> int:
        """Sıradaki dataset ID'sini al."""
        existing_ids = self.get_existing_dataset_ids()
        max_existing_id = max(existing_ids) if existing_ids else 0
        next_id = max(self.config['last_dataset_id'] + 1, max_existing_id + 1)
        return next_id
    
    def get_existing_dataset_ids(self) -> List[int]:
        """Mevcut dataset ID'lerini al."""
        ids = []
        if not os.path.exists(self.nnunet_raw_dir):
            return ids
        
        for item in os.listdir(self.nnunet_raw_dir):
            item_path = os.path.join(self.nnunet_raw_dir, item)
            if os.path.isdir(item_path):
                # Dataset001_Name formatındaki klasörleri ara
                match = re.match(r'Dataset(\d+)_', item)
                if match:
                    ids.append(int(match.group(1)))
        
        return sorted(ids)
    
    def get_existing_datasets(self) -> List[Dict]:
        """Mevcut dataset'leri listele."""
        datasets = []
        if not os.path.exists(self.nnunet_raw_dir):
            return datasets
        
        for item in os.listdir(self.nnunet_raw_dir):
            item_path = os.path.join(self.nnunet_raw_dir, item)
            if os.path.isdir(item_path):
                match = re.match(r'Dataset(\d+)_(.+)', item)
                if match:
                    dataset_id = int(match.group(1))
                    dataset_name = match.group(2)
                    
                    # Dataset bilgilerini topla
                    dataset_info = {
                        'id': dataset_id,
                        'name': dataset_name,
                        'folder_name': item,
                        'path': item_path,
                        'has_training_data': self._check_training_data(item_path),
                        'has_test_data': self._check_test_data(item_path),
                        'file_count': self._count_files(item_path)
                    }
                    datasets.append(dataset_info)
        
        return sorted(datasets, key=lambda x: x['id'])
    
    def _check_training_data(self, dataset_path: str) -> bool:
        """Dataset'te training data olup olmadığını kontrol et."""
        imagesTr_path = os.path.join(dataset_path, 'imagesTr')
        labelsTr_path = os.path.join(dataset_path, 'labelsTr')
        
        if os.path.exists(imagesTr_path) and os.path.exists(labelsTr_path):
            images_count = len([f for f in os.listdir(imagesTr_path) if f.endswith('.nii.gz')])
            labels_count = len([f for f in os.listdir(labelsTr_path) if f.endswith('.nii.gz')])
            return images_count > 0 and labels_count > 0
        return False
    
    def _check_test_data(self, dataset_path: str) -> bool:
        """Dataset'te test data olup olmadığını kontrol et."""
        imagesTs_path = os.path.join(dataset_path, 'imagesTs')
        
        if os.path.exists(imagesTs_path):
            test_count = len([f for f in os.listdir(imagesTs_path) if f.endswith('.nii.gz')])
            return test_count > 0
        return False
    
    def _count_files(self, dataset_path: str) -> Dict[str, int]:
        """Dataset'teki dosya sayılarını hesapla."""
        counts = {
            'images_tr': 0,
            'labels_tr': 0,
            'images_ts': 0,
            'labels_ts': 0
        }
        
        # Training images
        imagesTr_path = os.path.join(dataset_path, 'imagesTr')
        if os.path.exists(imagesTr_path):
            counts['images_tr'] = len([f for f in os.listdir(imagesTr_path) if f.endswith('.nii.gz')])
        
        # Training labels
        labelsTr_path = os.path.join(dataset_path, 'labelsTr')
        if os.path.exists(labelsTr_path):
            counts['labels_tr'] = len([f for f in os.listdir(labelsTr_path) if f.endswith('.nii.gz')])
        
        # Test images
        imagesTs_path = os.path.join(dataset_path, 'imagesTs')
        if os.path.exists(imagesTs_path):
            counts['images_ts'] = len([f for f in os.listdir(imagesTs_path) if f.endswith('.nii.gz')])
        
        # Test labels (optional)
        labelsTs_path = os.path.join(dataset_path, 'labelsTs')
        if os.path.exists(labelsTs_path):
            counts['labels_ts'] = len([f for f in os.listdir(labelsTs_path) if f.endswith('.nii.gz')])
        
        return counts
    
    def create_dataset_name(self, custom_name: Optional[str] = None) -> Tuple[str, int]:
        """
        Yeni dataset adı ve ID oluştur.
        
        Args:
            custom_name (str, optional): Özel dataset adı
            
        Returns:
            tuple: (dataset_name, dataset_id)
        """
        dataset_id = self.get_next_dataset_id()
        
        if custom_name:
            dataset_name = custom_name
        else:
            dataset_name = self.config['default_dataset_name']
        
        # Konfigürasyonu güncelle
        self.config['last_dataset_id'] = dataset_id
        entry = {
            'id': dataset_id,
            'name': dataset_name,
            'created_at': str(Path().resolve())  # basit timestamp yerine
        }
        self.config['dataset_history'].append(entry)
        self.save_config()
        
        return dataset_name, dataset_id
    
    def list_unconverted_datasets(self, raw_data_dirs: List[str]) -> List[Dict]:
        """
        Dönüştürülmemiş dataset'leri listele.
        
        Args:
            raw_data_dirs (List[str]): Ham veri dizinleri listesi
            
        Returns:
            List[Dict]: Dönüştürülmemiş dataset'ler
        """
        unconverted = []
        
        for data_dir in raw_data_dirs:
            if not os.path.exists(data_dir):
                continue
                
            # Her bir alt dizini kontrol et
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    # Bu bir dataset dizini mi kontrol et
                    if self._is_raw_dataset(item_path):
                        dataset_info = {
                            'name': item,
                            'path': item_path,
                            'size': self._get_directory_size(item_path),
                            'file_types': self._get_file_types(item_path),
                            'estimated_type': self._estimate_dataset_type(item_path)
                        }
                        unconverted.append(dataset_info)
        
        return unconverted
    
    def _is_raw_dataset(self, path: str) -> bool:
        """Dizinin ham dataset olup olmadığını kontrol et."""
        # NIfTI dosyaları var mı?
        nifti_files = list(Path(path).rglob("*.nii.gz")) + list(Path(path).rglob("*.nii"))
        if len(nifti_files) > 0:
            return True
        
        # ZIP dosyaları var mı?
        zip_files = list(Path(path).rglob("*.zip"))
        if len(zip_files) > 0:
            return True
        
        return False
    
    def _get_directory_size(self, path: str) -> str:
        """Dizin boyutunu hesapla."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            
            # Boyutu uygun birimde göster
            if total_size < 1024**2:  # MB
                return f"{total_size / 1024:.1f} KB"
            elif total_size < 1024**3:  # GB
                return f"{total_size / (1024**2):.1f} MB"
            else:
                return f"{total_size / (1024**3):.1f} GB"
        except:
            return "Bilinmiyor"
    
    def _get_file_types(self, path: str) -> List[str]:
        """Dizindeki dosya türlerini listele."""
        extensions = set()
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext:
                    extensions.add(ext)
        return sorted(list(extensions))
    
    def _estimate_dataset_type(self, path: str) -> str:
        """Dataset türünü tahmin et."""
        nifti_count = len(list(Path(path).rglob("*.nii.gz")) + list(Path(path).rglob("*.nii")))
        
        if nifti_count > 0:
            return f"Medical Imaging ({nifti_count} NIfTI files)"
        
        zip_count = len(list(Path(path).rglob("*.zip")))
        if zip_count > 0:
            return f"Archived Data ({zip_count} ZIP files)"
        
        return "Unknown"
    
    def display_datasets_menu(self, datasets: List[Dict]) -> int:
        """Dataset seçim menüsünü göster."""
        if not datasets:
            print("\nDönüştürülecek dataset bulunamadı.")
            return -1
        
        print("\n" + "="*80)
        print("DÖNÜŞTÜRÜLECEK DATASET'LER".center(80))
        print("="*80)
        
        for i, dataset in enumerate(datasets):
            print(f"\n{i+1}. {dataset['name']}")
            print(f"   Yol: {dataset['path']}")
            print(f"   Boyut: {dataset['size']}")
            print(f"   Dosya türleri: {', '.join(dataset['file_types'])}")
            print(f"   Tahmini tür: {dataset['estimated_type']}")
        
        print(f"\n{len(datasets)+1}. İptal")
        
        while True:
            try:
                choice = int(input(f"\nSeçiminiz (1-{len(datasets)+1}): "))
                if 1 <= choice <= len(datasets):
                    return choice - 1  # 0-based index
                elif choice == len(datasets) + 1:
                    return -1  # İptal
                else:
                    print(f"Lütfen 1-{len(datasets)+1} arasında bir sayı girin.")
            except ValueError:
                print("Lütfen geçerli bir sayı girin.")
    
    def display_existing_datasets(self) -> None:
        """Mevcut dataset'leri göster."""
        datasets = self.get_existing_datasets()
        
        if not datasets:
            print("\nMevcut dataset bulunamadı.")
            return
        
        print("\n" + "="*80)
        print("MEVCUT DATASET'LER".center(80))
        print("="*80)
        
        for dataset in datasets:
            status = "✅" if dataset['has_training_data'] else "❌"
            print(f"\n{status} Dataset{dataset['id']:03d}_{dataset['name']}")
            print(f"   Training: {dataset['file_count']['images_tr']} görüntü, {dataset['file_count']['labels_tr']} etiket")
            print(f"   Test: {dataset['file_count']['images_ts']} görüntü")
            print(f"   Yol: {dataset['path']}")
