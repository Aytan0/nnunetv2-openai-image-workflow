
import os
import shutil
from pathlib import Path
import json
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Optimize Edilmiş Fonksiyonlar (Normalde ayrı bir modülde olurdu) ---

def organize_and_convert(input_folder, nnunet_raw_dir, task_name, task_id):
    input_folder = Path(input_folder)
    nnunet_raw_dir = Path(nnunet_raw_dir)
    dataset_name = f"Dataset{task_id:03d}_{task_name}"
    task_dir = nnunet_raw_dir / dataset_name

    if task_dir.exists():
        logging.warning(f"Dizin zaten var: {task_dir}. İçerik siliniyor.")
        shutil.rmtree(task_dir)

    imagesTr_dir = task_dir / "imagesTr"
    labelsTr_dir = task_dir / "labelsTr"
    imagesTr_dir.mkdir(parents=True, exist_ok=True)
    labelsTr_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"'{input_folder}' içeriği taranıyor...")
    
    # Görüntü ve etiket dosyalarını bul ve kopyala
    # Bu kısım, IYILESTIRMELER.md'deki mantığı uygular
    copied_files = 0
    for item in os.listdir(input_folder):
        item_path = Path(input_folder) / item
        if item_path.is_dir():
             for f in item_path.glob('**/*.nii.gz'):
                # Dosya adlandırma şemasına göre ayırma
                if '_0000.nii.gz' in f.name: # Görüntü dosyası
                    new_name = f.name.replace('_0000', '')
                    shutil.copy(f, imagesTr_dir / new_name)
                    logging.info(f"Kopyalandı (Görüntü): {f.name} -> {imagesTr_dir / new_name}")
                    copied_files += 1
                elif '.nii.gz' in f.name and 'mask' not in f.name: # Etiket dosyası
                    shutil.copy(f, labelsTr_dir / f.name)
                    logging.info(f"Kopyalandı (Etiket): {f.name} -> {labelsTr_dir / f.name}")
                    copied_files += 1

    if copied_files == 0:
        logging.error("Hiçbir .nii.gz dosyası bulunamadı veya kopyalanamadı. Lütfen 'input_folder' yapısını kontrol edin.")
        return False

    _create_dataset_json(task_dir, task_name)

    logging.info(f"Veri kümesi başarıyla {task_dir} konumuna düzenlendi.")
    return True

def _create_dataset_json(task_dir, task_name):
    imagesTr_count = len(list((task_dir / "imagesTr").glob("*.nii.gz")))
    dataset_info = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "cancer": 1},
        "numTraining": imagesTr_count,
        "file_ending": ".nii.gz",
        "name": task_name
    }
    json_path = task_dir / "dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)
    logging.info(f"'dataset.json' oluşturuldu: {json_path}")

# --- Test Senaryosu ---

if __name__ == "__main__":
    # Test parametreleri
    TEST_INPUT_DIR = "/workspace/test_data/raw"
    TEST_NNUNET_RAW_DIR = "/workspace/data/nnUNet_raw_test"
    TEST_TASK_NAME = "TestTask"
    TEST_TASK_ID = 999

    logging.info("--- Test Başlatılıyor ---")
    logging.info(f"Giriş Dizini: {TEST_INPUT_DIR}")
    logging.info(f"Çıkış Dizini: {TEST_NNUNET_RAW_DIR}")

    # Testi çalıştır
    success = organize_and_convert(
        input_folder=TEST_INPUT_DIR,
        nnunet_raw_dir=TEST_NNUNET_RAW_DIR,
        task_name=TEST_TASK_NAME,
        task_id=TEST_TASK_ID
    )

    if success:
        logging.info("--- Test Başarılı ---")
        # Sonucu doğrula (isteğe bağlı)
        expected_dir = Path(TEST_NNUNET_RAW_DIR) / f"Dataset{TEST_TASK_ID:03d}_{TEST_TASK_NAME}"
        logging.info(f"Oluşturulan dizin: {expected_dir}")
        logging.info("İçerik:")
        for root, dirs, files in os.walk(expected_dir):
            for name in files:
                logging.info(os.path.join(root, name))
            for name in dirs:
                logging.info(os.path.join(root, name))
    else:
        logging.error("--- Test Başarısız ---")
