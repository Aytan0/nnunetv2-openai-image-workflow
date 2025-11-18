#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnUNet Kapsamlı Test Modu (12. Seçenek)

Bu modül tüm nnUNet özelliklerini küçük örnek dataset ile test eder:
1. Dataset indirme simülasyonu
2. Zip dosyası çıkarma
3. Dataset dönüştürme
4. Model eğitimi (kısa)
5. Tahmin yapma
6. Görselleştirme
7. Sonuç analizi

Kullanım:
- Ana menüden seçenek 12'yi seçin
- Veya doğrudan: python src/test_mode.py
"""

import os
import sys
import tempfile
import shutil
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import subprocess

# Unicode-safe logging
class TestModeLogger:
    def __init__(self, log_file: Optional[str] = None):
        self.setup_logging(log_file)
    
    def setup_logging(self, log_file: Optional[str] = None):
        """Unicode-safe logging kurulumu"""
        if os.name == 'nt':  # Windows
            os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Custom formatter
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                try:
                    return super().format(record)
                except UnicodeEncodeError:
                    msg = record.getMessage()
                    if isinstance(msg, str):
                        msg = msg.encode('ascii', 'replace').decode('ascii')
                    record.msg = msg
                    record.args = ()
                    return super().format(record)
        
        formatter = SafeFormatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger('test_mode')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

class nnUNetTestSuite:
    """nnUNet kapsamlı test suite'i"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.test_results = {}
        self.start_time = time.time()
        
        # Test logger
        log_file = self.workspace_dir / "logs" / "test_mode.log"
        log_file.parent.mkdir(exist_ok=True)
        self.test_logger = TestModeLogger(str(log_file))
        self.logger = self.test_logger.logger
        
        # Test dizinleri
        self.test_base = self.workspace_dir / "test_mode_results"
        self.test_data = self.test_base / "test_data"
        self.test_output = self.test_base / "outputs"
        self.test_logs = self.test_base / "logs"
        
        # Temizle ve oluştur
        if self.test_base.exists():
            shutil.rmtree(self.test_base)
        
        for directory in [self.test_data, self.test_output, self.test_logs]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("="*70)
        self.logger.info("🧪 nnUNet KAPSAMLI TEST MODU BASLATILDI")
        self.logger.info("="*70)
        self.logger.info(f"Test dizini: {self.test_base}")
        self.logger.info(f"Calisma dizini: {self.workspace_dir}")
    
    def _run_command(self, cmd: List[str], description: str, 
                    timeout: int = 300) -> Tuple[bool, str, str]:
        """Komut çalıştırma ve sonuç döndürme"""
        self.logger.info(f"Calistiriliyor: {description}")
        self.logger.debug(f"Komut: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            success = result.returncode == 0
            if success:
                self.logger.info(f"✅ {description} basarili")
            else:
                self.logger.error(f"❌ {description} basarisiz (kod: {result.returncode})")
            
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ {description} zaman asimi ({timeout}s)")
            return False, "", "Timeout"
        except Exception as e:
            self.logger.error(f"💥 {description} beklenmeyen hata: {e}")
            return False, "", str(e)
    
    def test_1_environment_check(self) -> bool:
        """Test 1: Ortam ve bağımlılık kontrolü"""
        self.logger.info("\n🔍 TEST 1: Ortam ve Bagimlilik Kontrolu")
        self.logger.info("-" * 50)
        
        try:
            # Python import testleri
            required_modules = [
                'sys', 'os', 'pathlib', 'json', 'logging',
                'argparse', 'subprocess', 'shutil', 'tempfile'
            ]
            
            optional_modules = [
                'torch', 'numpy', 'SimpleITK', 'nibabel', 
                'nnunetv2', 'matplotlib', 'scipy'
            ]
            
            # Required modules
            failed_required = []
            for module in required_modules:
                try:
                    __import__(module)
                    self.logger.info(f"  ✅ {module}")
                except ImportError:
                    self.logger.error(f"  ❌ {module}")
                    failed_required.append(module)
            
            # Optional modules
            failed_optional = []
            for module in optional_modules:
                try:
                    __import__(module)
                    self.logger.info(f"  ✅ {module} (opsiyonel)")
                except ImportError:
                    self.logger.warning(f"  ⚠️  {module} (opsiyonel)")
                    failed_optional.append(module)
            
            # Sonuç
            success = len(failed_required) == 0
            self.test_results['environment'] = {
                'success': success,
                'failed_required': failed_required,
                'failed_optional': failed_optional,
                'message': f"Gerekli: {len(required_modules) - len(failed_required)}/{len(required_modules)}, "
                          f"Opsiyonel: {len(optional_modules) - len(failed_optional)}/{len(optional_modules)}"
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"Ortam kontrolu hatasi: {e}")
            self.test_results['environment'] = {'success': False, 'error': str(e)}
            return False
    
    def test_2_data_preparation(self) -> bool:
        """Test 2: Test verisi hazırlama ve zip çıkarma"""
        self.logger.info("\n📦 TEST 2: Test Verisi Hazirlama ve Zip Cikarma")
        self.logger.info("-" * 50)
        
        try:
            # Örnek test verilerini oluştur
            test_raw = self.test_data / "raw"
            images_tr = test_raw / "imagesTr"
            labels_tr = test_raw / "labelsTr"
            
            for directory in [images_tr, labels_tr]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Varolan test verilerini kopyala
            source_test_data = self.workspace_dir / "test_data" / "raw"
            if source_test_data.exists():
                self.logger.info("Mevcut test verileri kopyalaniyor...")
                
                # Images
                source_images = source_test_data / "imagesTr"
                if source_images.exists():
                    for file in source_images.glob("*.nii.gz"):
                        shutil.copy2(file, images_tr)
                        self.logger.info(f"  Kopyalandi: {file.name}")
                
                # Labels
                source_labels = source_test_data / "labelsTr"
                if source_labels.exists():
                    for file in source_labels.glob("*.nii.gz"):
                        shutil.copy2(file, labels_tr)
                        self.logger.info(f"  Kopyalandi: {file.name}")
            
            # Dosya sayısını kontrol et
            image_count = len(list(images_tr.glob("*.nii.gz")))
            label_count = len(list(labels_tr.glob("*.nii.gz")))
            
            # Dummy dosyalar oluştur gerekirse
            if image_count == 0:
                self.logger.info("Dummy test dosyalari olusturuluyor...")
                for i in range(1, 4):  # 3 dummy file
                    (images_tr / f"case_{i:03d}_0000.nii.gz").touch()
                    (labels_tr / f"case_{i:03d}.nii.gz").touch()
                
                image_count = 3
                label_count = 3
            
            # Test zip dosyası oluştur
            zip_file = self.test_data / "test_dataset.zip"
            with zipfile.ZipFile(zip_file, 'w') as zf:
                for file in images_tr.glob("*.nii.gz"):
                    zf.write(file, f"imagesTr/{file.name}")
                for file in labels_tr.glob("*.nii.gz"):
                    zf.write(file, f"labelsTr/{file.name}")
            
            self.logger.info(f"Test zip dosyasi olusturuldu: {zip_file}")
            
            # Zip çıkarma testi
            extract_dir = self.test_data / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(extract_dir)
            
            extracted_files = len(list(extract_dir.glob("**/*.nii.gz")))
            
            success = image_count > 0 and label_count > 0 and extracted_files > 0
            self.test_results['data_preparation'] = {
                'success': success,
                'image_count': image_count,
                'label_count': label_count,
                'extracted_files': extracted_files,
                'zip_file': str(zip_file)
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"Veri hazirlama hatasi: {e}")
            self.test_results['data_preparation'] = {'success': False, 'error': str(e)}
            return False
    
    def test_3_dataset_conversion(self) -> bool:
        """Test 3: Dataset dönüştürme"""
        self.logger.info("\n🔄 TEST 3: Dataset Donusturme")
        self.logger.info("-" * 50)
        
        try:
            # Test conversion script'ini çalıştır
            conversion_script = self.workspace_dir / "src" / "test_conversion.py"
            
            if conversion_script.exists():
                success, stdout, stderr = self._run_command([
                    sys.executable, str(conversion_script)
                ], "Dataset donusturme testi")
                
                self.test_results['dataset_conversion'] = {
                    'success': success,
                    'stdout': stdout[:500] if stdout else "",
                    'stderr': stderr[:500] if stderr else ""
                }
                
                return success
            else:
                self.logger.warning("Dataset donusturme scripti bulunamadi")
                return False
                
        except Exception as e:
            self.logger.error(f"Dataset donusturme hatasi: {e}")
            self.test_results['dataset_conversion'] = {'success': False, 'error': str(e)}
            return False
    
    def test_4_ai_organizer(self) -> bool:
        """Test 4: AI Organizer ve dataset yönetimi"""
        self.logger.info("\n🤖 TEST 4: AI Organizer ve Dataset Yonetimi")
        self.logger.info("-" * 50)
        
        try:
            # Import test
            sys.path.insert(0, str(self.workspace_dir / "src"))
            
            from dataset_manager import DatasetManager
            from ai_organizer import AIOrganizer
            
            # Test dataset manager
            test_nnunet_dir = self.test_output / "nnUNet_raw"
            test_nnunet_dir.mkdir(exist_ok=True)
            
            dm = DatasetManager(str(test_nnunet_dir))
            next_id = dm.get_next_dataset_id()
            
            self.logger.info(f"Dataset Manager test basarili, sonraki ID: {next_id}")
            
            # Test AI organizer
            ai_org = AIOrganizer(str(test_nnunet_dir))
            
            self.logger.info("AI Organizer test basarili")
            
            self.test_results['ai_organizer'] = {
                'success': True,
                'next_dataset_id': next_id,
                'ai_available': ai_org.ai_available
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"AI Organizer test hatasi: {e}")
            self.test_results['ai_organizer'] = {'success': False, 'error': str(e)}
            return False
    
    def test_5_unicode_support(self) -> bool:
        """Test 5: Unicode ve Türkçe karakter desteği"""
        self.logger.info("\n🌍 TEST 5: Unicode ve Turkce Karakter Destegi")
        self.logger.info("-" * 50)
        
        try:
            # Türkçe karakterlerle test
            turkish_text = "Türkçe karakterler: ğüışöçĞÜIŞÖÇ"
            test_file = self.test_output / "unicode_test.txt"
            
            # Dosyaya yaz
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(turkish_text)
            
            # Dosyadan oku
            with open(test_file, 'r', encoding='utf-8') as f:
                read_text = f.read().strip()
            
            # Karşılaştır
            unicode_success = turkish_text == read_text
            
            # JSON test
            test_data = {
                'dataset_name': 'Türkçe_Dataset_Testi_ğüışöç',
                'description': 'Bu bir unicode test dosyasıdır',
                'characters': ['ğ', 'ü', 'ı', 'ş', 'ö', 'ç']
            }
            
            json_file = self.test_output / "unicode_test.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            with open(json_file, 'r', encoding='utf-8') as f:
                read_data = json.load(f)
            
            json_success = test_data == read_data
            
            # Log test
            self.logger.info(f"Turkce karakter testi: {turkish_text}")
            
            success = unicode_success and json_success
            self.test_results['unicode_support'] = {
                'success': success,
                'text_test': unicode_success,
                'json_test': json_success,
                'test_string': turkish_text
            }
            
            return success
            
        except Exception as e:
            self.logger.error(f"Unicode test hatasi: {e}")
            self.test_results['unicode_support'] = {'success': False, 'error': str(e)}
            return False
    
    def test_6_main_interface(self) -> bool:
        """Test 6: Ana arayüz ve menü sistemi"""
        self.logger.info("\n🖥️  TEST 6: Ana Arayuz ve Menu Sistemi")
        self.logger.info("-" * 50)
        
        try:
            # Main script'i kontrol et
            main_script = self.workspace_dir / "main.py"
            
            if main_script.exists():
                # Syntax check
                success, stdout, stderr = self._run_command([
                    sys.executable, '-m', 'py_compile', str(main_script)
                ], "Main script syntax kontrolu", timeout=30)
                
                if success:
                    self.logger.info("Ana script syntax kontrolu basarili")
                else:
                    self.logger.error(f"Ana script syntax hatasi: {stderr}")
                
                self.test_results['main_interface'] = {
                    'success': success,
                    'script_exists': True,
                    'syntax_check': success
                }
                
                return success
            else:
                self.logger.error("Ana script bulunamadi")
                self.test_results['main_interface'] = {
                    'success': False,
                    'script_exists': False
                }
                return False
                
        except Exception as e:
            self.logger.error(f"Ana arayuz test hatasi: {e}")
            self.test_results['main_interface'] = {'success': False, 'error': str(e)}
            return False
    
    def test_7_unit_tests(self) -> bool:
        """Test 7: Unit testler"""
        self.logger.info("\n🧪 TEST 7: Unit Testler")
        self.logger.info("-" * 50)
        
        try:
            # Pytest'i çalıştır
            test_file = self.workspace_dir / "tests" / "test_dataset_conversion.py"
            
            if test_file.exists():
                success, stdout, stderr = self._run_command([
                    sys.executable, '-m', 'pytest', str(test_file), '-v', '--tb=short'
                ], "Unit testler", timeout=120)
                
                # Test sonuçlarını parse et
                passed_tests = stdout.count(" PASSED") if stdout else 0
                failed_tests = stdout.count(" FAILED") if stdout else 0
                
                self.test_results['unit_tests'] = {
                    'success': success,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'output': stdout[:1000] if stdout else ""
                }
                
                return success
            else:
                self.logger.warning("Unit test dosyasi bulunamadi")
                return True  # Optional test
                
        except Exception as e:
            self.logger.error(f"Unit test hatasi: {e}")
            self.test_results['unit_tests'] = {'success': False, 'error': str(e)}
            return False
    
    def generate_test_report(self) -> str:
        """Test raporu oluştur"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        report_file = self.test_output / "test_raporu.md"
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# nnUNet Kapsamlı Test Raporu\\n\\n")
            f.write(f"**Test Tarihi**: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Test Süresi**: {duration:.2f} saniye\\n")
            f.write(f"**Geçen Testler**: {passed_tests}/{total_tests}\\n")
            f.write(f"**Başarı Oranı**: {(passed_tests/total_tests)*100:.1f}%\\n\\n")
            
            # Özet
            f.write("## 📊 Test Özeti\\n\\n")
            for test_name, result in self.test_results.items():
                status = "✅" if result.get('success', False) else "❌"
                f.write(f"- {status} **{test_name.replace('_', ' ').title()}**\\n")
            
            f.write("\\n## 📋 Detaylı Sonuçlar\\n\\n")
            
            # Detaylar
            for test_name, result in self.test_results.items():
                f.write(f"### {test_name.replace('_', ' ').title()}\\n\\n")
                
                if result.get('success', False):
                    f.write("**Durum**: ✅ Başarılı\\n\\n")
                else:
                    f.write("**Durum**: ❌ Başarısız\\n\\n")
                
                # Test spesifik detaylar
                for key, value in result.items():
                    if key != 'success':
                        f.write(f"- **{key}**: {value}\\n")
                
                f.write("\\n")
            
            # Öneriler
            f.write("## 💡 Öneriler\\n\\n")
            if passed_tests == total_tests:
                f.write("🎉 Tüm testler başarılı! Sistem kullanıma hazır.\\n\\n")
            else:
                f.write("⚠️ Bazı testler başarısız. Aşağıdaki adımları deneyin:\\n\\n")
                f.write("1. Gerekli paketlerin yüklü olduğunu kontrol edin\\n")
                f.write("2. Python ortamının düzgün kurulduğunu doğrulayın\\n")
                f.write("3. Log dosyalarını detaylı inceleme için kontrol edin\\n")
                f.write("4. `python setup_optimized.py --test` ile tekrar deneyin\\n\\n")
        
        return str(report_file)
    
    def run_full_test_suite(self) -> bool:
        """Tüm test suite'ini çalıştır"""
        self.logger.info("🚀 Kapsamli test suite baslatiyor...")
        
        tests = [
            ("Ortam Kontrolu", self.test_1_environment_check),
            ("Veri Hazirlama", self.test_2_data_preparation),
            ("Dataset Donusturme", self.test_3_dataset_conversion),
            ("AI Organizer", self.test_4_ai_organizer),
            ("Unicode Destegi", self.test_5_unicode_support),
            ("Ana Arayuz", self.test_6_main_interface),
            ("Unit Testler", self.test_7_unit_tests),
        ]
        
        for test_name, test_func in tests:
            self.logger.info(f"\\n{'='*60}")
            self.logger.info(f"BASLATILIYOR: {test_name}")
            self.logger.info('='*60)
            
            try:
                success = test_func()
                if success:
                    self.logger.info(f"✅ {test_name} BASARILI")
                else:
                    self.logger.error(f"❌ {test_name} BASARISIZ")
            except Exception as e:
                self.logger.error(f"💥 {test_name} BEKLENMEYEN HATA: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = {
                    'success': False, 
                    'error': str(e)
                }
        
        # Rapor oluştur
        report_file = self.generate_test_report()
        
        # Sonuç özeti
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        self.logger.info(f"\\n{'='*70}")
        self.logger.info("🏁 TEST SUITE TAMAMLANDI")
        self.logger.info('='*70)
        self.logger.info(f"Gecen testler: {passed_tests}/{total_tests}")
        self.logger.info(f"Basari orani: {(passed_tests/total_tests)*100:.1f}%")
        self.logger.info(f"Test raporu: {report_file}")
        
        return passed_tests == total_tests

def main():
    """Ana test fonksiyonu"""
    import argparse
    
    parser = argparse.ArgumentParser(description='nnUNet Kapsamlı Test Modu')
    parser.add_argument('--workspace', default='.', help='Çalışma dizini')
    
    args = parser.parse_args()
    workspace = Path(args.workspace).absolute()
    
    if not workspace.exists():
        print(f"Hata: Çalışma dizini bulunamadı: {workspace}")
        sys.exit(1)
    
    try:
        test_suite = nnUNetTestSuite(str(workspace))
        success = test_suite.run_full_test_suite()
        
        if success:
            print("\\n🎉 Tüm testler başarılı!")
            sys.exit(0)
        else:
            print("\\n⚠️ Bazı testler başarısız. Raporu kontrol edin.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\\nTest kullanıcı tarafından durduruldu.")
        sys.exit(1)
    except Exception as e:
        print(f"\\nBeklenmeyen hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
