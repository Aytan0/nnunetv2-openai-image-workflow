#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nnUNet Otomatik Kurulum ve Test Sistemi
Optimize edilmis versiyon - Tum sistem bagimliliklari dahil

Bu script sunlari yapar:
1. Sistem gereksinimlerini kontrol eder
2. Python ortamini kurar
3. nnUNet ve bagimliliklarini yukler
4. Test verilerini hazirlar
5. Unicode destegini saglar
6. Tam sistem testini yapar
"""

import os
import sys
import argparse
import subprocess
import platform
import logging
import shutil
from pathlib import Path
from typing import List

# Unicode-safe logging konfigurasyonu
class UnicodeFormatter(logging.Formatter):
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

def setup_logging():
    """Unicode-safe logging kurulumu"""
    if os.name == 'nt':  # Windows
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    formatter = UnicodeFormatter('%(asctime)s - %(levelname)s - %(message)s')
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    
    return logging.getLogger('setup_nnunet')

logger = setup_logging()

class OptimizedSetupManager:
    """Optimize edilmis nnUNet kurulum yoneticisi"""
    
    def __init__(self, 
                 env_name: str = 'nnunetv2',
                 python_version: str = '3.10',
                 test_mode: bool = False):
        self.env_name = env_name
        self.python_version = python_version
        self.test_mode = test_mode
        self.is_windows = platform.system() == "Windows"
        
        # Paths
        self.script_dir = Path(__file__).parent.absolute()
        self.data_dir = self.script_dir / "data"
        self.test_data_dir = self.script_dir / "test_data"
        self.logs_dir = self.script_dir / "logs"
        self.requirements_file = self.script_dir / "requirements.txt"
        
        # Create necessary directories
        for directory in [self.data_dir, self.test_data_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info("="*60)
        logger.info("nnUNet Otomatik Kurulum Sistemi Baslatildi")
        logger.info("="*60)
        logger.info(f"Isletim Sistemi: {platform.system()} {platform.release()}")
        logger.info(f"Python Surumu: {sys.version}")
        logger.info(f"Ortam Adi: {self.env_name}")
        logger.info(f"Test Modu: {'Aktif' if self.test_mode else 'Pasif'}")
    
    def _run_command(self, cmd: List[str], description: str, 
                    check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Komut calistirma wrapper'i"""
        logger.info(f"{description}: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                text=True,
                check=check,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.stdout and capture_output:
                logger.debug(f"Stdout: {result.stdout[:500]}...")
            if result.stderr and capture_output:
                logger.warning(f"Stderr: {result.stderr[:500]}...")
                
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Komut basarisiz: {description}")
            logger.error(f"Hata kodu: {e.returncode}")
            if e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            raise
    
    def check_system_requirements(self) -> bool:
        """Sistem gereksinimlerini kontrol et"""
        logger.info("Sistem gereksinimleri kontrol ediliyor...")
        
        requirements_met = True
        
        # Python version check - FIX: Tuple karsilastirmasi
        current_version = (sys.version_info.major, sys.version_info.minor)
        required_version = (3, 9)
        
        if current_version < required_version:
            logger.error(f"Python 3.9+ gerekli, mevcut: {current_version[0]}.{current_version[1]}")
            requirements_met = False
        else:
            logger.info(f"Python surumu uygun: {current_version[0]}.{current_version[1]}")
        
        # Disk space check
        disk_usage = shutil.disk_usage(self.script_dir)
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 10:
            logger.warning(f"Disk alani az: {free_gb:.1f}GB bos alan")
        else:
            logger.info(f"Disk alani yeterli: {free_gb:.1f}GB bos alan")
        
        return requirements_met
    
    def setup_python_environment(self) -> bool:
        """Python ortamini kur"""
        logger.info("Python ortami kuruluyor...")
        
        try:
            # Check if conda is available
            conda_cmd = None
            for cmd in ['conda', 'mamba']:
                try:
                    self._run_command([cmd, '--version'], f"{cmd} version check")
                    conda_cmd = cmd
                    logger.info(f"{cmd} bulundu, kullanilacak")
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if conda_cmd:
                # Create conda environment
                logger.info(f"Conda ortami olusturuluyor: {self.env_name}")
                try:
                    self._run_command([
                        conda_cmd, 'create', '-n', self.env_name, 
                        f'python={self.python_version}', '-y'
                    ], "Conda ortami olusturma")
                except subprocess.CalledProcessError:
                    logger.warning(f"Ortam zaten mevcut: {self.env_name}")
                
                logger.info("Conda ortami basariyla olusturuldu")
            else:
                # Use venv
                logger.info("Conda bulunamadi, venv kullaniliyor")
                venv_path = self.script_dir / f".venv_{self.env_name}"
                
                if not venv_path.exists():
                    self._run_command([
                        sys.executable, '-m', 'venv', str(venv_path)
                    ], "Virtual environment olusturma")
                    logger.info(f"Virtual environment olusturuldu: {venv_path}")
                else:
                    logger.info(f"Virtual environment zaten mevcut: {venv_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Python ortami kurulumu basarisiz: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Bagimliliklari yukle"""
        logger.info("Bagimliliklar yukleniyor...")
        
        try:
            # Upgrade pip first
            logger.info("pip guncelleniyor...")
            self._run_command([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], "pip guncelleme", capture_output=False)
            
            # Core dependencies
            core_packages = [
                'torch',
                'torchvision',
                'numpy',
                'scipy',
                'scikit-image',
                'SimpleITK',
                'nibabel',
                'nnunetv2',
                'pandas',
                'matplotlib',
                'tqdm',
                'python-dotenv',
                'synapseclient',
                'openai',
            ]
            
            # Install packages
            pip_cmd = [sys.executable, '-m', 'pip', 'install']
            
            logger.info(f"Paketler yukleniyor: {core_packages}")
            self._run_command(
                pip_cmd + core_packages,
                "Paket kurulumu",
                capture_output=False
            )
            
            # Install from requirements.txt if exists
            if self.requirements_file.exists():
                logger.info("requirements.txt den ek paketler yukleniyor...")
                self._run_command(
                    pip_cmd + ['-r', str(self.requirements_file)],
                    "Requirements.txt kurulumu",
                    capture_output=False
                )
            
            logger.info("Bagimliliklar basariyla yuklendi")
            return True
            
        except Exception as e:
            logger.error(f"Bagimlilik kurulumu basarisiz: {e}")
            return False
    
    def setup_nnunet_environment(self) -> bool:
        """nnUNet ortam degiskenlerini ayarla"""
        logger.info("nnUNet ortam degiskenleri ayarlaniyor...")
        
        try:
            # nnUNet directories
            nnunet_base = self.data_dir / "nnUNet"
            nnunet_raw = nnunet_base / "nnUNet_raw"
            nnunet_preprocessed = nnunet_base / "nnUNet_preprocessed"
            nnunet_results = nnunet_base / "nnUNet_results"
            
            # Create directories
            for directory in [nnunet_raw, nnunet_preprocessed, nnunet_results]:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Dizin olusturuldu: {directory}")
            
            # Set environment variables
            env_vars = {
                'nnUNet_raw': str(nnunet_raw),
                'nnUNet_preprocessed': str(nnunet_preprocessed),
                'nnUNet_results': str(nnunet_results),
                'PYTHONIOENCODING': 'utf-8'
            }
            
            # Write to .env file
            env_file = self.script_dir / '.env'
            with open(env_file, 'w', encoding='utf-8') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
                    os.environ[key] = value
                    logger.info(f"Ortam degiskeni ayarlandi: {key}={value}")
            
            logger.info(f"Ortam degiskenleri .env dosyasina kaydedildi: {env_file}")
            return True
            
        except Exception as e:
            logger.error(f"nnUNet ortam kurulumu basarisiz: {e}")
            return False
    
    def prepare_test_data(self) -> bool:
        """Test verilerini hazirla"""
        logger.info("Test verileri hazirlaniyor...")
        
        try:
            # Create data/raw directory
            raw_dir = self.data_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a .gitkeep file
            gitkeep = raw_dir / ".gitkeep"
            gitkeep.touch()
            
            logger.info(f"data/raw klasoru olusturuldu: {raw_dir}")
            logger.info("Test verileri icin bu klasore dataset zip dosyalarini ekleyin")
            
            return True
            
        except Exception as e:
            logger.error(f"Test verisi hazirliginda hata: {e}")
            return False
    
    def run_system_tests(self) -> bool:
        """Sistem testlerini calistir"""
        logger.info("Sistem testleri calistiriliyor...")
        
        try:
            # Import test
            logger.info("Import testleri...")
            test_imports = [
                'torch',
                'nnunetv2',
                'numpy',
                'SimpleITK',
                'nibabel'
            ]
            
            for module in test_imports:
                try:
                    __import__(module)
                    logger.info(f"OK {module} basariyla import edildi")
                except ImportError as e:
                    logger.error(f"HATA {module} import hatasi: {e}")
                    return False
            
            logger.info("Tum import testleri basarili")
            return True
            
        except Exception as e:
            logger.error(f"Sistem testlerinde hata: {e}")
            return False
    
    def create_runner_script(self) -> bool:
        """Ana calistirma script'ini olustur"""
        logger.info("Calistirma scripti olusturuluyor...")
        
        try:
            runner_content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
nnUNet Ana Calistirici Script
\"\"\"

import os
import sys
from pathlib import Path

# Unicode support
if os.name == 'nt':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'src'))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(script_dir / '.env')
except ImportError:
    pass

# Import and run main
if __name__ == "__main__":
    try:
        from main import main
        main()
    except KeyboardInterrupt:
        print("\\nProgram kullanici tarafindan durduruldu.")
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
"""
            
            runner_file = self.script_dir / "run_nnunet.py"
            with open(runner_file, 'w', encoding='utf-8') as f:
                f.write(runner_content)
            
            # Make executable on Unix systems
            if not self.is_windows:
                os.chmod(runner_file, 0o755)
            
            logger.info(f"Calistirma scripti olusturuldu: {runner_file}")
            return True
            
        except Exception as e:
            logger.error(f"Calistirma scripti olusturma hatasi: {e}")
            return False
    
    def generate_installation_report(self, success: bool) -> None:
        """Kurulum raporu olustur"""
        logger.info("Kurulum raporu olusturuluyor...")
        
        report_file = self.script_dir / "kurulum_raporu.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("nnUNet Kurulum Raporu\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Durum: {'BASARILI' if success else 'BASARISIZ'}\n")
            f.write(f"Isletim Sistemi: {platform.system()} {platform.release()}\n")
            f.write(f"Python Surumu: {sys.version}\n\n")
            
            if success:
                f.write("Basariyla Kurulan Bilesenler:\n")
                f.write("- Python ortami\n")
                f.write("- nnUNet ve bagimliliklari\n")
                f.write("- Test verileri\n")
                f.write("- Unicode destegi\n\n")
                
                f.write("Kullanim:\n")
                f.write(f"conda activate {self.env_name}\n")
                f.write("python run_nnunet.py\n\n")
                
                f.write("VEYA dogrudan:\n")
                f.write("python run_nnunet.py\n\n")
                
                f.write("Dizin Yapisi:\n")
                f.write("data/nnUNet/nnUNet_raw/     # Ham veriler\n")
                f.write("data/nnUNet/nnUNet_results/ # Sonuclar\n")
                f.write("data/raw/                   # Zip dosyalari\n")
                f.write("logs/                       # Log dosyalari\n")
            else:
                f.write("Kurulum sirasinda hatalar olustu.\n")
                f.write("Lutfen log dosyalarini kontrol edin.\n")
        
        logger.info(f"Kurulum raporu olusturuldu: {report_file}")
    
    def full_setup(self) -> bool:
        """Tam kurulum islemi"""
        logger.info("Tam kurulum baslatiyor...")
        
        steps = [
            ("Sistem gereksinimleri", self.check_system_requirements),
            ("Python ortami", self.setup_python_environment),
            ("Bagimliliklar", self.install_dependencies),
            ("nnUNet ortami", self.setup_nnunet_environment),
            ("Test verileri", self.prepare_test_data),
            ("Calistirma scripti", self.create_runner_script),
        ]
        
        if self.test_mode:
            steps.append(("Sistem testleri", self.run_system_tests))
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"ADIM: {step_name}")
            logger.info('='*50)
            
            try:
                success = step_func()
                if success:
                    logger.info(f"OK {step_name} basarili")
                else:
                    logger.error(f"HATA {step_name} basarisiz")
                    return False
            except Exception as e:
                logger.error(f"HATA {step_name} hatasi: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        logger.info(f"\n{'='*60}")
        logger.info("nnUNet kurulumu basariyla tamamlandi!")
        logger.info('='*60)
        
        return True

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='nnUNet Otomatik Kurulum Sistemi')
    parser.add_argument('--env-name', default='nnunetv2', help='Conda/venv ortam adi')
    parser.add_argument('--python-version', default='3.10', help='Python surumu')
    parser.add_argument('--test', action='store_true', help='Test modunu aktif et')
    parser.add_argument('--quick', action='store_true', help='Hizli kurulum (test yok)')
    
    args = parser.parse_args()
    
    setup_manager = OptimizedSetupManager(
        env_name=args.env_name,
        python_version=args.python_version,
        test_mode=args.test and not args.quick
    )
    
    try:
        success = setup_manager.full_setup()
        setup_manager.generate_installation_report(success)
        
        if success:
            logger.info("\nKurulum basarili! 'python run_nnunet.py' ile calistirin.")
            sys.exit(0)
        else:
            logger.error("\nKurulum basarisiz! Raporu kontrol edin.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nKurulum kullanici tarafindan durduruldu.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nBeklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
