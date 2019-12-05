
from PyQt5.QtWidgets import  QApplication
from status.status import status

if __name__ == "__main__":
    
    import sys
    app = QApplication(sys.argv)
    ui = status()
    ui.show()
    sys.exit(app.exec_())

#ME AYTO EGINE ENA PROGRAMMA
#pyinstaller --onefile --windowed STATUS.py
