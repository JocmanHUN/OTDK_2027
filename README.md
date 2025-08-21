# OTDK 2027 – Sports Betting Analysis

**Dolgozat címe:**  
*A sportfogadás nyereségességének vizsgálata valószínűségi modellek és fogadási stratégiák együttes alkalmazásával*

---

## 🎯 Projekt célja
A rendszer célja, hogy különböző predikciós modellek és fogadási stratégiák segítségével vizsgálja, lehetséges-e hosszú távon nyereséges sportfogadási módszert találni.

A program:
- futballmérkőzések 1X2 kimeneteire készít előrejelzéseket,
- ezek alapján különféle stratégiákat alkalmaz,
- statisztikailag kiértékeli a profitabilitást.

---

## ⚙️ Alkalmazott modellek
- Poisson modell  
- Monte Carlo szimuláció  
- Elo modell  
- Logisztikus regresszió  
- Veto modell (saját fejlesztés)  
- Balance modell (saját fejlesztés)  

---

## 🧮 Stratégiák
- Flat Betting  
- Martingale  
- Fibonacci  
- Value Betting  
- Kelly Criterion  

---

## 📊 Fő funkciók
- API-FOOTBALL integráció (meccsadatok + oddsok)  
- Modellek futtatása napi/ heti mérkőzésekre  
- Odds shopping és várható érték (EV) számítás  
- Stratégiák szerinti bankroll-szimuláció  
- Mérkőzéscsoportok generálása és kiértékelése  

---

## 🛠️ Tech stack
- **Python 3.11+**
- Tkinter (UI)  
- Pandas, Numpy, Matplotlib  
- Pytest (tesztelés)  
- Ruff, Black, Isort, Mypy (tooling)  

---

## 📂 Mappastruktúra (terv)
