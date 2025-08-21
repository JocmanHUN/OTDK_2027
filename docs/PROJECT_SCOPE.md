# Project Scope

## Dolgozat címe
**A sportfogadás nyereségességének vizsgálata valószínűségi modellek és fogadási stratégiák együttes alkalmazásával**

---

## Projekt célja
A rendszer célja, hogy különböző predikciós modellek és fogadási stratégiák segítségével vizsgálja, lehetséges-e hosszú távon nyereséges sportfogadási módszert találni.  

A program futballmérkőzések 1X2 kimeneteire készít előrejelzéseket, majd ezek alapján különféle stratégiákat alkalmaz, és statisztikailag kiértékeli a profitabilitást.

---

## Alkalmazott modellek
- Poisson modell  
- Monte Carlo szimuláció  
- Elo modell  
- Logisztikus regresszió  
- Veto modell *(saját fejlesztés)*  
- Balance modell *(saját fejlesztés)*  

---

## Alkalmazott stratégiák
- Flat Betting  
- Martingale  
- Fibonacci  
- Value Betting  
- Kelly Criterion  

---

## Rendszer működése
1. **Ligák kiválasztása**: a program indulásakor lekérdezi az aktuális szezon azon ligáit az API-FOOTBALL-ból, amelyekhez elérhetők oddsok és statisztikák.  
2. **Napi/Heti mérkőzéslista**: lekérjük a napi (később heti) mérkőzéskínálatot.  
3. **Használható meccsek szűrése**: csak azok a mérkőzések kerülnek be, ahol legalább egy fogadóiroda kínál 1X2 oddsot.  
4. **Előzményadatok lekérése**: a használható mérkőzésekhez statisztikák és előzmények betöltése (pl. formák, gólátlagok, ELO pontszámok).  
5. **Predikciók készítése**: ha minden szükséges adat elérhető, az összes modell lefuttatja a saját előrejelzését (1/X/2 valószínűségek).  
6. **Megjelenítendő mérkőzések listája**: a sikeresen megtippelt meccsek oddsokkal és predikciókkal együtt kerülnek a felhasználó elé.  
7. **Szűrés és rendezés**: lehetőség van szűrni fogadóiroda szerint, odds shoppingra (legjobb odds kiválasztása), várható érték (EV) vagy modell alapján történő rendezésre.  
8. **Stratégiák alkalmazása**: a kiválasztott mérkőzésekre a felhasználó különböző stratégiákat próbálhat ki (pl. Kelly alapján kiszámolt tét).  
9. **Mérkőzéscsoportok generálása**:  
   - *Valós csoportok*: a felhasználó saját válogatása alapján.  
   - *Optimalizált csoportok*: a rendszer automatikusan kiválasztja a legjobb kombinációt.  
   - *Mesterséges csoportok*: pl. 10 000 véletlen 25-meccses csoport generálása, amellyel vizsgálható egy modell hosszú távú teljesítménye.  

---

## Out of Scope (nem célok)
- Valós pénzes fogadás kezelése.  
- Mobil- vagy webes felület (első körben Tkinter alapú GUI elég).  
- Minden fogadási piac támogatása (csak 1X2).  
- Valós idejű bot vagy odds arbitrázs rendszer.  

---

## Mérőszámok
- **Adatkezelés**: napi 200–500 mérkőzés feldolgozása.  
- **Futási idő**: teljes napi feldolgozás ≤ 1 perc (cache + párhuzamos futtatás).  
- **Kimenetek**: predikciós pontosság, várható érték, ROI, bankroll görbék, szimulációs statisztikák.  
- **Reprodukálhatóság**: minden futás session exportban menthető, így később újrafuttatható.  
