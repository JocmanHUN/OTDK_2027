# Architektúra elvek

## Áttekintés
A rendszer **réteges architektúrát** követ, amely világosan elválasztja az egyes felelősségi köröket.  
Fő cél: átláthatóság, könnyű tesztelhetőség, bővíthetőség.

---

## Rétegek
1. **UI (User Interface)**  
   - Tkinter alapú GUI.  
   - Feladata: eredmények megjelenítése, felhasználói input kezelése.  
   - Nem tartalmaz üzleti logikát, csak hívja az Application Service réteget.  

2. **Application Services**  
   - A felhasználói kérések kiszolgálása.  
   - Példa: „Lekérni az aktuális napi mérkőzéseket és futtatni a modelleket”.  
   - Orchestrációs szerep: több domain modult kapcsol össze.  
   - Itt történik a session menedzsment (mentés, betöltés).  

3. **Domain (Üzleti logika)**  
   - Modellek implementációja (Poisson, Monte Carlo, Elo, Logistic Regression, Veto, Balance).  
   - Fogadási stratégiák (Flat, Martingale, Fibonacci, Value, Kelly).  
   - Predikció és ROI számítás.  
   - Teljesen UI-független.  

4. **Infrastructure (Infrastruktúra)**  
   - API-FOOTBALL adapter (külön modulban, pl. `src/infrastructure/api_football_adapter.py`).  
   - Adatbázis-kezelés (SQLite/JSON/CSV export).  
   - Logging és konfiguráció.  
   - Cache kezelés (API hívások minimalizálása).  

---

## Hibakezelési elvek
- **„Skip & log” elv**: ha egy meccs vagy adat nem elérhető, azt a rendszer *kihagyja* a feldolgozásból, de **logolja az okot** (pl. „nincs odds”, „hibás API válasz”).  
- Minden kivételhez tartozik log-szint: `INFO`, `WARNING`, `ERROR`.  
- Session logok mentése (egy `.log` fájl per session).  

---

## Aszinkronitás terve
- Python `threading` + `queue.Queue` alapú párhuzamos feldolgozás.  
- API hívások és modellek futtatása külön szálakban → UI nem akad le.  
- **Producer–Consumer minta**:  
  - Producer: API hívásokat végző szál.  
  - Consumer: predikciót és stratégiát futtató szál.  
- Queue biztosítja az adatátvitelt a szálak között.  

---

## Példa folyamat (napi futás)
1. UI: felhasználó megnyomja a „Napi mérkőzések betöltése” gombot.  
2. Application Services: meghívja az API adaptert, amely szálban lekéri a mérkőzéseket.  
3. Domain: minden modell lefut, eredmények Queue-n át kerülnek a főszálhoz.  
4. UI: progress bar jelzi az állapotot, végén táblázatban megjelennek az eredmények.  
