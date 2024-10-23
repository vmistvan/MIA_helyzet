# Semi supervised - Részben felügyelt
## Részben elkészült. Mindeközben engem részben elvitt a férfinátha, de már nem érdemes tovább halogatni a feltöltést.

## Talán a fentiekből kiderült ez úgy nincs kész, ahogy van!
Ha nálam jobban benne vagy a témában, amire azért van esély, akkor persze ebből is tudsz dolgozni, de legalább az install.txt-t olvasd el!

# Részben felügyelt tanulás
## a példaprogram letesztelt változata
## Nagyon fontos, hogy megfelelő sorrendben hajtsd végre az utasításokat!

0. Értelmezd pontosan, amit itt leírok! Amennyiben nem érted, vagy nem működik, nézd meg a hibajegyeket, ha kell, nyiss újat, fel fogom dolgozni. De nagyon fontos, hogy a python verziója megfelelő legyen, és az egyes modulok telepítésre kerüljenek!
1. Környezet előkészítése: Az install.txt-ben lévő modulok telepítését a parancssorokkal kell elsőnek végrehajtanod.
2. Adatok előkészítése: Az install.txt-ben vannak !wget kezdetű sorok, amik a megadott módon a Google Colab rendszerében működiik. Ha windows alatt akarod használni, akkor a kezdő felkiáltójel nélkül fog működni.
3. Adatok összeállítása: 01_assembly_data.py - ezt kell a telepítés és letöltés után elsőnek futtatnod. Mivel ez létrehoz, töröl fájlokat, könyvtárakat, egyszer végrehajtható, az eredeti fájlokat jobb, ha elmented valahova.
4. Betanulás: main_learn.py - minden modellnek először tanulnia kell. ez a fájl 3 feladatot lát el:
   - elkészíti az adatbetöltőt, amivel eléri majd a tréninghez szükséges adatokat;
   - betölti a nyers modellt, amit be akarsz tanítani;
   - végrehajtja a betanítást.
5. 
