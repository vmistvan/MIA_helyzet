# Time Series - Idősor modellek

## Kedves ellenségem ez a modelltípus :) Ne várj ettől a projekttől túl sokat!

Valójában sokkal többet dolgoztam ezzel a modelltípussal, és minden probléma megoldása újabb kérdéseket vetett fel, amiket aztán szintén megoldva egyre inkább elmaradt a modell az elvárásaimtól. Valójában nem is számítottam jobbra. Viszont ebben van minden, amiből lehet tanulni!
Kell tisztítani az adatokat, és a pandas szuper tulajdonságait tudod élesben kipróbálni, szóval adattudóskodhatsz, elmerülhetsz a vizualizációban, van benne __feature engineering__, és predikció. Ja, igen. Ez prediktív és nem regressziós AI. Valójában a legfőbb oka a sikertelenségemnek az volt, hogy próbáltam árfolyamadatokra átírni a modellt, egy nehéz időszakra, amikor
1. folyamatos előre nem látható - azaz nem prediktálható beavatkozás volt az árfolyamba,
2. semmi ciklikussága nem volt az árfolyamadatoknak, és még visszatekintve sem tudok trendet meghatározni.

Ennek ellenére jó volt a hiperparamétereket tekergetni, és nézni, hogy ad vissza jobb eredményt, mekkora időablakokat használjak.

Ha elfogadsz egy tanácsot, akkor ezerszer inkább hőmérsékletadatokra írod át! :)

Valójában ez egy példa hibáinak kijavítása, ezért nincs benne explicit predikció, egyszer talán beledolgozom azt is a példaprogramba, amit én működőnek találtam, vagy megosztom azt a programom, amivel a Bitcoin árfolyamgörbéjét szálaztam. elvesztette titkos jellegét, hisz pont nem lehetett semmire használni :)

Ide is fog egy __install.txt__ kerülni, a futtatáshoz szükséges tudnivalókkal. Gépigénye elenyésző, kisfloppyra (tudod még mi az?) ráférne a modell :)
